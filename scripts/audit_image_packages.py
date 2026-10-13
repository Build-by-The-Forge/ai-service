"""Audit the Python packages installed in a built AI-service image.

    python scripts/audit_image_packages.py <image>

Lists the image's installed distributions (`pip freeze --all` inside it)
and runs pip-audit against exactly those versions. Fails on any advisory
that is not in security/accepted-image-advisories.txt, and on any accepted
entry that no longer matches, so the list cannot outlive its reason.

Needs docker and pip-audit on the host, and network access for the
advisory database.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCEPTED_FILE = ROOT / "security" / "accepted-image-advisories.txt"


def accepted_advisories() -> dict[str, str]:
    """Advisory id -> package, from lines `<id> <package>  # reason`."""
    accepted = {}
    for line in ACCEPTED_FILE.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            advisory, package = line.split()
            accepted[advisory] = package.lower()
    return accepted


def installed(image: str) -> str:
    return subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--entrypoint", "pip", image]
        + ["freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def found_advisories(freeze: str) -> set[tuple[str, str, str]]:
    with tempfile.TemporaryDirectory() as tmp:
        requirements = Path(tmp) / "image.txt"
        requirements.write_text(freeze, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "pip_audit", "--progress-spinner=off", "--format=json"]
            + ["--no-deps", "--disable-pip", "-r", str(requirements)],
            capture_output=True,
            text=True,
        )
    # Exit 1 means "vulnerabilities found"; anything else without JSON is an error.
    if result.returncode not in (0, 1) or not result.stdout.strip():
        sys.exit(f"pip-audit failed ({result.returncode}):\n{result.stderr}")
    return {
        (vuln["id"], dep["name"].lower(), dep["version"])
        for dep in json.loads(result.stdout)["dependencies"]
        for vuln in dep.get("vulns", [])
    }


def main(image: str) -> int:
    freeze = installed(image)
    print(f"image: {image}: {len(freeze.splitlines())} installed distributions")
    found = found_advisories(freeze)
    accepted = accepted_advisories()

    unaccepted = sorted(f for f in found if accepted.get(f[0]) != f[1])
    stale = sorted(set(accepted.items()) - {(a, p) for a, p, _ in found})
    for advisory, package, version in sorted(found):
        state = "unaccepted" if (advisory, package, version) in unaccepted else "accepted"
        print(f"{state}: {advisory} {package} {version}")
    for advisory, package in stale:
        print(f"stale acceptance (no longer found): {advisory} {package}")
    if unaccepted or stale:
        return 1
    print("OK: every advisory in the image is a reviewed acceptance")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <image>")
    sys.exit(main(sys.argv[1]))
