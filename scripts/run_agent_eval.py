"""Run the agent evaluation suite offline.

    python scripts/run_agent_eval.py --policy reference
    python scripts/run_agent_eval.py --policy naive --out report.json
    python scripts/run_agent_eval.py --write-baseline

Policies are the scripted reference and three negative controls; none calls a
model, a provider or the network. A real model is a model factory handed to
``app.core.agentic.evalkit.harness.run_suite`` by a reviewed script that has
provider approval and a budget (docs/agentic-architecture.md).
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.agentic.evalkit.harness import Report, run_suite  # noqa: E402
from app.core.agentic.evalkit.policies import POLICIES  # noqa: E402
from app.core.agentic.evalkit.suite import load_suite  # noqa: E402

BASELINE = ROOT / "data" / "agent_eval" / "v1" / "baseline.json"


def summary(report: Report) -> dict:
    return {
        "policy": report.policy,
        "pass_rate": report.pass_rate,
        "mean": report.mean,
        "by_category": report.by_category,
        "report_digest": report.digest(),
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=sorted(POLICIES), default="reference")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()
    suite = load_suite()
    if args.write_baseline:
        rows = []
        for name in sorted(POLICIES):
            report, _ = await run_suite(suite, POLICIES[name], name)
            rows.append(summary(report))
        BASELINE.write_text(
            json.dumps({"suite_sha256": suite.sha256, "policies": rows}, indent=1) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"wrote {BASELINE}")
        return 0
    report, _ = await run_suite(suite, POLICIES[args.policy], args.policy)
    text = report.model_dump_json(indent=1)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(json.dumps(summary(report), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
