import sys
from pathlib import Path

FORBIDDEN = [
    "drop_table(",
    "drop_column(",
]

versions = Path("alembic/versions")

errors = []

for file in versions.glob("*.py"):
    text = file.read_text()

    for rule in FORBIDDEN:
        if rule in text:
            errors.append(f"{file.name}: contains {rule}")

if errors:
    print("\n".join(errors))
    sys.exit(1)

print("Migration review passed")
