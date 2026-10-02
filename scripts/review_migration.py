import ast
import sys
from pathlib import Path

FORBIDDEN = [
    "drop_table(",
    "drop_column(",
]

versions = Path("alembic/versions")

errors = []


def upgrade_source(text: str) -> str:
    # Only upgrade() runs on deploy; downgrade() is expected to drop what
    # upgrade() created, so it is deliberately not checked.
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == "upgrade":
            return ast.get_source_segment(text, node) or ""
    return ""


for file in versions.glob("*.py"):
    text = upgrade_source(file.read_text())

    for rule in FORBIDDEN:
        if rule in text:
            errors.append(f"{file.name}: upgrade() contains {rule}")

if errors:
    print("\n".join(errors))
    sys.exit(1)

print("Migration review passed")
