"""A small JSON-Schema subset, for tool arguments and results.

Portable by design: pure Python, no imports from the rest of the app, so the
same schemas and the same checks can run in the server's runtime and in an
on-device one (a port of this file is the whole validator).

It supports the keywords tool contracts use and refuses the rest. A schema that
uses a keyword this module does not enforce (``anyOf``, ``$ref``, ``pattern``
...) is rejected by ``check_schema`` instead of being half-enforced: a schema
that looks stricter than it is would be worse than no schema.

Validation errors name the path of the offending value and the rule it broke,
never the value itself: a value can be a student's text.
"""

from __future__ import annotations

import math
from typing import Any

MAX_DEPTH = 8
MAX_SCHEMA_NODES = 400

# Keywords that validate, and so must be understood.
_VALIDATING = frozenset(
    {
        "type",
        "enum",
        "const",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
    }
)
# Keywords that only describe.
_ANNOTATIONS = frozenset({"description", "title", "default", "examples", "$schema"})
_TYPES = frozenset({"string", "integer", "number", "boolean", "object", "array", "null"})


class SchemaError(ValueError):
    """A schema this module can't enforce, or that is malformed."""


def check_schema(schema: Any, *, _depth: int = 0, _nodes: list[int] | None = None) -> None:
    """Raise SchemaError unless every keyword in the schema is one that
    validate() enforces, and the schema is bounded."""
    nodes = _nodes if _nodes is not None else [0]
    nodes[0] += 1
    if nodes[0] > MAX_SCHEMA_NODES:
        raise SchemaError("schema is too large")
    if _depth > MAX_DEPTH:
        raise SchemaError("schema is nested too deeply")
    if not isinstance(schema, dict):
        raise SchemaError("a schema is an object")
    for keyword in schema:
        if keyword not in _VALIDATING and keyword not in _ANNOTATIONS:
            raise SchemaError(f"unsupported schema keyword: {keyword}")
    declared = schema.get("type")
    if declared is not None:
        names = declared if isinstance(declared, list) else [declared]
        if not names or any(not isinstance(n, str) or n not in _TYPES for n in names):
            raise SchemaError("unknown type")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise SchemaError("enum is a non-empty list")
    if "required" in schema:
        required = schema["required"]
        if not isinstance(required, list) or any(not isinstance(r, str) for r in required):
            raise SchemaError("required is a list of names")
    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, dict):
            raise SchemaError("properties is an object")
        for sub in properties.values():
            check_schema(sub, _depth=_depth + 1, _nodes=nodes)
    extra = schema.get("additionalProperties")
    if extra is not None and not isinstance(extra, bool):
        check_schema(extra, _depth=_depth + 1, _nodes=nodes)
    if "items" in schema:
        check_schema(schema["items"], _depth=_depth + 1, _nodes=nodes)
    for keyword in ("minItems", "maxItems", "minLength", "maxLength"):
        if keyword in schema and (
            not _is_int(schema[keyword]) or schema[keyword] < 0  # type: ignore[operator]
        ):
            raise SchemaError(f"{keyword} is a non-negative integer")
    for keyword in ("minimum", "maximum"):
        if keyword in schema and not _is_number(schema[keyword]):
            raise SchemaError(f"{keyword} is a number")


def validate(schema: dict[str, Any], value: Any) -> list[str]:
    """The problems with value, as ``path: rule`` strings; empty when it is
    valid. The schema is assumed to have passed check_schema."""
    problems: list[str] = []
    _validate(schema, value, "$", 0, problems)
    return problems


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, int)


def _matches(name: str, value: Any) -> bool:
    if name == "string":
        return isinstance(value, str)
    if name == "integer":
        return _is_int(value)
    if name == "number":
        return _is_number(value)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    return value is None  # "null"


def _validate(schema: dict[str, Any], value: Any, path: str, depth: int, out: list[str]) -> None:
    if depth > MAX_DEPTH + 4:  # the value, not the schema, is too deep
        out.append(f"{path}: nested too deeply")
        return
    declared = schema.get("type")
    if declared is not None:
        names = declared if isinstance(declared, list) else [declared]
        if not any(_matches(n, value) for n in names):
            out.append(f"{path}: wrong type")
            return
    if "enum" in schema and not any(_same(value, option) for option in schema["enum"]):
        out.append(f"{path}: not an allowed value")
    if "const" in schema and not _same(value, schema["const"]):
        out.append(f"{path}: not the required value")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            out.append(f"{path}: too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            out.append(f"{path}: too long")
    if _is_number(value):
        if "minimum" in schema and value < schema["minimum"]:
            out.append(f"{path}: below the minimum")
        if "maximum" in schema and value > schema["maximum"]:
            out.append(f"{path}: above the maximum")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            out.append(f"{path}: too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            out.append(f"{path}: too many items")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate(schema["items"], item, f"{path}[{index}]", depth + 1, out)
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                out.append(f"{path}.{name}: required")
        extra = schema.get("additionalProperties", True)
        for name, item in value.items():
            if name in properties:
                _validate(properties[name], item, f"{path}.{name}", depth + 1, out)
            elif extra is False:
                out.append(f"{path}.{str(name)[:40]}: not allowed")
            elif isinstance(extra, dict):
                _validate(extra, item, f"{path}.{name}", depth + 1, out)


def _same(a: Any, b: Any) -> bool:
    """Equality that does not conflate True with 1."""
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return bool(a == b)
