"""The JSON-Schema subset: what it enforces, what it refuses to pretend to."""

import pytest

from app.core.agentic.schema import SchemaError, check_schema, validate

OBJ = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1, "maxLength": 10},
        "limit": {"type": "integer", "minimum": 1, "maximum": 5},
        "mode": {"enum": ["a", "b"]},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
    },
    "required": ["query"],
    "additionalProperties": False,
}


def test_valid_value_has_no_problems():
    assert validate(OBJ, {"query": "heat", "limit": 3, "mode": "a", "tags": ["x"]}) == []


@pytest.mark.parametrize(
    "value, fragment",
    [
        ({}, "$.query: required"),
        ({"query": ""}, "too short"),
        ({"query": "x" * 11}, "too long"),
        ({"query": "x", "limit": 0}, "below the minimum"),
        ({"query": "x", "limit": 6}, "above the maximum"),
        ({"query": "x", "limit": 2.5}, "wrong type"),
        ({"query": "x", "mode": "c"}, "not an allowed value"),
        ({"query": "x", "tags": ["a", "b", "c"]}, "too many items"),
        ({"query": "x", "tags": [1]}, "$.tags[0]: wrong type"),
        ({"query": "x", "tenant_id": "other"}, "$.tenant_id: not allowed"),
    ],
)
def test_each_rule_reports_its_path(value, fragment):
    assert any(fragment in p for p in validate(OBJ, value))


def test_a_boolean_is_not_an_integer_or_a_number():
    assert validate({"type": "integer"}, True) == ["$: wrong type"]
    assert validate({"type": "number"}, False) == ["$: wrong type"]
    assert validate({"enum": [1]}, True) != []


def test_non_finite_numbers_are_refused():
    assert validate({"type": "number"}, float("nan")) == ["$: wrong type"]
    assert validate({"type": "number"}, float("inf")) == ["$: wrong type"]


def test_problems_never_quote_the_value():
    secret = "the student's private answer"
    problems = validate(OBJ, {"query": secret})
    assert problems and all(secret not in p for p in problems)
    odd_key = "k" * 200
    assert all(len(p) < 80 for p in validate(OBJ, {"query": "x", odd_key: 1}))


def test_validation_is_bounded_by_the_schema_not_the_value():
    """A deeply nested value is only walked as deep as the schema describes."""
    value: dict = {}
    cursor = value
    for _ in range(500):
        cursor["a"] = {}
        cursor = cursor["a"]
    assert validate({"type": "object"}, value) == []
    assert validate({"type": "object", "additionalProperties": {"type": "object"}}, value) == []


@pytest.mark.parametrize(
    "schema",
    [
        {"anyOf": [{"type": "string"}]},
        {"$ref": "#/definitions/x"},
        {"type": "string", "pattern": "^a+$"},
        {"type": "wibble"},
        {"type": "object", "properties": {"a": {"oneOf": []}}},
        {"enum": []},
        {"required": "name"},
        {"minItems": -1},
        {"minimum": "1"},
        "string",
    ],
)
def test_a_schema_it_cannot_enforce_is_refused(schema):
    with pytest.raises(SchemaError):
        check_schema(schema)


def test_annotations_are_allowed():
    check_schema({"type": "string", "description": "d", "title": "t", "default": "x"})


def test_a_schema_that_is_too_deep_or_too_large_is_refused():
    schema: dict = {"type": "string"}
    for _ in range(12):
        schema = {"type": "object", "properties": {"a": schema}}
    with pytest.raises(SchemaError):
        check_schema(schema)
    wide = {"type": "object", "properties": {f"p{i}": {"type": "string"} for i in range(500)}}
    with pytest.raises(SchemaError):
        check_schema(wide)
