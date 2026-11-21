"""Deterministic scorers for the AI-12 offline benchmark.

Standard library only, so the same code runs in the desktop container, in Termux
on a phone, and in CI. No scorer calls a model: every number in a report comes
from these functions applied to saved outputs, so a run can be re-scored later.
"""

from __future__ import annotations

import ast
import json
import operator
import re
import unicodedata
from typing import Any

LANGUAGE_NAMES = {"en": "English", "yo": "Yoruba", "ha": "Hausa", "ig": "Igbo"}

# Function words used as a language-share proxy. Written without diacritics because
# text is compared after the marks are stripped. Single letters that several of these
# languages share with English ("a", "o", "n") are left out on purpose.
STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        "the and is of to in that it for are this with as be by on which when because than "
        "from an or its so not can has have at".split()
    ),
    "yo": frozenset(
        "ni ti ati je si won ki fun pe naa yii re ko bi lati ninu wa se awon ohun tabi "
        "nitori sugbon maa lo gbogbo ba eyi nigba".split()
    ),
    "ha": frozenset(
        "da na ta ya shi ita ne ce cikin kuma don wanda yana suna ba za daga wannan akwai "
        "idan saboda amma haka wato ko su mai ake aka kan".split()
    ),
    "ig": frozenset(
        "na ya nke bu ka ndi maka di site gi ha ga ma nwere ihe mgbe otu ebe ahu iji kwa "
        "anyi unu ike nwa".split()
    ),
}

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
_NUMERIC = re.compile(r"^-?\d+(?:[.,]\d+)?$")


def strip_marks(text: str) -> str:
    """Lower-case, NFKC, and drop combining marks (tones, under-dots)."""
    decomposed = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", text))
    bare = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return bare.replace("−", "-").replace("–", "-").lower()


def strip_reasoning(text: str) -> str:
    """Remove <think> blocks some chat templates emit even with thinking disabled."""
    return _THINK.sub("", text or "").strip()


def matches(text: str, alternative: str) -> bool:
    """True when the alternative appears in the text.

    Numbers must stand alone (so "10" does not match "100"); words match as
    substrings after case and diacritics are removed.
    """
    haystack = strip_marks(text)
    needle = strip_marks(alternative)
    if _NUMERIC.match(needle):
        pattern = r"(?<![\d.,])" + re.escape(needle) + r"(?![\d]|[.,]\d)"
        return re.search(pattern, haystack) is not None
    return needle in haystack


def group_recall(text: str, groups: list[list[str]]) -> float:
    """Fraction of groups with at least one matching alternative (1.0 when empty)."""
    if not groups:
        return 1.0
    hit = sum(1 for alternatives in groups if any(matches(text, a) for a in alternatives))
    return hit / len(groups)


def language_share(text: str, lang: str) -> float | None:
    """Share of recognised function words that belong to the target language.

    A proxy for "did the model answer in the requested language", not a quality score.
    Returns None when no function word of either language is found.
    """
    tokens = re.findall(r"[a-z']+", strip_marks(text))
    target = STOPWORDS[lang]
    others = STOPWORDS["en"] if lang != "en" else STOPWORDS["yo"] | STOPWORDS["ha"]
    hits = sum(1 for t in tokens if t in target)
    misses = sum(1 for t in tokens if t in others and t not in target)
    if hits + misses == 0:
        return None
    return hits / (hits + misses)


def parse_json(text: str) -> tuple[Any, str]:
    """Parse a model reply as JSON.

    Returns (value, status). status is "strict" when the whole reply is one JSON
    value, "lenient" when JSON had to be cut out of a code fence or surrounding
    prose, "invalid" otherwise (value None).
    """
    body = strip_reasoning(text)
    try:
        return json.loads(body), "strict"
    except (json.JSONDecodeError, ValueError):
        pass
    fenced = _FENCE.match(body)
    candidates = [fenced.group(1)] if fenced else []
    start = body.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(body)):
            ch = body[index]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(body[start : index + 1])
                    break
    for candidate in candidates:
        try:
            return json.loads(candidate), "lenient"
        except (json.JSONDecodeError, ValueError):
            continue
    return None, "invalid"


def schema_errors(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Validate the JSON Schema subset the task set uses. Returns error strings."""
    errors: list[str] = []
    kind = schema.get("type")
    checks = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "boolean": lambda v: isinstance(v, bool),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    }
    if kind in checks and not checks[kind](value):
        return [f"{path}: expected {kind}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: not one of {schema['enum']}")
    if kind in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum")
    if kind == "array":
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: too many items")
        for i, item in enumerate(value):
            errors.extend(schema_errors(item, schema.get("items", {}), f"{path}[{i}]"))
    if kind == "object":
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: missing")
        for key, item in value.items():
            if key in properties:
                errors.extend(schema_errors(item, properties[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{key}: not allowed")
    return errors


GROUNDED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "answerable": {"type": "boolean"},
    },
    "required": ["answer", "citations", "answerable"],
    "additionalProperties": False,
}


def score_grounded(task: dict[str, Any], reply: str) -> dict[str, Any]:
    """Answer correctness, citation faithfulness and abstention for one reply."""
    value, status = parse_json(reply)
    result: dict[str, Any] = {"json": status, "schema_ok": False}
    if status == "invalid" or not isinstance(value, dict):
        result.update(answerable_pred=None, correct=False, faithful=False)
        return result
    result["schema_ok"] = not schema_errors(value, GROUNDED_SCHEMA)
    answer = value.get("answer") if isinstance(value.get("answer"), str) else ""
    raw_citations = value.get("citations")
    citations = (
        [c for c in raw_citations if isinstance(c, str)] if isinstance(raw_citations, list) else []
    )
    cited = {c.strip().strip("[]").upper() for c in citations}
    predicted = value.get("answerable")
    if not isinstance(predicted, bool):
        predicted = None
    provided = set(task["passages"])
    supporting = set(task["supporting"])
    result["answerable_pred"] = predicted
    result["cited"] = sorted(cited)
    result["cites_unknown_passage"] = bool(cited - provided)
    if task["answerable"]:
        recall = group_recall(answer, task["facts"])
        result["fact_recall"] = recall
        result["citation_precision"] = len(cited & supporting) / len(cited) if cited else 0.0
        result["citation_recall"] = len(cited & supporting) / len(supporting)
        result["correct"] = predicted is True and recall == 1.0
        # Faithful: right answer, at least one supporting passage cited, nothing cited
        # that does not support it.
        result["faithful"] = bool(result["correct"] and cited and cited <= supporting)
        result["false_abstention"] = predicted is False
    else:
        abstained = predicted is False
        result["correct"] = abstained
        result["faithful"] = abstained
        result["hallucinated"] = predicted is True
    if task.get("lang", "en") != "en":
        result["language_share"] = language_share(answer, task["lang"])
    return result


def _expected_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return isinstance(actual, str) and any(matches(actual, e) for e in expected)
    if isinstance(expected, (int, float)):
        return isinstance(actual, (int, float)) and abs(actual - expected) <= 1e-6 * max(
            1.0, abs(expected)
        )
    return actual == expected


def score_json_extract(task: dict[str, Any], reply: str) -> dict[str, Any]:
    """JSON validity, schema conformance and field accuracy."""
    value, status = parse_json(reply)
    result: dict[str, Any] = {"json": status}
    if status == "invalid":
        result.update(schema_ok=False, field_accuracy=0.0, correct=False)
        return result
    errors = schema_errors(value, task["schema"])
    result["schema_ok"] = not errors
    result["schema_errors"] = errors[:5]
    expected = task["expected"]
    fields = value if isinstance(value, dict) else {}
    hits = sum(1 for k, e in expected.items() if _expected_match(fields.get(k), e))
    result["field_accuracy"] = hits / len(expected)
    if task.get("structural") == "mcq" and isinstance(value, dict):
        options = value.get("options")
        result["mcq_ok"] = (
            isinstance(options, list)
            and len(options) == 4
            and len({strip_marks(str(o)) for o in options}) == 4
            and isinstance(value.get("answer_index"), int)
            and 0 <= value["answer_index"] <= 3
        )
        result["correct"] = bool(result["schema_ok"] and result["mcq_ok"] and hits == len(expected))
    else:
        result["correct"] = bool(result["schema_ok"] and hits == len(expected))
    return result


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> float | None:
    """Evaluate +-*/** arithmetic on numbers only; None for anything else."""

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](walk(node.operand))
        raise ValueError("unsupported expression")

    try:
        text = expression.replace("×", "*").replace("x", "*").replace("^", "**")
        return walk(ast.parse(text, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None


def _argument_ok(actual: Any, rule: dict[str, Any]) -> bool:
    if "contains" in rule:
        return isinstance(actual, str) and any(matches(actual, c) for c in rule["contains"])
    if "course" in rule:
        return isinstance(actual, str) and re.sub(r"\s+", "", actual).upper() == rule["course"]
    if "equals" in rule:
        try:
            return float(actual) == float(rule["equals"])
        except (TypeError, ValueError):
            return False
    if "evaluates_to" in rule:
        value = safe_eval(actual) if isinstance(actual, str) else None
        return value is not None and abs(value - rule["evaluates_to"]) < 1e-9
    raise ValueError(f"unknown rule {rule}")


def score_tool_call(task: dict[str, Any], tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Tool choice and argument correctness for an OpenAI-style tool_calls list."""
    expected = task["expected"]
    result: dict[str, Any] = {"calls": len(tool_calls)}
    if expected is None:
        result["correct"] = not tool_calls
        result["name_ok"] = result["args_parse"] = result["args_ok"] = result["correct"]
        return result
    if not tool_calls:
        result.update(name_ok=False, args_parse=False, args_ok=False, correct=False)
        return result
    call = tool_calls[0].get("function", {})
    result["name_ok"] = call.get("name") == expected["name"]
    raw = call.get("arguments")
    arguments: Any = raw
    if isinstance(raw, str):
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError:
            arguments = None
    result["args_parse"] = isinstance(arguments, dict)
    args = arguments if isinstance(arguments, dict) else {}
    checks = {k: _argument_ok(args.get(k), rule) for k, rule in expected["arguments"].items()}
    result["failed_args"] = sorted(k for k, ok in checks.items() if not ok)
    result["args_ok"] = all(checks.values())
    result["correct"] = bool(result["name_ok"] and result["args_ok"] and len(tool_calls) == 1)
    return result


def score_explain(task: dict[str, Any], reply: str) -> dict[str, Any]:
    """Proxy metrics for free-text explanations; quality needs the human review sheet."""
    text = strip_reasoning(reply)
    words = len(text.split())
    result: dict[str, Any] = {
        "term_recall": group_recall(text, task["required_terms"]),
        "words": words,
        "within_limit": 0 < words <= task["max_words"],
        "language_share": language_share(text, task["lang"]),
    }
    share = result["language_share"]
    result["in_language"] = share is not None and share >= 0.6
    result["correct"] = bool(result["term_recall"] == 1.0 and result["in_language"])
    return result
