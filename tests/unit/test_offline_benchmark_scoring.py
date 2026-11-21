"""benchmarks/offline: the AI-12 scorers and task set (no model is loaded)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarks" / "offline"
_spec = importlib.util.spec_from_file_location("offline_scoring", BENCH / "scoring.py")
scoring = importlib.util.module_from_spec(_spec)
sys.modules["offline_scoring"] = scoring
_spec.loader.exec_module(scoring)


def load(name):
    return [json.loads(line) for line in (BENCH / "tasks" / name).read_text("utf-8").splitlines()]


PASSAGES = {p["id"]: p for p in load("passages.jsonl")}
TASKS = load("tasks.jsonl")
TOOLS = {t["function"]["name"] for t in json.loads((BENCH / "tasks" / "tools.json").read_text())}


def task(task_id):
    return next(t for t in TASKS if t["id"] == task_id)


# --- task set integrity -------------------------------------------------------


def test_task_ids_are_unique_and_reference_real_passages():
    assert len({t["id"] for t in TASKS}) == len(TASKS)
    for t in TASKS:
        assert set(t.get("passages", [])) <= set(PASSAGES), t["id"]
        assert set(t.get("supporting", [])) <= set(t.get("passages", [])), t["id"]


def test_grounded_tasks_are_consistent():
    for t in (t for t in TASKS if t["type"] == "grounded"):
        if t["answerable"]:
            assert t["facts"] and t["supporting"], t["id"]
            # Every fact is literally present in a supporting passage.
            text = " ".join(PASSAGES[p]["text"] for p in t["supporting"])
            for group in t["facts"]:
                assert any(scoring.matches(text, alt) for alt in group), (t["id"], group)
        else:
            assert not t["facts"] and not t["supporting"], t["id"]


def test_tool_tasks_name_defined_tools():
    for t in (t for t in TASKS if t["type"] == "tool_call" and t["expected"]):
        assert t["expected"]["name"] in TOOLS


def test_non_english_tasks_are_marked_for_native_review():
    for t in TASKS:
        if t.get("lang", "en") != "en":
            assert t.get("review") == "unreviewed", t["id"]


def test_extraction_expectations_are_in_their_schema():
    for t in (t for t in TASKS if t["type"] == "json_extract"):
        assert set(t["expected"]) <= set(t["schema"]["properties"]), t["id"]


# --- scorers -------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,alt,hit",
    [
        ("about 22.4 L", "22.4", True),
        ("at most 100 comparisons", "10", False),
        ("2^10 = 1,024", "10", True),
        ("ìpíndọ́gba jẹ́ mẹ́jọ", "mejo", True),
        ("elasticity is −2.5", "-2.5", True),
        ("Safranin", "safranin", True),
    ],
)
def test_matches(text, alt, hit):
    assert scoring.matches(text, alt) is hit


def test_parse_json_strict_lenient_invalid():
    assert scoring.parse_json('{"a": 1}') == ({"a": 1}, "strict")
    assert scoring.parse_json('```json\n{"a": 1}\n```') == ({"a": 1}, "lenient")
    assert scoring.parse_json('Sure! {"a": "}"} hope this helps') == ({"a": "}"}, "lenient")
    assert scoring.parse_json('<think>x</think>\n{"a": 2}') == ({"a": 2}, "strict")
    assert scoring.parse_json("no json here") == (None, "invalid")


def test_grounded_correct_and_faithful():
    m = scoring.score_grounded(
        task("G01"), '{"answer": "About 22.4 L", "citations": ["P01"], "answerable": true}'
    )
    assert m["correct"] and m["faithful"] and m["citation_precision"] == 1.0


def test_grounded_correct_but_cites_irrelevant_passage_is_not_faithful():
    m = scoring.score_grounded(
        task("G01"), '{"answer": "22.4 L", "citations": ["P01", "P09"], "answerable": true}'
    )
    assert m["correct"] and not m["faithful"] and m["citation_precision"] == 0.5


def test_grounded_unanswerable_rewards_abstention_and_flags_hallucination():
    t = task("A03")
    ok = scoring.score_grounded(t, '{"answer": "", "citations": [], "answerable": false}')
    bad = scoring.score_grounded(t, '{"answer": "1884", "citations": ["P05"], "answerable": true}')
    assert ok["correct"] and not ok["hallucinated"]
    assert not bad["correct"] and bad["hallucinated"]


def test_grounded_invalid_json_scores_zero():
    m = scoring.score_grounded(task("G01"), "22.4 litres")
    assert m["json"] == "invalid" and not m["correct"] and not m["faithful"]


def test_json_extract_fields_and_schema():
    good = json.dumps(
        {
            "cement_parts": 1,
            "sand_parts": 2,
            "coarse_aggregate_parts": 4,
            "min_curing_days": 7,
            "strength_test_age_days": 28,
        }
    )
    assert scoring.score_json_extract(task("J01"), good)["correct"]
    wrong = scoring.score_json_extract(task("J01"), good.replace("28", '"28"'))
    assert not wrong["schema_ok"] and wrong["field_accuracy"] == 0.8


def test_mcq_structure():
    mcq = {"question": "q", "options": ["a", "b", "c", "d"], "answer_index": 2, "source": "P03"}
    assert scoring.score_json_extract(task("J06"), json.dumps(mcq))["correct"]
    mcq["options"] = ["a", "a", "c", "d"]
    assert not scoring.score_json_extract(task("J06"), json.dumps(mcq))["correct"]


def call(name, **arguments):
    return [{"function": {"name": name, "arguments": json.dumps(arguments)}}]


def test_tool_calls():
    assert scoring.score_tool_call(
        task("T01"), call("search_course_pack", query="binary search notes", course_code="csc201")
    )["correct"]
    assert scoring.score_tool_call(task("T03"), call("calculate", expression="0.75*120+18"))[
        "correct"
    ]
    wrong = scoring.score_tool_call(
        task("T04"), call("schedule_review", topic="Gram", days_from_now=4)
    )
    assert not wrong["correct"] and wrong["failed_args"] == ["days_from_now"]
    assert scoring.score_tool_call(task("T05"), [])["correct"]
    assert not scoring.score_tool_call(task("T05"), call("calculate", expression="1"))["correct"]


def test_safe_eval_refuses_code():
    assert scoring.safe_eval("2 ** 3 - 1") == 7
    assert scoring.safe_eval("__import__('os')") is None


def test_language_share_separates_languages():
    yoruba = "Ẹ̀gẹ́ ní àwọn kẹ́míkà tí ó lè di májèlé, nítorí náà a gbọ́dọ̀ ṣe é dáadáa"
    english = "Cassava contains chemicals that can become poisonous, so it is processed."
    assert scoring.language_share(yoruba, "yo") > 0.6
    assert scoring.language_share(english, "yo") < 0.4
    assert scoring.language_share("", "ha") is None
