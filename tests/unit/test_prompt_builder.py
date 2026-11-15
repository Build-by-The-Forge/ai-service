from app.core.prompt_builder import PromptBuilder


def test_build_chat_tutor_prompt_includes_memory_context_and_question():
    builder = PromptBuilder()

    prompt = builder.build(
        name="chat_tutor",
        memory="algebra: mastery=0.75",
        context="sample context",
        question="What is 2+2?",
    )

    assert "STUDENT MEMORY:" in prompt["user"]
    assert "COURSE PASSAGES:" in prompt["user"]
    assert "QUESTION:" in prompt["user"]
    assert "What is 2+2?" in prompt["user"]


def test_the_raw_text_exam_prompt_is_gone():
    """AI-10 generates through structured_questions; the free-text CBT
    prompt it used before is removed, so nothing can ask for raw questions."""
    prompt = PromptBuilder().build(name="question_generator", topic="geometry", context="ctx")

    assert prompt == {"system": "", "user": ""}


def test_the_question_prompt_is_unchanged_without_a_course():
    user = PromptBuilder().build(
        name="structured_questions",
        topic="cells",
        question_type="mcq",
        difficulty="easy",
        count=2,
        context="[1] passage",
    )["user"]

    assert user == (
        "\nTOPIC: cells\nQUESTION TYPE: mcq\nDIFFICULTY: easy\nNUMBER OF QUESTIONS: 2\n"
        "\nCONTEXT:\n[1] passage\n"
    )


def test_a_course_leads_the_question_prompt():
    user = PromptBuilder().build(
        name="structured_questions", course="Mathematics", topic="functions", context="c"
    )["user"]

    assert user.startswith("\nCOURSE: Mathematics\nTOPIC: functions\n")
