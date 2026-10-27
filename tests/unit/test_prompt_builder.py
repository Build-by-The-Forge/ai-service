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


def test_build_question_generator_prompt_uses_topic_and_count():
    builder = PromptBuilder()

    prompt = builder.build(
        name="question_generator",
        topic="geometry",
        difficulty="hard",
        count=3,
        context="triangle rules",
    )

    assert "COURSE: geometry" in prompt["user"]
    assert "DIFFICULTY: hard" in prompt["user"]
    assert "NUMBER OF QUESTIONS: 3" in prompt["user"]
    assert "triangle rules" in prompt["user"]
