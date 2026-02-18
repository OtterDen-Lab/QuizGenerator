from QuizGenerator.generate import (
    _build_practice_question,
    _question_tags_for_source,
    _tags_match,
)
from QuizGenerator.question import QuestionRegistry


def test_tags_match_any_and_all_modes():
    candidate = {"cst334", "memory", "practice"}
    requested = {"cst334", "memory"}

    assert _tags_match(candidate, requested, match_all=True)
    assert _tags_match(candidate, {"cst334", "io"}, match_all=False)
    assert not _tags_match(candidate, {"cst334", "io"}, match_all=True)


def test_build_practice_question_uses_defaults_for_fromtext():
    QuestionRegistry.load_premade_questions()
    question_cls = QuestionRegistry._registry["fromtext"]
    question, error = _build_practice_question("fromtext", question_cls, points_value=1.0)

    assert error is None
    assert question is not None
    assert question.points_value == 1.0


def test_question_tags_for_source_selects_expected_set():
    class _Dummy:
        explicit_tags = {"course:cst334"}
        derived_tags = {"topic:memory"}
        tags = {"course:cst334", "topic:memory", "practice"}

    question = _Dummy()
    assert _question_tags_for_source(question, "explicit") == {"course:cst334"}
    assert _question_tags_for_source(question, "derived") == {"topic:memory"}
    assert _question_tags_for_source(question, "merged") == {"course:cst334", "topic:memory", "practice"}
