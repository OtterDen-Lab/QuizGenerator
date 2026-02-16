import sys

import pytest

from QuizGenerator.generate import _build_practice_question, _tags_match, parse_args
from QuizGenerator.question import QuestionRegistry


def _parse(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["quizgen", *argv])
    return parse_args()


def test_parse_args_generate_practice_allows_missing_yaml(monkeypatch):
    args = _parse(monkeypatch, ["--generate_practice", "cst334", "--course_id", "12345"])
    assert args.quiz_yaml is None
    assert args.generate_practice == ["cst334"]
    assert args.practice_variations == 5


def test_parse_args_generate_practice_requires_course_id(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, ["--generate_practice", "cst334"])


def test_parse_args_requires_yaml_without_generate_practice(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, [])


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
