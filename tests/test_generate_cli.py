import sys

import pytest

from QuizGenerator.generate import (
    _build_practice_question,
    _question_tags_for_source,
    _tags_match,
    parse_args,
)
from QuizGenerator.question import QuestionRegistry


def _parse(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["quizgen", *argv])
    return parse_args()


def test_parse_args_generate_practice_allows_missing_yaml(monkeypatch):
    args = _parse(monkeypatch, ["--generate_practice", "cst334", "--course_id", "12345"])
    assert args.quiz_yaml is None
    assert args.generate_practice == ["cst334"]
    assert args.practice_variations == 5
    assert args.practice_question_groups == 5
    assert args.practice_tag_source == "merged"


def test_parse_args_subcommand_practice_positional_tags(monkeypatch):
    args = _parse(monkeypatch, ["practice", "course:cst334", "topic:memory", "--course_id", "12345"])
    assert args.command == "practice"
    assert args.generate_practice == ["course:cst334", "topic:memory"]


def test_parse_args_generate_practice_requires_course_id(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, ["--generate_practice", "cst334"])


def test_parse_args_requires_yaml_without_generate_practice(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, [])


def test_parse_args_subcommand_generate_requires_yaml(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, ["generate", "--num_pdfs", "1"])


def test_parse_args_subcommand_test_positional_variations(monkeypatch):
    args = _parse(monkeypatch, ["test", "7"])
    assert args.command == "test"
    assert args.test_all == 7


def test_parse_args_legacy_check_deps_routes_to_deps(monkeypatch):
    args = _parse(monkeypatch, ["--check-deps", "--yaml", "example_files/example_exam.yaml", "--num_pdfs", "1"])
    assert args.command == "deps"
    assert args.check_deps is True


def test_parse_args_generate_practice_question_groups(monkeypatch):
    args = _parse(
        monkeypatch,
        [
            "--generate_practice",
            "cst334",
            "--course_id",
            "12345",
            "--practice_question_groups",
            "5",
        ],
    )
    assert args.practice_question_groups == 5


def test_parse_args_generate_practice_question_groups_must_be_positive(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(
            monkeypatch,
            [
                "--generate_practice",
                "cst334",
                "--course_id",
                "12345",
                "--practice_question_groups",
                "0",
            ],
        )


def test_parse_args_generate_practice_tag_source(monkeypatch):
    args = _parse(
        monkeypatch,
        [
            "--generate_practice",
            "course:cst334",
            "--course_id",
            "12345",
            "--practice_tag_source",
            "explicit",
        ],
    )
    assert args.practice_tag_source == "explicit"


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
