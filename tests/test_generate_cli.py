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


def test_parse_args_subcommand_practice_allows_missing_yaml(monkeypatch):
    args = _parse(monkeypatch, ["practice", "cst334", "--course-id", "12345"])
    assert args.command == "practice"
    assert args.tags == ["cst334"]
    assert args.practice_variations == 5
    assert args.practice_question_groups == 5
    assert args.practice_tag_source == "merged"


def test_parse_args_subcommand_practice_positional_tags(monkeypatch):
    args = _parse(monkeypatch, ["practice", "course:cst334", "topic:memory", "--course-id", "12345"])
    assert args.command == "practice"
    assert args.tags == ["course:cst334", "topic:memory"]


def test_parse_args_subcommand_practice_requires_course_id(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, ["practice", "cst334"])


def test_parse_args_requires_subcommand(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, [])


def test_parse_args_subcommand_generate_requires_yaml(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, ["generate", "--num-pdfs", "1"])


def test_parse_args_generate_pdf_aids_default_true(monkeypatch):
    args = _parse(
        monkeypatch,
        ["generate", "--yaml", "example_files/example_exam.yaml", "--num-pdfs", "1"],
    )
    assert args.show_pdf_aids is True


def test_parse_args_generate_no_pdf_aids(monkeypatch):
    args = _parse(
        monkeypatch,
        ["generate", "--yaml", "example_files/example_exam.yaml", "--num-pdfs", "1", "--no-pdf-aids"],
    )
    assert args.show_pdf_aids is False


def test_parse_args_subcommand_test_positional_variations(monkeypatch):
    args = _parse(monkeypatch, ["test", "7"])
    assert args.command == "test"
    assert args.num_variations == 7


def test_parse_args_subcommand_test_variations_after_test_questions(monkeypatch):
    args = _parse(monkeypatch, ["test", "--test-questions", "MLFQQuestion", "1"])
    assert args.command == "test"
    assert args.num_variations == 1
    assert args.test_questions == ["MLFQQuestion"]


def test_parse_args_subcommand_test_variations_before_test_questions(monkeypatch):
    args = _parse(monkeypatch, ["test", "1", "--test-questions", "MLFQQuestion"])
    assert args.command == "test"
    assert args.num_variations == 1
    assert args.test_questions == ["MLFQQuestion"]


def test_parse_args_subcommand_tags_defaults_to_list(monkeypatch):
    args = _parse(monkeypatch, ["tags"])
    assert args.command == "tags"
    assert args.tags_command == "list"
    assert args.tag_source == "merged"


def test_parse_args_subcommand_tags_explain(monkeypatch):
    args = _parse(monkeypatch, ["tags", "explain", "sched"])
    assert args.command == "tags"
    assert args.tags_command == "explain"
    assert args.query == "sched"


def test_parse_args_subcommand_deps(monkeypatch):
    args = _parse(monkeypatch, ["deps"])
    assert args.command == "deps"


def test_parse_args_subcommand_practice_question_groups(monkeypatch):
    args = _parse(
        monkeypatch,
        [
            "practice",
            "cst334",
            "--course-id",
            "12345",
            "--practice-question-groups",
            "5",
        ],
    )
    assert args.practice_question_groups == 5


def test_parse_args_subcommand_practice_question_groups_must_be_positive(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(
            monkeypatch,
            [
                "practice",
                "cst334",
                "--course-id",
                "12345",
                "--practice-question-groups",
                "0",
            ],
        )


def test_parse_args_subcommand_practice_tag_source(monkeypatch):
    args = _parse(
        monkeypatch,
        [
            "practice",
            "course:cst334",
            "--course-id",
            "12345",
            "--practice-tag-source",
            "explicit",
        ],
    )
    assert args.practice_tag_source == "explicit"


def test_parse_args_subcommand_test_requires_positive_num_variations(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, ["test", "0"])


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
