"""
Tests for YAML configuration parsing.

These tests verify that Quiz.from_yaml() correctly parses various YAML configurations
and handles edge cases appropriately.
"""

import tempfile
from pathlib import Path

import pytest

from QuizGenerator.question import Question, QuestionGroup
from QuizGenerator.quiz import Quiz


@pytest.fixture
def temp_yaml_file():
    """Create a temporary YAML file and return path; clean up after test."""
    files = []

    def _create(content: str) -> str:
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        f.write(content)
        f.flush()
        f.close()
        files.append(f.name)
        return f.name

    yield _create

    # Cleanup
    for path in files:
        try:
            Path(path).unlink()
        except OSError:
            pass


class TestBasicYamlParsing:
    """Tests for basic YAML structure parsing."""

    def test_minimal_yaml_parses(self, temp_yaml_file):
        yaml_content = """
name: "Minimal Quiz"
questions:
  1:
    "Simple Question":
      class: FromText
      kwargs:
        text: "What is 2+2?"
"""
        path = temp_yaml_file(yaml_content)
        quizzes = Quiz.from_yaml(path)

        assert len(quizzes) == 1
        assert quizzes[0].name == "Minimal Quiz"
        assert len(quizzes[0].questions) == 1

    def test_quiz_name_with_time_placeholder(self, temp_yaml_file):
        yaml_content = """
name: "Quiz $TIME{%Y}"
questions:
  1:
    "Q1":
      class: FromText
      kwargs:
        text: "Test"
"""
        path = temp_yaml_file(yaml_content)
        quizzes = Quiz.from_yaml(path)

        # The $TIME{...} should be replaced with actual time
        assert "$TIME" not in quizzes[0].name
        # Should contain a 4-digit year
        import re
        assert re.search(r"\d{4}", quizzes[0].name)

    def test_multiple_documents_in_yaml(self, temp_yaml_file):
        yaml_content = """
name: "Quiz 1"
questions:
  1:
    "Q1":
      class: FromText
      kwargs:
        text: "First quiz question"
---
name: "Quiz 2"
questions:
  1:
    "Q2":
      class: FromText
      kwargs:
        text: "Second quiz question"
"""
        path = temp_yaml_file(yaml_content)
        quizzes = Quiz.from_yaml(path)

        assert len(quizzes) == 2
        assert quizzes[0].name == "Quiz 1"
        assert quizzes[1].name == "Quiz 2"

    def test_practice_flag_parsing(self, temp_yaml_file):
        yaml_content = """
name: "Practice Quiz"
practice: true
questions:
  1:
    "Q1":
      class: FromText
      kwargs:
        text: "Practice question"
"""
        path = temp_yaml_file(yaml_content)
        quizzes = Quiz.from_yaml(path)

        assert quizzes[0].practice is True

    def test_practice_false_by_default(self, temp_yaml_file):
        yaml_content = """
name: "Regular Quiz"
questions:
  1:
    "Q1":
      class: FromText
      kwargs:
        text: "Regular question"
"""
        path = temp_yaml_file(yaml_content)
        quizzes = Quiz.from_yaml(path)

        assert quizzes[0].practice is False

    def test_description_parsing(self, temp_yaml_file):
        yaml_content = """
name: "Quiz with Description"
description: "This quiz covers Chapter 1"
questions:
  1:
    "Q1":
      class: FromText
      kwargs:
        text: "Question"
"""
        path = temp_yaml_file(yaml_content)
        quizzes = Quiz.from_yaml(path)

        assert quizzes[0].description is not None
        assert "Chapter 1" in quizzes[0].description


class TestQuestionParsing:
    """Tests for question-level parsing."""

    def test_multiple_questions_same_point_value(self, temp_yaml_file):
        """Test multiple questions within the same point value tier."""
        yaml_content = """
name: "Multi-Question Quiz"
questions:
  5:
    "Question A":
      class: FromText
      kwargs:
        text: "First question worth 5 points"
    "Question B":
      class: FromText
      kwargs:
        text: "Second question worth 5 points"
"""
        path = temp_yaml_file(yaml_content)
        quizzes = Quiz.from_yaml(path)

        questions = quizzes[0].questions
        assert len(questions) == 2
        assert all(q.points_value == 5 for q in questions)

    def test_question_topic_parsing(self, temp_yaml_file):
        yaml_content = """
name: "Topic Quiz"
questions:
  1:
    "Memory Question":
      class: FromText
      topic: memory
      kwargs:
        text: "Memory question"
"""
        path = temp_yaml_file(yaml_content)
        quizzes = Quiz.from_yaml(path)

        question = quizzes[0].questions[0]
        assert question.topic == Question.Topic.SYSTEM_MEMORY

    def test_question_tags_parsing(self, temp_yaml_file):
        yaml_content = """
name: "Tag Quiz"
questions:
  1:
    "Tagged Question":
      class: FromText
      topic: memory
      tags: [cst334, practice]
      kwargs:
        text: "Tagged question"
"""
        path = temp_yaml_file(yaml_content)
        quizzes = Quiz.from_yaml(path)

        question = quizzes[0].questions[0]
        assert "cst334" in question.tags
        assert "practice" in question.tags
        assert "memory" in question.tags

    def test_legacy_kind_bootstraps_tag(self, temp_yaml_file):
        yaml_content = """
name: "Kind Tag Quiz"
questions:
  1:
    "Legacy Kind Question":
      class: FromText
      kind: programming
      kwargs:
        text: "Question with legacy kind metadata"
"""
        path = temp_yaml_file(yaml_content)
        quizzes = Quiz.from_yaml(path)

        question = quizzes[0].questions[0]
        assert "programming" in question.tags

    def test_question_spacing_preset(self, temp_yaml_file):
        yaml_content = """
name: "Spacing Quiz"
questions:
  1:
    "Long Answer":
      class: FromText
      kwargs:
        text: "Explain in detail"
        spacing: LONG
"""
        path = temp_yaml_file(yaml_content)
        quizzes = Quiz.from_yaml(path)

        question = quizzes[0].questions[0]
        assert question.spacing == 9  # LONG preset

    def test_question_spacing_numeric(self, temp_yaml_file):
        yaml_content = """
name: "Spacing Quiz"
questions:
  1:
    "Custom Spacing":
      class: FromText
      kwargs:
        text: "Question"
        spacing: 7.5
"""
        path = temp_yaml_file(yaml_content)
        quizzes = Quiz.from_yaml(path)

        question = quizzes[0].questions[0]
        assert question.spacing == 7.5


class TestConfigOptions:
    """Tests for _config parsing."""

    def test_repeat_config(self, temp_yaml_file):
        yaml_content = """
name: "Repeat Quiz"
questions:
  1:
    "Repeated Question":
      class: FromText
      _config:
        repeat: 3
      kwargs:
        text: "Same question, different seed"
"""
        path = temp_yaml_file(yaml_content)
        quizzes = Quiz.from_yaml(path)

        # Should create 3 copies of the question
        assert len(quizzes[0].questions) == 3

    def test_group_config_creates_question_group(self, temp_yaml_file):
        yaml_content = """
name: "Group Quiz"
questions:
  1:
    "Question Group":
      _config:
        group: true
        topic: memory
      "Option A":
        class: FromText
        kwargs:
          text: "Option A content"
      "Option B":
        class: FromText
        kwargs:
          text: "Option B content"
"""
        path = temp_yaml_file(yaml_content)
        quizzes = Quiz.from_yaml(path)

        assert len(quizzes[0].questions) == 1
        assert isinstance(quizzes[0].questions[0], QuestionGroup)
        assert len(quizzes[0].questions[0].questions) == 2

    def test_group_tags_apply_to_children(self, temp_yaml_file):
        yaml_content = """
name: "Tagged Group Quiz"
questions:
  1:
    "Question Group":
      _config:
        group: true
        topic: memory
        tags: [cst334, memory]
      "Option A":
        class: FromText
        tags: [fifo]
        kwargs:
          text: "Option A content"
      "Option B":
        class: FromText
        kwargs:
          text: "Option B content"
"""
        path = temp_yaml_file(yaml_content)
        quizzes = Quiz.from_yaml(path)

        group = quizzes[0].questions[0]
        assert isinstance(group, QuestionGroup)
        assert "cst334" in group.tags
        assert "memory" in group.tags
        assert "fifo" in group.tags

    def test_preserve_order_config(self, temp_yaml_file):
        yaml_content = """
name: "Preserve Order Quiz"
questions:
  10:
    _config:
      preserve_order: true
    "Q1":
      class: FromText
      kwargs:
        text: "First question"
    "Q2":
      class: FromText
      kwargs:
        text: "Second question"
"""
        path = temp_yaml_file(yaml_content)
        quizzes = Quiz.from_yaml(path)

        assert 10 in quizzes[0].preserve_order_point_values


class TestSortOrder:
    """Tests for sort order configuration."""

    def test_sort_order_parsing(self, temp_yaml_file):
        yaml_content = """
name: "Sorted Quiz"
sort order:
  - memory
  - processes
  - io
questions:
  1:
    "Q1":
      class: FromText
      kwargs:
        text: "Question"
"""
        path = temp_yaml_file(yaml_content)
        quizzes = Quiz.from_yaml(path)

        sort_order = quizzes[0].question_sort_order
        assert sort_order[0] == Question.Topic.SYSTEM_MEMORY
        assert sort_order[1] == Question.Topic.SYSTEM_PROCESSES
        assert sort_order[2] == Question.Topic.SYSTEM_IO

    def test_unknown_topic_maps_to_misc(self, temp_yaml_file):
        yaml_content = """
name: "Unknown Topic Quiz"
questions:
  1:
    "Q1":
      class: FromText
      topic: unknown_topic_xyz
      kwargs:
        text: "Question"
"""
        path = temp_yaml_file(yaml_content)
        quizzes = Quiz.from_yaml(path)

        question = quizzes[0].questions[0]
        assert question.topic == Question.Topic.MISC


class TestErrorHandling:
    """Tests for error handling in YAML parsing."""

    def test_legacy_pick_key_raises_error(self, temp_yaml_file):
        yaml_content = """
name: "Legacy Pick Quiz"
questions:
  1:
    "Q1":
      class: FromText
      pick: 2
      kwargs:
        text: "Question"
"""
        path = temp_yaml_file(yaml_content)

        with pytest.raises(ValueError, match="Legacy 'pick' key"):
            Quiz.from_yaml(path)

    def test_legacy_repeat_key_raises_error(self, temp_yaml_file):
        yaml_content = """
name: "Legacy Repeat Quiz"
questions:
  1:
    "Q1":
      class: FromText
      repeat: 2
      kwargs:
        text: "Question"
"""
        path = temp_yaml_file(yaml_content)

        with pytest.raises(ValueError, match="Legacy 'repeat' key"):
            Quiz.from_yaml(path)

    def test_unknown_question_class_raises_error(self, temp_yaml_file):
        yaml_content = """
name: "Unknown Class Quiz"
questions:
  1:
    "Q1":
      class: NonExistentQuestionClass12345
      kwargs:
        text: "Question"
"""
        path = temp_yaml_file(yaml_content)

        with pytest.raises(ValueError, match="Unknown question type"):
            Quiz.from_yaml(path)

    def test_missing_questions_key_raises_error(self, temp_yaml_file):
        yaml_content = """
name: "No Questions Quiz"
"""
        path = temp_yaml_file(yaml_content)

        with pytest.raises((KeyError, TypeError)):
            Quiz.from_yaml(path)

    def test_invalid_tags_type_raises_error(self, temp_yaml_file):
        yaml_content = """
name: "Bad Tags Quiz"
questions:
  1:
    "Q1":
      class: FromText
      tags:
        bad: shape
      kwargs:
        text: "Question"
"""
        path = temp_yaml_file(yaml_content)

        with pytest.raises(ValueError, match="tags"):
            Quiz.from_yaml(path)


class TestCustomModules:
    """Tests for custom_modules loading."""

    def test_custom_modules_import_failure_raises(self, temp_yaml_file):
        yaml_content = """
name: "Custom Module Quiz"
custom_modules:
  - nonexistent_module_xyz123
questions:
  1:
    "Q1":
      class: FromText
      kwargs:
        text: "Question"
"""
        path = temp_yaml_file(yaml_content)

        with pytest.raises(ImportError):
            Quiz.from_yaml(path)


class TestQuizMethods:
    """Tests for Quiz instance methods."""

    def test_quiz_iteration_sorts_by_points(self, temp_yaml_file):
        yaml_content = """
name: "Multi-Point Quiz"
questions:
  1:
    "Low":
      class: FromText
      kwargs:
        text: "Low points"
  10:
    "High":
      class: FromText
      kwargs:
        text: "High points"
  5:
    "Medium":
      class: FromText
      kwargs:
        text: "Medium points"
"""
        path = temp_yaml_file(yaml_content)
        quiz = Quiz.from_yaml(path)[0]

        # Iteration should yield highest points first
        questions_list = list(quiz)
        point_values = [q.points_value for q in questions_list]
        assert point_values == sorted(point_values, reverse=True)

    def test_get_quiz_returns_document(self, temp_yaml_file):
        yaml_content = """
name: "Document Quiz"
questions:
  1:
    "Q1":
      class: FromText
      kwargs:
        text: "Question content"
"""
        path = temp_yaml_file(yaml_content)
        quiz = Quiz.from_yaml(path)[0]

        import QuizGenerator.contentast as ca
        document = quiz.get_quiz(rng_seed=42)
        assert isinstance(document, ca.Document)

    def test_get_quiz_with_seed_is_deterministic(self, temp_yaml_file):
        yaml_content = """
name: "Deterministic Quiz"
questions:
  1:
    "Q1":
      class: FromText
      kwargs:
        text: "Question content"
"""
        path = temp_yaml_file(yaml_content)
        quiz = Quiz.from_yaml(path)[0]

        doc1 = quiz.get_quiz(rng_seed=42)
        doc2 = quiz.get_quiz(rng_seed=42)

        # Same seed should produce same content
        assert doc1.render("html") == doc2.render("html")

    def test_get_quiz_seed_group_reuses_seed(self, temp_yaml_file):
        yaml_content = """
name: "Seed Group Quiz"
question_order: yaml
questions:
  - name: "Q1"
    points: 1
    class: FromText
    seed_group: "sched_compare"
    kwargs:
      text: "Question 1"
  - name: "Q2"
    points: 1
    class: FromText
    seed_group: "sched_compare"
    kwargs:
      text: "Question 2"
  - name: "Q3"
    points: 1
    class: FromText
    kwargs:
      text: "Question 3"
"""
        path = temp_yaml_file(yaml_content)
        quiz = Quiz.from_yaml(path)[0]
        document = quiz.get_quiz(rng_seed=42)

        question_seeds = [question.generation_seed for question in document.elements]
        assert question_seeds == [478163327, 478163327, 107420369]

    def test_describe_does_not_crash(self, temp_yaml_file, capsys):
        yaml_content = """
name: "Describable Quiz"
questions:
  10:
    "Q1":
      class: FromText
      topic: memory
      kwargs:
        text: "Question"
  5:
    "Q2":
      class: FromText
      topic: processes
      kwargs:
        text: "Question"
"""
        path = temp_yaml_file(yaml_content)
        quiz = Quiz.from_yaml(path)[0]

        # Should not raise
        quiz.describe()

        captured = capsys.readouterr()
        assert "Describable Quiz" in captured.out
        assert "15" in captured.out or "10" in captured.out  # points
