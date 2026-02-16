"""
Tests for list-style YAML question format and ordering behavior.

This tests the ordered list format for questions and the question_order option.
"""

import tempfile
from pathlib import Path

import pytest

from QuizGenerator.question import Question
from QuizGenerator.quiz import Quiz


def _write_tmp_yaml(text):
    """Write YAML text to a temporary file and return the path."""
    path = Path(tempfile.mkdtemp()) / "ordered.yaml"
    path.write_text(text)
    return str(path)


class TestListFormatBasics:
    """Tests for basic list-format YAML parsing."""

    def test_yaml_list_preserves_order(self):
        """List format should preserve YAML order by default."""
        yaml_text = """
name: "Ordered Quiz"
questions:
  - name: "First"
    points: 2
    class: FromText
    kwargs:
      text: "Q1"
  - name: "Second"
    points: 10
    class: FromText
    kwargs:
      text: "Q2"
"""
        quiz = Quiz.from_yaml(_write_tmp_yaml(yaml_text))[0]
        ordered = quiz.get_ordered_questions()
        assert [q.name for q in ordered] == ["First", "Second"]

    def test_list_format_parses_all_questions(self):
        """All questions in list format should be parsed."""
        yaml_text = """
name: "Multi Question Quiz"
questions:
  - name: "Q1"
    points: 5
    class: FromText
    kwargs:
      text: "First"
  - name: "Q2"
    points: 3
    class: FromText
    kwargs:
      text: "Second"
  - name: "Q3"
    points: 7
    class: FromText
    kwargs:
      text: "Third"
"""
        quiz = Quiz.from_yaml(_write_tmp_yaml(yaml_text))[0]
        assert len(quiz.questions) == 3

    def test_list_format_preserves_point_values(self):
        """Each question should have its specified point value."""
        yaml_text = """
name: "Point Value Quiz"
questions:
  - name: "Low Points"
    points: 1
    class: FromText
    kwargs:
      text: "Easy"
  - name: "High Points"
    points: 20
    class: FromText
    kwargs:
      text: "Hard"
"""
        quiz = Quiz.from_yaml(_write_tmp_yaml(yaml_text))[0]
        questions = quiz.get_ordered_questions()

        assert questions[0].name == "Low Points"
        assert questions[0].points_value == 1
        assert questions[1].name == "High Points"
        assert questions[1].points_value == 20

    def test_list_format_with_topics(self):
        """Questions in list format should support topic assignment."""
        yaml_text = """
name: "Topic Quiz"
questions:
  - name: "Memory Q"
    points: 5
    topic: memory
    class: FromText
    kwargs:
      text: "Memory question"
  - name: "Process Q"
    points: 5
    topic: processes
    class: FromText
    kwargs:
      text: "Process question"
"""
        quiz = Quiz.from_yaml(_write_tmp_yaml(yaml_text))[0]
        questions = quiz.get_ordered_questions()

        assert questions[0].topic == Question.Topic.SYSTEM_MEMORY
        assert questions[1].topic == Question.Topic.SYSTEM_PROCESSES

    def test_list_format_with_tags(self):
        """Questions in list format should support explicit tags."""
        yaml_text = """
name: "Tag Quiz"
questions:
  - name: "Tagged Q"
    points: 5
    class: FromText
    tags: [cst334, practice]
    kwargs:
      text: "Tag me"
"""
        quiz = Quiz.from_yaml(_write_tmp_yaml(yaml_text))[0]
        question = quiz.get_ordered_questions()[0]

        assert "course:cst334" in question.tags
        assert "practice" in question.tags

    def test_list_format_with_spacing(self):
        """Questions in list format should support spacing configuration."""
        yaml_text = """
name: "Spacing Quiz"
questions:
  - name: "Short Space"
    points: 5
    class: FromText
    kwargs:
      text: "Short answer"
      spacing: SHORT
  - name: "Long Space"
    points: 10
    class: FromText
    kwargs:
      text: "Essay"
      spacing: LONG
"""
        quiz = Quiz.from_yaml(_write_tmp_yaml(yaml_text))[0]
        questions = quiz.get_ordered_questions()

        assert questions[0].spacing == 4  # SHORT
        assert questions[1].spacing == 9  # LONG


class TestQuestionOrderOption:
    """Tests for the question_order top-level option."""

    def test_question_order_yaml_preserves_order(self):
        """question_order: yaml should preserve YAML order."""
        yaml_text = """
name: "Yaml Order Quiz"
question_order: yaml
questions:
  - name: "First (low points)"
    points: 1
    class: FromText
    kwargs:
      text: "First"
  - name: "Second (high points)"
    points: 100
    class: FromText
    kwargs:
      text: "Second"
"""
        quiz = Quiz.from_yaml(_write_tmp_yaml(yaml_text))[0]
        ordered = quiz.get_ordered_questions()

        # Despite high points on second, should stay in YAML order
        assert [q.name for q in ordered] == ["First (low points)", "Second (high points)"]

    def test_question_order_points_reorders(self):
        """question_order: points should reorder by point value."""
        yaml_text = """
name: "Points Order Quiz"
question_order: points
questions:
  - name: "Low Points First in YAML"
    points: 1
    class: FromText
    kwargs:
      text: "Low"
  - name: "High Points Second in YAML"
    points: 100
    class: FromText
    kwargs:
      text: "High"
"""
        quiz = Quiz.from_yaml(_write_tmp_yaml(yaml_text))[0]
        ordered = quiz.get_ordered_questions()

        # Should reorder by points (high first)
        assert ordered[0].points_value >= ordered[1].points_value

    def test_question_order_aliases(self):
        """question_order should accept aliases: given, preserve, value, score."""
        for yaml_order in ["yaml", "given", "preserve"]:
            yaml_text = f"""
name: "Alias Test"
question_order: {yaml_order}
questions:
  - name: "First"
    points: 1
    class: FromText
    kwargs:
      text: "First"
  - name: "Second"
    points: 100
    class: FromText
    kwargs:
      text: "Second"
"""
            quiz = Quiz.from_yaml(_write_tmp_yaml(yaml_text))[0]
            assert quiz.preserve_yaml_order is True, f"Failed for {yaml_order}"

        for points_order in ["points", "value", "score"]:
            yaml_text = f"""
name: "Alias Test"
question_order: {points_order}
questions:
  - name: "First"
    points: 1
    class: FromText
    kwargs:
      text: "First"
  - name: "Second"
    points: 100
    class: FromText
    kwargs:
      text: "Second"
"""
            quiz = Quiz.from_yaml(_write_tmp_yaml(yaml_text))[0]
            assert quiz.preserve_yaml_order is False, f"Failed for {points_order}"


class TestListFormatWithConfig:
    """Tests for list format with _config options."""

    def test_list_format_with_repeat(self):
        """List format should support _config.repeat."""
        yaml_text = """
name: "Repeat Quiz"
questions:
  - name: "Repeated Question"
    points: 5
    class: FromText
    _config:
      repeat: 3
    kwargs:
      text: "Same question, different seeds"
"""
        quiz = Quiz.from_yaml(_write_tmp_yaml(yaml_text))[0]
        # Should create 3 copies
        assert len(quiz.questions) == 3

    def test_list_format_mixed_points(self):
        """List format should handle questions with various point values."""
        yaml_text = """
name: "Mixed Points Quiz"
questions:
  - name: "Five Points"
    points: 5
    class: FromText
    kwargs:
      text: "5 pts"
  - name: "Ten Points"
    points: 10
    class: FromText
    kwargs:
      text: "10 pts"
  - name: "Three Points"
    points: 3
    class: FromText
    kwargs:
      text: "3 pts"
  - name: "Seven Points"
    points: 7
    class: FromText
    kwargs:
      text: "7 pts"
"""
        quiz = Quiz.from_yaml(_write_tmp_yaml(yaml_text))[0]
        ordered = quiz.get_ordered_questions()

        # List format preserves order by default
        names = [q.name for q in ordered]
        assert names == ["Five Points", "Ten Points", "Three Points", "Seven Points"]


class TestListVsMappingFormat:
    """Tests comparing list format to mapping format behavior."""

    def test_list_format_default_preserves_order(self):
        """List format should preserve order by default (no question_order needed)."""
        yaml_text = """
name: "List Default"
questions:
  - name: "A"
    points: 1
    class: FromText
    kwargs:
      text: "A"
  - name: "B"
    points: 99
    class: FromText
    kwargs:
      text: "B"
"""
        quiz = Quiz.from_yaml(_write_tmp_yaml(yaml_text))[0]
        assert quiz.preserve_yaml_order is True
        ordered = quiz.get_ordered_questions()
        assert [q.name for q in ordered] == ["A", "B"]

    def test_mapping_format_default_orders_by_points(self):
        """Mapping format should order by points by default."""
        yaml_text = """
name: "Mapping Default"
questions:
  5:
    "Question A":
      class: FromText
      kwargs:
        text: "A"
    "Question B":
      class: FromText
      kwargs:
        text: "B"
"""
        quiz = Quiz.from_yaml(_write_tmp_yaml(yaml_text))[0]
        # Mapping format defaults to point ordering
        assert quiz.preserve_yaml_order is False


class TestListFormatEdgeCases:
    """Edge case tests for list format."""

    def test_empty_list_creates_empty_quiz(self):
        """Empty questions list should create quiz with no questions."""
        yaml_text = """
name: "Empty Quiz"
questions: []
"""
        quiz = Quiz.from_yaml(_write_tmp_yaml(yaml_text))[0]
        assert len(quiz.questions) == 0

    def test_single_question_list(self):
        """Single question in list format should work."""
        yaml_text = """
name: "Single Question"
questions:
  - name: "Only Question"
    points: 10
    class: FromText
    kwargs:
      text: "The one"
"""
        quiz = Quiz.from_yaml(_write_tmp_yaml(yaml_text))[0]
        assert len(quiz.questions) == 1
        assert quiz.questions[0].name == "Only Question"

    def test_list_format_missing_name_raises(self):
        """List format entry without name should raise error."""
        yaml_text = """
name: "Missing Name Quiz"
questions:
  - points: 5
    class: FromText
    kwargs:
      text: "No name"
"""
        with pytest.raises(ValueError, match="name"):
            Quiz.from_yaml(_write_tmp_yaml(yaml_text))

    def test_list_format_missing_points_raises(self):
        """List format entry without points should raise error."""
        yaml_text = """
name: "Missing Points Quiz"
questions:
  - name: "No Points"
    class: FromText
    kwargs:
      text: "Missing points"
"""
        with pytest.raises(ValueError, match="points"):
            Quiz.from_yaml(_write_tmp_yaml(yaml_text))

    def test_list_format_class_defaults_to_fromtext(self):
        """List format without class should default to FromText."""
        yaml_text = """
name: "Default Class Quiz"
questions:
  - name: "No Class Specified"
    points: 5
    kwargs:
      text: "Should use FromText"
"""
        quiz = Quiz.from_yaml(_write_tmp_yaml(yaml_text))[0]
        # Should not raise - FromText is default
        assert len(quiz.questions) == 1


class TestOrderingWithOptimizeSpace:
    """Tests for question ordering with optimize_space flag."""

    def test_get_ordered_questions_respects_preserve_yaml_order(self):
        """get_ordered_questions should respect preserve_yaml_order setting."""
        yaml_text = """
name: "Preserve Order Quiz"
question_order: yaml
questions:
  - name: "First"
    points: 1
    class: FromText
    kwargs:
      text: "First"
  - name: "Second"
    points: 100
    class: FromText
    kwargs:
      text: "Second"
"""
        quiz = Quiz.from_yaml(_write_tmp_yaml(yaml_text))[0]

        # With preserve_yaml_order=True (quiz default from question_order: yaml)
        ordered = quiz.get_ordered_questions()
        assert [q.name for q in ordered] == ["First", "Second"]

        # Override at call time
        ordered_by_points = quiz.get_ordered_questions(preserve_yaml_order=False)
        # Should now order by points (high first)
        assert ordered_by_points[0].points_value >= ordered_by_points[1].points_value

    def test_quiz_iteration_uses_ordered_questions(self):
        """Iterating over quiz should use get_ordered_questions."""
        yaml_text = """
name: "Iteration Quiz"
question_order: yaml
questions:
  - name: "First"
    points: 1
    class: FromText
    kwargs:
      text: "First"
  - name: "Second"
    points: 100
    class: FromText
    kwargs:
      text: "Second"
"""
        quiz = Quiz.from_yaml(_write_tmp_yaml(yaml_text))[0]

        # Iteration should respect the quiz's preserve_yaml_order setting
        names = [q.name for q in quiz]
        assert names == ["First", "Second"]
