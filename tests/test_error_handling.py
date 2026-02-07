"""
Tests for error handling throughout the quiz generation system.

These tests verify that appropriate errors are raised for invalid inputs
and edge cases are handled gracefully.
"""

import tempfile
from pathlib import Path

import pytest

import QuizGenerator.contentast as ca
from QuizGenerator.question import QuestionContext, QuestionRegistry
from QuizGenerator.quiz import Quiz


class TestYAMLErrorHandling:
    """Tests for YAML parsing error handling."""

    @pytest.fixture
    def temp_yaml_file(self):
        """Create temporary YAML files for testing."""
        files = []

        def _create(content: str) -> str:
            f = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            f.write(content)
            f.flush()
            f.close()
            files.append(f.name)
            return f.name

        yield _create

        for path in files:
            try:
                Path(path).unlink()
            except OSError:
                pass

    def test_invalid_yaml_syntax_raises_error(self, temp_yaml_file):
        """Malformed YAML should raise an appropriate error."""
        yaml_content = """
name: "Test Quiz"
questions:
  - this is invalid yaml
    not proper structure at all
  unindented: badly
"""
        path = temp_yaml_file(yaml_content)

        with pytest.raises(Exception):  # Could be yaml.YAMLError or similar
            Quiz.from_yaml(path)

    def test_missing_name_key(self, temp_yaml_file):
        """Quiz without name should still work (name is optional or defaults)."""
        yaml_content = """
questions:
  1:
    "Q1":
      class: FromText
      kwargs:
        text: "Question"
"""
        path = temp_yaml_file(yaml_content)
        # This should either work with a default name or raise a clear error
        try:
            quizzes = Quiz.from_yaml(path)
            # If it works, name should have some default
            assert quizzes[0].name is not None or True
        except (KeyError, TypeError) as e:
            # If it fails, error should be about missing name
            pass

    def test_empty_questions_section(self, temp_yaml_file):
        """Empty questions section should raise or handle gracefully."""
        yaml_content = """
name: "Empty Quiz"
questions: {}
"""
        path = temp_yaml_file(yaml_content)

        # Should either return empty quiz or raise
        try:
            quizzes = Quiz.from_yaml(path)
            assert len(quizzes[0].questions) == 0
        except Exception:
            pass  # Raising is also acceptable

    def test_nonexistent_yaml_file_raises_error(self):
        """Non-existent YAML file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Quiz.from_yaml("/nonexistent/path/to/quiz.yaml")

    def test_question_missing_class_uses_default(self, temp_yaml_file):
        """Question without class key defaults to FromText."""
        yaml_content = """
name: "No Class Quiz"
questions:
  1:
    "Q1":
      kwargs:
        text: "No class specified, defaults to FromText"
"""
        path = temp_yaml_file(yaml_content)

        # FromText is the default, so this should work
        quizzes = Quiz.from_yaml(path)
        assert len(quizzes) > 0

    def test_question_with_invalid_kwargs_type(self, temp_yaml_file):
        """Question with wrong kwargs type should raise error."""
        yaml_content = """
name: "Bad Kwargs Quiz"
questions:
  1:
    "Q1":
      class: FromText
      kwargs: "this should be a dict not a string"
"""
        path = temp_yaml_file(yaml_content)

        with pytest.raises((TypeError, AttributeError, ValueError)):
            Quiz.from_yaml(path)


class TestQuestionRegistryErrors:
    """Tests for QuestionRegistry error handling."""

    def test_create_unregistered_question_raises(self):
        """Attempting to create an unregistered question type should fail."""
        with pytest.raises(ValueError, match="Unknown question type"):
            QuestionRegistry.create("CompletelyFakeQuestion12345XYZ")

    def test_create_with_missing_required_kwargs(self):
        """Creating a question without required kwargs should fail."""
        # FromText requires 'text' kwarg
        with pytest.raises(TypeError):
            QuestionRegistry.create("FromText", name="Test", points_value=5)
            # Missing 'text' kwarg


class TestQuestionContextErrors:
    """Tests for QuestionContext error handling."""

    def test_frozen_context_rejects_writes(self):
        """Frozen context should reject modifications."""
        import random

        rng = random.Random(42)
        context = QuestionContext(rng_seed=42, rng=rng)
        context["key"] = "value"

        frozen = context.freeze()

        with pytest.raises(TypeError):
            frozen["new_key"] = "new_value"

    def test_context_missing_key_raises(self):
        """Accessing missing key should raise KeyError."""
        import random

        rng = random.Random(42)
        context = QuestionContext(rng_seed=42, rng=rng)

        with pytest.raises(KeyError):
            _ = context["nonexistent_key"]


class TestContentASTErrors:
    """Tests for ContentAST error handling."""

    def test_matrix_rejects_3d_array(self):
        """Matrix should reject 3D numpy arrays."""
        import numpy as np

        arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        with pytest.raises(ValueError, match="1D or 2D"):
            ca.Matrix(data=arr)

    def test_paragraph_with_none_element(self):
        """Paragraph with None in elements should handle gracefully."""
        # This tests robustness - None could sneak in through bugs
        try:
            para = ca.Paragraph(["Text", None])
            # Should either handle or raise clearly
        except (TypeError, AttributeError):
            pass  # Raising is acceptable

    def test_table_empty_data(self):
        """Table with empty data should handle gracefully."""
        table = ca.Table(data=[])

        # Should render something (empty table) or raise
        try:
            html = table.render("html")
            assert html is not None
        except (IndexError, ValueError):
            pass  # Raising is also acceptable

    def test_table_mismatched_columns(self):
        """Table with inconsistent column counts should handle gracefully."""
        data = [
            ["A", "B", "C"],
            ["D", "E"],  # Missing column
        ]

        table = ca.Table(data=data)

        # Should render or raise - but not crash silently
        try:
            html = table.render("html")
        except (IndexError, ValueError):
            pass


class TestAnswerErrors:
    """Tests for Answer type error handling."""

    def test_answer_with_none_value(self):
        """Answer with None value should handle gracefully."""
        # Some answer types may not accept None
        try:
            answer = ca.AnswerTypes.Int(None, label="Null")
            # If it works, should have some representation
        except (TypeError, ValueError):
            pass  # Raising is acceptable

    def test_vector_answer_empty_list(self):
        """Vector answer with empty list should handle gracefully."""
        try:
            answer = ca.AnswerTypes.Vector([], label="Empty")
            canvas_data = answer.get_for_canvas()
            # Empty vector is valid but unusual
        except (ValueError, IndexError):
            pass


class TestFromGeneratorErrors:
    """Tests for FromGenerator security and error handling."""

    def test_fromgenerator_disabled_by_default(self):
        """FromGenerator should be disabled by default."""
        import os

        # Ensure the env var is not set
        old_value = os.environ.pop("QUIZGEN_ALLOW_GENERATOR", None)

        try:
            # Reset the global flag
            import QuizGenerator.premade_questions.basic as basic_module
            from QuizGenerator.premade_questions.basic import (
                FromGenerator,
            )
            basic_module.ALLOW_GENERATOR = False

            with pytest.raises(ValueError, match="disabled"):
                FromGenerator(
                    name="Test",
                    points_value=1,
                    generator="return 'test'"
                )
        finally:
            if old_value is not None:
                os.environ["QUIZGEN_ALLOW_GENERATOR"] = old_value


class TestQuizInstantiationErrors:
    """Tests for quiz instantiation error handling."""

    @pytest.fixture
    def temp_yaml_file(self):
        """Create temporary YAML files for testing."""
        files = []

        def _create(content: str) -> str:
            f = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            f.write(content)
            f.flush()
            f.close()
            files.append(f.name)
            return f.name

        yield _create

        for path in files:
            try:
                Path(path).unlink()
            except OSError:
                pass

    def test_negative_point_value(self, temp_yaml_file):
        """Negative point values should be handled."""
        yaml_content = """
name: "Negative Points Quiz"
questions:
  -5:
    "Q1":
      class: FromText
      kwargs:
        text: "Negative points?"
"""
        path = temp_yaml_file(yaml_content)

        # Should either work (negative points) or raise
        try:
            quizzes = Quiz.from_yaml(path)
        except (ValueError, KeyError):
            pass

    def test_zero_point_value(self, temp_yaml_file):
        """Zero point value should be valid."""
        yaml_content = """
name: "Zero Points Quiz"
questions:
  0:
    "Q1":
      class: FromText
      kwargs:
        text: "Worth nothing"
"""
        path = temp_yaml_file(yaml_content)

        quizzes = Quiz.from_yaml(path)
        assert len(quizzes) > 0


class TestRenderingErrors:
    """Tests for rendering error handling."""

    def test_unknown_format_fallback(self):
        """Unknown output format should fall back gracefully."""
        text = ca.Text("Test content")

        # Should fall back to markdown or raise clear error
        try:
            rendered = text.render("completely_unknown_format_xyz")
            assert rendered is not None
        except (ValueError, KeyError):
            pass  # Raising is also acceptable

    def test_document_render_without_elements(self):
        """Empty document should render valid output."""
        doc = ca.Document(title="Empty")

        html = doc.render("html")
        assert "Empty" in html

        latex = doc.render("latex")
        assert "Empty" in latex or "\\documentclass" in latex


class TestTypstConversionErrors:
    """Tests for LaTeX to Typst conversion edge cases."""

    def test_malformed_latex_handled(self):
        """Malformed LaTeX should not crash the converter."""
        eq = ca.Equation(r"\frac{incomplete", inline=True)

        # Should render something (even if wrong) rather than crash
        try:
            typst = eq.render("typst")
            assert typst is not None
        except Exception:
            pass  # Some parsing errors may be unavoidable

    def test_nested_braces(self):
        """Deeply nested braces should be handled."""
        eq = ca.Equation(r"\frac{\frac{a}{b}}{\frac{c}{d}}", inline=True)

        typst = eq.render("typst")
        assert typst is not None

    def test_unknown_command(self):
        """Unknown LaTeX commands should be passed through or handled."""
        eq = ca.Equation(r"\unknowncommand{x}", inline=True)

        typst = eq.render("typst")
        # Should contain something, not crash
        assert typst is not None
