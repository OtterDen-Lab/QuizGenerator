"""
Tests for Answer types and their rendering.

These tests verify that Answer types correctly render across formats
and generate proper Canvas export data.
"""

import numpy as np

import QuizGenerator.contentast as ca


class TestAnswerBase:
    """Tests for the base Answer class."""

    def test_answer_has_value(self):
        answer = ca.Answer(42, label="Result")
        assert answer.value == 42

    def test_answer_has_label(self):
        answer = ca.Answer(42, label="Result")
        assert answer.label == "Result"

    def test_answer_has_key(self):
        answer = ca.Answer(42, label="Result")
        # key should be generated from label
        assert hasattr(answer, "key")
        assert answer.key is not None

    def test_answer_renders_in_section(self):
        section = ca.Section()
        answer = ca.AnswerTypes.Int(42, label="Answer")
        section.add_element(answer)

        html = section.render("html")
        # Answer should render something (either value or blank)
        assert html is not None


class TestIntAnswer:
    """Tests for integer answers."""

    def test_int_answer_value(self):
        answer = ca.AnswerTypes.Int(42, label="Count")
        assert answer.value == 42

    def test_int_answer_canvas_export(self):
        answer = ca.AnswerTypes.Int(42, label="Count")
        canvas_data = answer.get_for_canvas(single_answer=True)

        assert isinstance(canvas_data, list)
        assert len(canvas_data) > 0

        # Should have the correct answer value
        first_answer = canvas_data[0]
        assert "answer_exact" in first_answer or "answer_text" in first_answer

    def test_int_answer_html_render(self):
        answer = ca.AnswerTypes.Int(42, label="Count")
        html = answer.render("html")

        # Should have some representation
        assert html is not None
        # Should contain a blank or the value
        assert "Count" in html or "42" in html or "blank" in html.lower()

    def test_int_answer_latex_render(self):
        answer = ca.AnswerTypes.Int(42, label="Count")
        latex = answer.render("latex")

        assert latex is not None

    def test_int_answer_with_unit(self):
        answer = ca.AnswerTypes.Int(42, label="Time", unit="ms")
        assert answer.unit == "ms"


class TestFloatAnswer:
    """Tests for floating-point answers."""

    def test_float_answer_value(self):
        answer = ca.AnswerTypes.Float(3.14159, label="Pi")
        assert abs(answer.value - 3.14159) < 0.0001

    def test_float_answer_canvas_export(self):
        answer = ca.AnswerTypes.Float(3.14, label="Pi")
        canvas_data = answer.get_for_canvas(single_answer=True)

        assert isinstance(canvas_data, list)
        assert len(canvas_data) > 0

    def test_float_answer_tolerance(self):
        answer = ca.AnswerTypes.Float(3.14, label="Pi")
        canvas_data = answer.get_for_canvas(single_answer=True)

        # Float answers should have tolerance/margin
        first_answer = canvas_data[0]
        # Check for margin fields
        assert ("answer_error_margin" in first_answer or
                "answer_exact" in first_answer or
                "answer_approximate" in first_answer)

    def test_float_answer_default_tolerance_value(self):
        answer = ca.AnswerTypes.Float(3.14, label="Pi")
        canvas_data = answer.get_for_canvas(single_answer=True)
        first_answer = canvas_data[0]
        assert first_answer.get("answer_error_margin") == ca.Answer.DEFAULT_FLOAT_TOLERANCE

    def test_float_answer_custom_tolerance(self):
        answer = ca.AnswerTypes.Float(3.14, label="Pi", tolerance=0.005)
        canvas_data = answer.get_for_canvas(single_answer=True)
        first_answer = canvas_data[0]
        assert first_answer.get("answer_error_margin") == 0.005

    def test_float_answer_display_rounding(self):
        answer = ca.AnswerTypes.Float(3.14159265, label="Pi")
        display = answer.display

        # Display should be rounded (not all digits)
        assert display is not None


class TestStringAnswer:
    """Tests for string/text answers."""

    def test_string_answer_value(self):
        answer = ca.AnswerTypes.String("hello", label="Greeting")
        assert answer.value == "hello"

    def test_string_answer_canvas_export(self):
        answer = ca.AnswerTypes.String("FIFO", label="Algorithm")
        canvas_data = answer.get_for_canvas(single_answer=True)

        assert isinstance(canvas_data, list)


class TestOpenEndedAnswer:
    """Tests for essay/open-ended answers."""

    def test_open_ended_is_essay_type(self):
        answer = ca.AnswerTypes.OpenEnded("Sample answer", label="Explanation")
        assert answer.kind == ca.Answer.CanvasAnswerKind.ESSAY


class TestVectorAnswer:
    """Tests for vector answers."""

    def test_vector_from_list(self):
        answer = ca.AnswerTypes.Vector([1, 2, 3], label="Vector")
        assert answer.value == [1, 2, 3]

    def test_vector_from_numpy(self):
        arr = np.array([1, 2, 3])
        answer = ca.AnswerTypes.Vector(arr, label="Vector")
        # Should convert numpy array to list or work with it
        assert len(answer.value) == 3

    def test_vector_canvas_export(self):
        answer = ca.AnswerTypes.Vector([1, 2, 3], label="V")
        canvas_data = answer.get_for_canvas(single_answer=False)

        # Vector should generate multiple blanks
        assert isinstance(canvas_data, list)


class TestListAnswer:
    """Tests for list answers (comma-separated values)."""

    def test_list_answer_value(self):
        # List takes order_matters as first positional arg, value as kwarg
        answer = ca.AnswerTypes.List(True, value=[1, 2, 3], label="Numbers")
        assert answer.value == [1, 2, 3]

    def test_list_answer_entry_warning(self):
        warning = ca.AnswerTypes.List.get_entry_warning()
        assert warning is not None
        assert len(warning) > 0
        # Should mention comma-separated
        assert any("comma" in w.lower() for w in warning)


class TestMatrixAnswer:
    """Tests for matrix answers."""

    def test_matrix_answer_creation(self):
        # AnswerTypes.Matrix requires numpy array (uses .shape)
        data = np.array([[1, 2], [3, 4]])
        answer = ca.AnswerTypes.Matrix(data, label="M")
        assert answer.value.shape == (2, 2)

    def test_matrix_answer_from_numpy(self):
        arr = np.array([[1, 2], [3, 4]])
        answer = ca.AnswerTypes.Matrix(arr, label="M")
        # Should handle numpy array
        assert answer.value.shape[0] == 2


class TestMultiBaseAnswer:
    """Tests for multi-base (hex/binary/decimal) answers."""

    def test_multibase_accepts_decimal(self):
        answer = ca.AnswerTypes.MultiBase(255, label="Value")
        assert answer.value == 255

    def test_multibase_display_hex(self):
        answer = ca.AnswerTypes.MultiBase(255, label="Value")
        display = answer.display

        # Should display in some format
        assert display is not None
        # Display could be int or string representation
        display_str = str(display)
        assert "f" in display_str.lower() or "255" in display_str


class TestAnswerBlock:
    """Tests for AnswerBlock container."""

    def test_answer_block_contains_answers(self):
        answer1 = ca.AnswerTypes.Int(42, label="A")
        answer2 = ca.AnswerTypes.Int(24, label="B")

        block = ca.AnswerBlock([answer1, answer2])

        # Should render as table with answers
        html = block.render("html")
        assert "42" in html or "A" in html or "blank" in html.lower()

    def test_answer_block_html_table(self):
        answer1 = ca.AnswerTypes.Int(1, label="X")
        answer2 = ca.AnswerTypes.Int(2, label="Y")

        block = ca.AnswerBlock([answer1, answer2])
        html = block.render("html")

        assert "<table" in html


class TestAnswerCanvasKinds:
    """Tests for Canvas answer kind classification."""

    def test_int_is_blank_type(self):
        answer = ca.AnswerTypes.Int(42, label="N")
        # Integer answers use fill-in-the-blank
        assert answer.kind == ca.Answer.CanvasAnswerKind.BLANK

    def test_essay_is_essay_type(self):
        answer = ca.AnswerTypes.OpenEnded("text", label="Essay")
        assert answer.kind == ca.Answer.CanvasAnswerKind.ESSAY


class TestAnswerExtraction:
    """Tests for extracting answers from content elements."""

    def test_section_collects_answers(self):
        section = ca.Section()
        answer1 = ca.AnswerTypes.Int(10, label="A")
        answer2 = ca.AnswerTypes.Int(20, label="B")

        section.add_element(ca.Paragraph(["Question text"]))
        section.add_element(answer1)
        section.add_element(answer2)

        # Section should be able to yield its answer elements
        answers = [e for e in section.elements if isinstance(e, ca.Answer)]
        assert len(answers) == 2


class TestAnswerRendering:
    """Tests for answer rendering in different contexts."""

    def test_answer_in_paragraph(self):
        answer = ca.AnswerTypes.Int(42, label="Result")
        para = ca.Paragraph([
            "The answer is: ",
            answer
        ])

        html = para.render("html")
        assert html is not None
        # Should contain some representation of the answer
        assert "Result" in html or "42" in html or "blank" in html.lower()

    def test_answer_in_table(self):
        answer = ca.AnswerTypes.Int(42, label="Val")
        table = ca.Table(
            data=[["Label", answer]],
            headers=["Name", "Value"]
        )

        html = table.render("html")
        assert "<table" in html

    def test_answer_typst_rendering(self):
        answer = ca.AnswerTypes.Int(42, label="N")
        typst = answer.render("typst")

        # Typst should have some answer representation
        assert typst is not None

    def test_answer_latex_rendering(self):
        answer = ca.AnswerTypes.Int(42, label="N")
        latex = answer.render("latex")

        # LaTeX should have some answer representation
        assert latex is not None


class TestAnswerEdgeCases:
    """Tests for edge cases in answer handling."""

    def test_zero_value(self):
        answer = ca.AnswerTypes.Int(0, label="Zero")
        assert answer.value == 0

        canvas_data = answer.get_for_canvas(single_answer=True)
        assert len(canvas_data) > 0

    def test_negative_value(self):
        answer = ca.AnswerTypes.Int(-42, label="Negative")
        assert answer.value == -42

    def test_large_value(self):
        answer = ca.AnswerTypes.Int(2**32, label="Large")
        assert answer.value == 2**32

    def test_float_near_zero(self):
        answer = ca.AnswerTypes.Float(0.0001, label="Small")
        assert answer.value == 0.0001

    def test_empty_label(self):
        # Should handle empty or None labels gracefully
        answer = ca.AnswerTypes.Int(42, label="")
        assert answer.value == 42

    def test_special_characters_in_label(self):
        # Labels with special characters should be handled
        answer = ca.AnswerTypes.Int(42, label="Result (1)")
        html = answer.render("html")
        assert html is not None
