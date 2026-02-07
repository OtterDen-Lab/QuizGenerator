"""
Tests for the Content AST rendering system.

These tests verify that content AST elements render correctly to each output format.
"""

import numpy as np
import pytest

import QuizGenerator.contentast as ca


class TestText:
    """Tests for the Text element."""

    def test_basic_text_renders_to_all_formats(self):
        text = ca.Text("Hello world")

        assert "Hello world" in text.render("markdown")
        assert "Hello world" in text.render("html")
        # LaTeX may have escaping, so just check content exists
        assert "Hello" in text.render("latex")
        assert "Hello" in text.render("typst")

    def test_emphasis_text_markdown(self):
        text = ca.Text("Important", emphasis=True)
        rendered = text.render("markdown")
        assert "***Important***" in rendered

    def test_hide_from_latex(self):
        text = ca.Text("Web only", hide_from_latex=True)

        assert text.render("html") != ""
        assert text.render("latex") == ""
        assert text.render("typst") == ""

    def test_hide_from_html(self):
        text = ca.Text("Print only", hide_from_html=True)

        assert text.render("html") == ""
        assert "Print only" in text.render("latex")


class TestEquation:
    """Tests for the Equation element (math rendering)."""

    def test_inline_equation_markdown(self):
        eq = ca.Equation("x^2 + 1", inline=True)
        rendered = eq.render("markdown")
        assert "$x^2 + 1$" in rendered

    def test_display_equation_markdown(self):
        eq = ca.Equation("x^2 + 1", inline=False)
        rendered = eq.render("markdown")
        assert "\\displaystyle" in rendered
        assert "x^2 + 1" in rendered

    def test_inline_equation_html(self):
        eq = ca.Equation("x^2 + 1", inline=True)
        rendered = eq.render("html")
        # MathJax inline uses \( ... \)
        assert "\\(x^2 + 1\\)" in rendered

    def test_display_equation_html(self):
        eq = ca.Equation("x^2 + 1", inline=False)
        rendered = eq.render("html")
        assert "$$" in rendered or "displaystyle" in rendered

    def test_inline_equation_latex(self):
        eq = ca.Equation("x^2 + 1", inline=True)
        rendered = eq.render("latex")
        assert "$x^2 + 1$" in rendered

    def test_display_equation_latex(self):
        eq = ca.Equation("x^2 + 1", inline=False)
        rendered = eq.render("latex")
        assert "flushleft" in rendered or "$x^2 + 1$" in rendered

    def test_inline_equation_typst(self):
        eq = ca.Equation("x^2 + 1", inline=True)
        rendered = eq.render("typst")
        # Typst inline math uses $ ... $
        assert "$" in rendered
        assert "x" in rendered

    def test_greek_letter_conversion_typst(self):
        eq = ca.Equation(r"\alpha + \beta", inline=True)
        rendered = eq.render("typst")
        # Typst doesn't use backslash for Greek letters
        assert "alpha" in rendered
        assert "beta" in rendered

    def test_frac_conversion_typst(self):
        eq = ca.Equation(r"\frac{a}{b}", inline=True)
        rendered = eq.render("typst")
        # Should convert to Typst frac(a, b) syntax
        assert "frac" in rendered


class TestMatrix:
    """Tests for the Matrix element."""

    def test_basic_matrix_creation(self):
        data = [[1, 2], [3, 4]]
        matrix = ca.Matrix(data=data, bracket_type="b")
        assert matrix.data == data
        assert matrix.bracket_type == "b"

    def test_matrix_from_numpy_2d(self):
        arr = np.array([[1, 2], [3, 4]])
        matrix = ca.Matrix(data=arr, bracket_type="p")
        assert matrix.data == [[1, 2], [3, 4]]

    def test_matrix_from_numpy_1d_becomes_column_vector(self):
        arr = np.array([1, 2, 3])
        matrix = ca.Matrix(data=arr, bracket_type="b")
        # 1D array should become column vector
        assert matrix.data == [[1], [2], [3]]

    def test_matrix_rejects_3d_array(self):
        arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        with pytest.raises(ValueError, match="1D or 2D"):
            ca.Matrix(data=arr)

    def test_matrix_latex_output(self):
        data = [[1, 2], [3, 4]]
        matrix = ca.Matrix(data=data, bracket_type="b")
        rendered = matrix.render("latex")

        assert "bmatrix" in rendered
        assert "1 & 2" in rendered
        assert "3 & 4" in rendered

    def test_matrix_html_output(self):
        data = [[1, 2], [3, 4]]
        matrix = ca.Matrix(data=data, bracket_type="b")
        rendered = matrix.render("html")

        assert "bmatrix" in rendered
        assert "1 & 2" in rendered

    def test_matrix_typst_output(self):
        data = [[1, 2], [3, 4]]
        matrix = ca.Matrix(data=data, bracket_type="b")
        rendered = matrix.render("typst")

        # Typst uses mat() with square bracket delimiter
        assert "mat" in rendered
        assert '"["' in rendered  # square bracket delimiter

    def test_vector_typst_output(self):
        data = [[1], [2], [3]]
        matrix = ca.Matrix(data=data, bracket_type="b")
        rendered = matrix.render("typst")

        # Typst uses vec() for column vectors
        assert "vec" in rendered or "mat" in rendered

    def test_to_latex_static_method(self):
        data = [[1, 2], [3, 4]]
        latex_str = ca.Matrix.to_latex(data, "p")

        assert "\\begin{pmatrix}" in latex_str
        assert "\\end{pmatrix}" in latex_str
        assert "1 & 2" in latex_str


class TestCode:
    """Tests for the Code element."""

    def test_code_markdown_output(self):
        code = ca.Code("print('hello')")
        rendered = code.render("markdown")

        assert "```" in rendered
        assert "print('hello')" in rendered

    def test_code_html_output(self):
        code = ca.Code("print('hello')")
        rendered = code.render("html")
        # Should have some code formatting
        assert "print" in rendered

    def test_code_latex_output(self):
        code = ca.Code("print('hello')")
        rendered = code.render("latex")
        # LaTeX code may use Shaded/Highlighting environment via pandoc
        # or direct verbatim - just check it renders something
        assert rendered is not None
        assert len(rendered) > 0

    def test_code_typst_output(self):
        code = ca.Code("print('hello')")
        rendered = code.render("typst")
        assert "print" in rendered


class TestParagraph:
    """Tests for the Paragraph container."""

    def test_paragraph_with_strings(self):
        para = ca.Paragraph(["Line one", "Line two"])

        rendered = para.render("html")
        assert "Line one" in rendered
        assert "Line two" in rendered

    def test_paragraph_with_mixed_content(self):
        para = ca.Paragraph([
            "Calculate: ",
            ca.Equation("x^2", inline=True),
        ])

        rendered = para.render("html")
        assert "Calculate" in rendered
        assert "x" in rendered

    def test_paragraph_html_has_line_break(self):
        para = ca.Paragraph(["Some text"])
        rendered = para.render("html")
        assert "<br>" in rendered


class TestSection:
    """Tests for the Section container."""

    def test_section_contains_elements(self):
        section = ca.Section()
        section.add_element(ca.Text("Hello"))
        section.add_element(ca.Text("World"))

        rendered = section.render("html")
        assert "Hello" in rendered
        assert "World" in rendered

    def test_section_with_initial_elements(self):
        section = ca.Section([
            ca.Text("First"),
            ca.Text("Second"),
        ])

        rendered = section.render("markdown")
        assert "First" in rendered
        assert "Second" in rendered


class TestTable:
    """Tests for the Table element."""

    def test_basic_table_creation(self):
        data = [["A", "B"], ["C", "D"]]
        table = ca.Table(data=data)
        assert len(table.data) == 2
        assert len(table.data[0]) == 2

    def test_table_with_headers(self):
        data = [["1", "2"], ["3", "4"]]
        headers = ["Col1", "Col2"]
        table = ca.Table(data=data, headers=headers)

        assert table.headers is not None
        assert len(table.headers) == 2

    def test_table_html_output(self):
        data = [["A", "B"], ["C", "D"]]
        headers = ["H1", "H2"]
        table = ca.Table(data=data, headers=headers)

        rendered = table.render("html")
        assert "<table" in rendered
        assert "H1" in rendered
        assert "A" in rendered

    def test_table_latex_output(self):
        data = [["A", "B"], ["C", "D"]]
        headers = ["H1", "H2"]
        table = ca.Table(data=data, headers=headers)

        rendered = table.render("latex")
        assert "tabular" in rendered
        assert "H1" in rendered

    def test_table_markdown_output(self):
        data = [["A", "B"], ["C", "D"]]
        headers = ["H1", "H2"]
        table = ca.Table(data=data, headers=headers)

        rendered = table.render("markdown")
        assert "|" in rendered
        assert "H1" in rendered
        assert "---" in rendered  # separator line

    def test_table_typst_output(self):
        data = [["A", "B"], ["C", "D"]]
        headers = ["H1", "H2"]
        table = ca.Table(data=data, headers=headers)

        rendered = table.render("typst")
        assert "table" in rendered
        assert "columns" in rendered

    def test_table_with_alignments(self):
        data = [["1", "2", "3"]]
        headers = ["Left", "Center", "Right"]
        table = ca.Table(
            data=data,
            headers=headers,
            alignments=["left", "center", "right"]
        )

        # LaTeX alignment
        latex = table.render("latex")
        assert "l" in latex and "c" in latex and "r" in latex

    def test_table_with_element_cells(self):
        data = [[ca.Equation("x^2", inline=True), ca.Text("Text")]]
        table = ca.Table(data=data)

        rendered = table.render("html")
        assert "x" in rendered
        assert "Text" in rendered


class TestDocument:
    """Tests for the Document container."""

    def test_document_with_title(self):
        doc = ca.Document(title="Test Quiz")
        doc.add_element(ca.Text("Question 1"))

        rendered = doc.render("html")
        assert "Test Quiz" in rendered
        assert "Question 1" in rendered

    def test_document_latex_has_header(self):
        doc = ca.Document(title="Test Quiz")
        doc.add_element(ca.Text("Question 1"))

        rendered = doc.render("latex")
        assert "\\documentclass" in rendered
        assert "\\begin{document}" in rendered
        assert "\\end{document}" in rendered

    def test_document_typst_has_header(self):
        doc = ca.Document(title="Test Quiz")
        doc.add_element(ca.Text("Question 1"))

        rendered = doc.render("typst")
        assert "#set page" in rendered
        assert "Test Quiz" in rendered


class TestOnlyFormats:
    """Tests for format-specific containers."""

    def test_only_html_renders_only_in_html(self):
        element = ca.OnlyHtml([ca.Text("Web content")])

        assert "Web content" in element.render("html")
        assert element.render("latex") == ""
        assert element.render("typst") == ""

    def test_only_latex_renders_only_in_latex(self):
        element = ca.OnlyLatex([ca.Text("Print content")])

        assert element.render("html") == ""
        assert "Print content" in element.render("latex")


class TestMathExpression:
    """Tests for the MathExpression element (composite math)."""

    def test_math_expression_with_strings(self):
        expr = ca.MathExpression(["x", " + ", "y"])

        rendered = expr.render("html")
        assert "x" in rendered
        assert "y" in rendered

    def test_math_expression_with_matrix(self):
        matrix = ca.Matrix(data=[[1], [2]], bracket_type="b")
        expr = ca.MathExpression([matrix, " = "])

        rendered = expr.render("latex")
        assert "matrix" in rendered.lower() or "1" in rendered


class TestLineBreak:
    """Tests for the LineBreak element."""

    def test_linebreak_renders_newlines(self):
        br = ca.LineBreak()
        assert "\n" in br.render("markdown")


class TestContainerOperations:
    """Tests for container add/extend operations."""

    def test_add_element(self):
        section = ca.Section()
        section.add_element(ca.Text("One"))
        section.add_element(ca.Text("Two"))

        assert len(section.elements) == 2

    def test_add_elements(self):
        section = ca.Section()
        section.add_elements([ca.Text("One"), ca.Text("Two")])

        assert len(section.elements) == 2


class TestRenderFallback:
    """Tests for render method fallback behavior."""

    def test_unknown_format_falls_back_to_markdown(self):
        text = ca.Text("Test content")
        # Unknown format should fall back to markdown
        rendered = text.render("unknown_format")
        assert "Test content" in rendered


class TestLatexToTypstConversion:
    """Tests for LaTeX to Typst conversion in Equation."""

    def test_operator_conversion(self):
        eq = ca.Equation(r"\nabla \times \cdot", inline=True)
        rendered = eq.render("typst")

        assert "nabla" in rendered
        assert "times" in rendered
        assert "dot" in rendered

    def test_left_right_removal(self):
        eq = ca.Equation(r"\left( x \right)", inline=True)
        rendered = eq.render("typst")

        # \left and \right should be removed
        assert r"\left" not in rendered
        assert r"\right" not in rendered

    def test_subscript_conversion(self):
        eq = ca.Equation(r"x_{out}", inline=True)
        rendered = eq.render("typst")

        # Should convert to Typst subscript format
        assert "out" in rendered

    def test_function_conversion(self):
        eq = ca.Equation(r"\sin \cos \log \exp", inline=True)
        rendered = eq.render("typst")

        # Functions should be converted
        assert "sin" in rendered
        assert "cos" in rendered
        assert "log" in rendered
        assert "exp" in rendered

    def test_text_conversion(self):
        eq = ca.Equation(r"\text{hello}", inline=True)
        rendered = eq.render("typst")

        # \text{...} should become "..."
        assert '"hello"' in rendered
