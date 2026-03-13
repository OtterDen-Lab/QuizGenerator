import QuizGenerator.generation.contentast as ca
from QuizGenerator.generation.premade_questions.cst334.memory_questions import (
    HierarchicalPaging,
    Paging,
    Segmentation,
)
from QuizGenerator.generation.question import Question


class _MatrixQuestion(Question):
    VERSION = "1.0"

    @classmethod
    def _build_context(cls, *, rng_seed=None, **kwargs):
        return super()._build_context(rng_seed=rng_seed, **kwargs)

    @classmethod
    def _build_body(cls, context):
        body = ca.Section()
        body.add_element(ca.Paragraph(["Compute:"]))
        body.add_element(ca.Matrix(data=[[1, 2], [3, 4]], bracket_type="b"))
        body.add_element(ca.AnswerTypes.Int(5, label="Result"))
        return body

    @classmethod
    def _build_explanation(cls, context):
        explanation = ca.Section()
        explanation.add_element(ca.Paragraph(["Example solution."]))
        return explanation


class _TableQuestion(Question):
    VERSION = "1.0"

    @classmethod
    def _build_context(cls, *, rng_seed=None, **kwargs):
        return super()._build_context(rng_seed=rng_seed, **kwargs)

    @classmethod
    def _build_body(cls, context):
        body = ca.Section()
        table = ca.Table(
            data=[
                [ca.Text("a"), ca.Text("b")],
                [ca.Text("1"), ca.Text("2")],
            ],
        )
        body.add_element(table)
        body.add_element(ca.AnswerTypes.Int(3, label="Sum"))
        return body

    @classmethod
    def _build_explanation(cls, context):
        explanation = ca.Section()
        explanation.add_element(ca.Paragraph(["Explanation."]))
        return explanation


class _PdfAidQuestion(Question):
    VERSION = "1.0"

    @classmethod
    def _build_context(cls, *, rng_seed=None, **kwargs):
        return super()._build_context(rng_seed=rng_seed, **kwargs)

    @classmethod
    def _build_body(cls, context):
        body = ca.Section()
        body.add_element(ca.Paragraph(["Main content"]))
        body.add_element(ca.PDFAid([ca.Text("Aid overlay")]))
        return body

    @classmethod
    def _build_explanation(cls, context):
        return ca.Section()


def test_contentast_renders_html_and_typst():
    q = _MatrixQuestion(name="M", points_value=1.0)
    instance = q.instantiate(rng_seed=123)
    ast = q._build_question_ast(instance)
    html = ast.render("html")
    typst = ast.render("typst")
    assert "Compute" in html
    assert "matrix" in html or "bmatrix" in html
    assert "question(" in typst


def test_table_renders_html_and_typst():
    q = _TableQuestion(name="T", points_value=1.0)
    instance = q.instantiate(rng_seed=123)
    ast = q._build_question_ast(instance)
    html = ast.render("html")
    typst = ast.render("typst")
    assert "<table" in html
    assert "Sum" in html
    assert "question(" in typst


def test_typst_question_reserve_height_uses_context():
    header = ca.Document.TYPST_HEADER
    assert "context {" in header
    assert "measure(body).height" in header


def test_typst_page_spacing_breaks_after_question_body():
    header = ca.Document.TYPST_HEADER
    assert "#if spacing < 99cm and pdf_aid == none [" in header
    assert "if spacing < 99cm and pdf_aid != none {" in header
    assert "measure(aid_block).height >= 7cm" in header
    assert "pagebreak(weak: true)" in header


def test_pdf_aid_is_passed_as_question_parameter_in_typst():
    q = _PdfAidQuestion(name="Aid", points_value=1.0, spacing=4)
    instance = q.instantiate(rng_seed=123)
    ast = q._build_question_ast(instance)

    html = ast.render("html")
    assert "Aid overlay" not in html

    typst = ast.render("typst")
    assert "pdf_aid:" in typst
    assert "Aid overlay" in typst

    typst_without_aid = ast.render("typst", show_pdf_aids=False)
    assert "pdf_aid:" not in typst_without_aid


def test_pdf_aid_can_be_disabled_per_question():
    q = _PdfAidQuestion(name="Aid", points_value=1.0, spacing=4, show_pdf_aids=False)
    instance = q.instantiate(rng_seed=123)
    ast = q._build_question_ast(instance)

    typst = ast.render("typst")
    assert "pdf_aid:" not in typst


def test_typst_header_renders_pdf_aid_for_page_spacing():
    header = ca.Document.TYPST_HEADER
    assert "#if spacing >= 99cm and pdf_aid != none [" in header
    assert "#v(1fr)" in header
    assert "#align(bottom + left)" in header


def test_section_typst_adds_spacing_between_top_level_elements():
    body = ca.Section()
    body.add_element(ca.Paragraph(["Intro text"]))
    body.add_element(ca.Table(headers=["A"], data=[["B"]]))
    body.add_element(ca.AnswerBlock([ca.AnswerTypes.String("x", label="Value")]))

    rendered = body.render("typst")

    assert rendered.count("#v(0.35cm)") == 2


def test_memory_question_bodies_include_typst_block_spacing():
    for question_cls in (Segmentation, Paging):
        question = question_cls(name=question_cls.__name__, points_value=8)
        instance = question.instantiate(rng_seed=123)
        rendered = instance.body.render("typst")

        assert "#v(0.35cm)" in rendered


def test_hierarchical_paging_uses_grouped_tables_for_compact_layout():
    question = HierarchicalPaging(name="HierarchicalPaging", points_value=8)
    instance = question.instantiate(rng_seed=123)
    rendered = instance.body.render("typst")

    assert rendered.count("#grid(") >= 2
    assert "Address Info:" in rendered
    assert "Page Directory:" in rendered
