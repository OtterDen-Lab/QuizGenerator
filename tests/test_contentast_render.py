import QuizGenerator.contentast as ca
from QuizGenerator.question import Question


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
