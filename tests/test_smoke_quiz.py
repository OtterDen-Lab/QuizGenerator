import pytest

import QuizGenerator.contentast as ca
from QuizGenerator.generate import generate_typst
from QuizGenerator.quiz import Quiz
from QuizGenerator.typst_utils import check_typst_available
from QuizGenerator.question import Question


class _ReserveHeightQuestion(Question):
    VERSION = "1.0"

    @classmethod
    def _build_context(cls, *, rng_seed=None, **kwargs):
        return super()._build_context(rng_seed=rng_seed, **kwargs)

    @classmethod
    def _build_body(cls, context):
        body = ca.Section()
        body.add_element(ca.Paragraph(["Reserve height test."]))
        body.add_element(ca.AnswerTypes.Int(1, label="Answer"))
        return body

    @classmethod
    def _build_explanation(cls, context):
        explanation = ca.Section()
        explanation.add_element(ca.Paragraph(["Explanation."]))
        return explanation


def test_smoke_quiz_typst_render(tmp_path):
    yaml_text = """
name: "Smoke Test Quiz"
questions:
  1:
    "Simple text question":
      class: FromText
      kwargs:
        text: "What is 2 + 2?"
"""
    yaml_path = tmp_path / "smoke.yaml"
    yaml_path.write_text(yaml_text)

    quizzes = Quiz.from_yaml(str(yaml_path))
    assert quizzes, "Expected at least one quiz from smoke YAML"

    quiz = quizzes[0]
    typst_text = quiz.get_quiz(rng_seed=123).render("typst")
    assert "question(" in typst_text


def test_typst_compile_with_reserve_height(tmp_path, monkeypatch):
    if not check_typst_available():
        pytest.skip("Typst not available")

    monkeypatch.chdir(tmp_path)
    q = _ReserveHeightQuestion(name="R", points_value=1.0)
    instance = q.instantiate(rng_seed=123)
    ast = q._build_question_ast(instance)
    ast.reserve_height_cm = 5

    doc = ca.Document(title="Typst Compile Test")
    doc.add_element(ast)
    typst_text = doc.render("typst", embed_images_typst=False)

    assert generate_typst(typst_text, remove_previous=True, name_prefix="typst-compile-test")
