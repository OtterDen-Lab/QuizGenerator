"""Small reusable building blocks for course-specific mathematics questions."""

import random

import QuizGenerator.generation.contentast as ca
from QuizGenerator.generation.question import Question, QuestionRegistry


class CourseMathQuestion(Question):
  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.MATH)
    super().__init__(*args, **kwargs)


def make_question(course, name, context_builder):
  """Register a simple, deterministic math question built from a context dict."""
  def _build_context(cls, *, rng_seed=None, **kwargs):
    return context_builder(random.Random(rng_seed), **kwargs)

  def _build_body(cls, context):
    body = ca.Section([ca.Paragraph([context["question"]])])
    if context.get("equation"):
      body.add_element(ca.Equation(context["equation"]))
    if context.get("code"):
      body.add_element(ca.Code(context["code"]))
    body.add_element(ca.AnswerBlock(_answer(context)))
    return body

  def _build_explanation(cls, context):
    explanation = ca.Section()
    for paragraph in context["explanation"]:
      explanation.add_element(ca.Paragraph([paragraph]))
    for equation in context.get("explanation_equations", []):
      explanation.add_element(ca.Equation(equation))
    return explanation

  generated = type(name, (CourseMathQuestion,), {
    "__doc__": f"A {course.upper()} readiness question: {name}.",
    "_build_context": classmethod(_build_context),
    "_build_body": classmethod(_build_body),
    "_build_explanation": classmethod(_build_explanation),
  })
  return QuestionRegistry.register(f"{course}.{name}")(generated)


def _answer(context):
  kind = context.get("answer_kind", "float")
  value = context["answer"]
  label = context.get("label", "Answer")
  if kind == "int":
    return ca.AnswerTypes.Int(value, label=label)
  if kind == "string":
    return ca.AnswerTypes.String(value, label=label)
  if kind == "list":
    return ca.AnswerTypes.List(False, [str(item) for item in value], label=label)
  return ca.AnswerTypes.Float(value, label=label)
