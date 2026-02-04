#!env python
from __future__ import annotations

import QuizGenerator.contentast as ca
from QuizGenerator.question import Question, QuestionRegistry


@QuestionRegistry.register("ExampleSimpleQuestion")
class ExampleSimpleQuestion(Question):
  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    rng = context.rng
    max_value = kwargs.get("max_value", 10)

    left = rng.randint(1, max_value)
    right = rng.randint(1, max_value)
    context["left"] = left
    context["right"] = right
    context["result"] = left + right
    return context

  @classmethod
  def _build_body(cls, context) -> ca.Section:
    body = ca.Section([
      ca.Paragraph([
        "Compute the sum: ",
        ca.Equation(f"{context['left']} + {context['right']}", inline=True),
      ])
    ])

    body.add_element(ca.AnswerTypes.Int(context["result"], label="Sum"))
    return body

  @classmethod
  def _build_explanation(cls, context) -> ca.Section:
    explanation = ca.Section()
    explanation.add_element(
      ca.Paragraph([
        "Add the two values to get the sum.",
        ca.Equation(
          f"{context['left']} + {context['right']} = {context['result']}",
          inline=True
        ),
      ])
    )
    return explanation


@QuestionRegistry.register("ExampleContextQuestion")
class ExampleContextQuestion(Question):
  """
  Bare-bones example (context path): build a context dict, then render body/explanation.

  Use this when body/explanation need shared computed values.
  """

  VERSION = "1.0"

  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.MATH_GENERAL)
    super().__init__(*args, **kwargs)

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    rng = context.rng
    max_value = kwargs.get("max_value", 10)

    left = rng.randint(1, max_value)
    right = rng.randint(1, max_value)
    context["left"] = left
    context["right"] = right
    context["result"] = left + right
    return context

  @classmethod
  def _build_body(cls, context) -> ca.Section:
    body = ca.Section()

    body.add_element(
      ca.Paragraph([
        "Compute the sum: ",
        ca.Equation(f"{context['left']} + {context['right']}", inline=True),
      ])
    )

    answer = ca.AnswerTypes.Int(context["result"], label="Sum")
    body.add_element(ca.AnswerBlock([answer]))

    return body

  @classmethod
  def _build_explanation(cls, context) -> ca.Section:
    explanation = ca.Section()
    explanation.add_element(
      ca.Paragraph([
        "Add the two values to get the sum.",
        ca.Equation(
          f"{context['left']} + {context['right']} = {context['result']}",
          inline=True
        ),
      ])
    )
    return explanation
