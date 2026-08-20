"""Private adapter from QuizGenerator instances to QTI-neutral values.

This module deliberately does not alter Question, QuestionGroup, or Quiz.
"""
from __future__ import annotations

from dataclasses import dataclass

import QuizGenerator.generation.contentast as ca


class QtiCompatibilityError(ValueError):
  """A generated question cannot be represented by this prototype."""


@dataclass(frozen=True)
class ExportedQuestion:
  title: str
  points: float
  body_html: str
  answers: list[ca.Answer]
  answer_kind: ca.Answer.CanvasAnswerKind
  can_be_numerical: bool


def adapt_instance(question, instance, *, title: str, image_upload=None) -> ExportedQuestion:
  """Convert one generated instance, rejecting unsupported/mixed interactions."""
  answers = [answer for answer in instance.answers if not answer.pdf_only]
  kinds = {answer.kind for answer in answers}
  unsupported = {ca.Answer.CanvasAnswerKind.ESSAY}
  if kinds & unsupported:
    names = ", ".join(sorted(kind.value for kind in kinds & unsupported))
    raise QtiCompatibilityError(f"{title}: unsupported QTI interaction: {names}")
  if len(kinds) > 1:
    names = ", ".join(sorted(kind.value for kind in kinds))
    raise QtiCompatibilityError(f"{title}: mixed answer interactions are unsupported: {names}")
  if not answers:
    raise QtiCompatibilityError(f"{title}: question has no exportable answers")

  kind = next(iter(kinds))
  if kind not in {
      ca.Answer.CanvasAnswerKind.BLANK,
      ca.Answer.CanvasAnswerKind.MULTIPLE_ANSWER,
      ca.Answer.CanvasAnswerKind.MULTIPLE_DROPDOWN,
      ca.Answer.CanvasAnswerKind.MATCHING,
  }:
    raise QtiCompatibilityError(f"{title}: unsupported QTI interaction: {kind.value}")
  if kind == ca.Answer.CanvasAnswerKind.MULTIPLE_ANSWER and len(answers) != 1:
    raise QtiCompatibilityError(f"{title}: multiple-choice export requires exactly one answer")
  if kind == ca.Answer.CanvasAnswerKind.MATCHING and not all(
      isinstance(answer, ca.MatchingAnswer) for answer in answers
  ):
    raise QtiCompatibilityError(f"{title}: matching export requires matching answers")

  body_html = instance.body.render("html", upload_func=image_upload or (lambda _: ""))
  return ExportedQuestion(
    title=title,
    points=float(instance.value),
    body_html=body_html,
    answers=answers,
    answer_kind=kind,
    can_be_numerical=instance.can_be_numerical,
  )
