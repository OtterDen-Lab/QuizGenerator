from __future__ import annotations

import pytest
import yaml

from QuizGenerator.generation.iframe_export import export_iframe_variants
from QuizGenerator.generation.premade_questions.math130.questions import (
  LinearModelPrediction,
)
from QuizGenerator.generation.question import QuestionRegistry
from QuizGenerator.generation.quiz import Quiz


def _math130_quiz() -> Quiz:
  QuestionRegistry.load_premade_questions()
  exam = {
    "name": "Iframe test",
    "questions": {
      2: {
        "First": {"class": "math130.FunctionEvaluation"},
      },
      1: {
        "Second": {"class": "math130.LogarithmConversion"},
      },
    },
  }
  return Quiz.from_exam_dicts([exam])[0]


def test_iframe_export_writes_source_ordered_bare_fragments(tmp_path):
  output_dir = tmp_path / "MATH-130"
  written = export_iframe_variants(
    [_math130_quiz()],
    variations=2,
    output_dir=output_dir,
    base_seed=123,
  )

  assert [path.relative_to(output_dir).as_posix() for path in written] == [
    "q01/v001.yaml",
    "q01/v002.yaml",
    "q02/v001.yaml",
    "q02/v002.yaml",
  ]

  first = yaml.safe_load((output_dir / "q01" / "v001.yaml").read_text())
  second = yaml.safe_load((output_dir / "q02" / "v001.yaml").read_text())
  raw_first = (output_dir / "q01" / "v001.yaml").read_text()
  assert list(first) == ["question_html", "answer", "explanation_html"]
  assert "question_html: |-" in raw_first
  assert "explanation_html: |-" in raw_first
  assert "\\n" not in raw_first
  assert "<html" not in first["question_html"]
  assert "quizgen-answer-input" in first["question_html"]
  assert first["answer"][0]["accepted_values"]
  assert "\\log_" in second["question_html"]


def test_iframe_export_recreates_existing_output_deterministically(tmp_path):
  output_dir = tmp_path / "MATH-130"
  output_dir.mkdir()
  (output_dir / "stale.txt").write_text("old")

  export_iframe_variants([_math130_quiz()], variations=1, output_dir=output_dir, base_seed=9)
  first_run = (output_dir / "q01" / "v001.yaml").read_text()
  assert not (output_dir / "stale.txt").exists()

  export_iframe_variants([_math130_quiz()], variations=1, output_dir=output_dir, base_seed=9)
  assert (output_dir / "q01" / "v001.yaml").read_text() == first_run


def test_iframe_export_rejects_multiple_quizzes(tmp_path):
  quiz = _math130_quiz()
  with pytest.raises(ValueError, match="exactly one quiz"):
    export_iframe_variants([quiz, quiz], variations=1, output_dir=tmp_path / "output")


def test_linear_model_explanation_works_the_specific_problem():
  explanation = LinearModelPrediction._build_explanation({
    "intercept": 10,
    "slope": 8,
    "x": 8,
    "value": 74,
  }).render("html")

  assert "replace m with 8" in explanation
  assert "C(8) = 10 + 8(8) = 10 + 64 = 74" in explanation
