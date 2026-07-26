"""Export generated question variants as bare iframe-ready YAML snippets."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import yaml

import QuizGenerator.generation.contentast as ca
from QuizGenerator.generation.question import QuestionGroup


class _IframeYamlDumper(yaml.SafeDumper):
  """Prefer readable block scalars for HTML fragments."""


def _represent_string(dumper, value):
  style = "|" if "\n" in value else None
  return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_IframeYamlDumper.add_representer(str, _represent_string)


def _clean_html_fragment(fragment: str) -> str:
  """Remove renderer-only blank lines while retaining intentional HTML lines."""
  return "\n".join(line.strip() for line in fragment.splitlines() if line.strip())


def _accepted_values(answer: ca.Answer) -> list[str]:
  """Return unique accepted answer strings in the same form Canvas receives."""
  accepted: list[str] = []
  try:
    for canvas_answer in answer.get_for_canvas():
      if canvas_answer.get("answer_weight", 0) <= 0:
        continue
      value = str(canvas_answer.get("answer_text", ""))
      if value not in accepted:
        accepted.append(value)
  except Exception:
    pass

  if not accepted:
    display = answer.get_display_string() if hasattr(answer, "get_display_string") else answer.value
    accepted.append(str(display))
  return accepted


def _answer_data(answer: ca.Answer) -> dict[str, object]:
  data: dict[str, object] = {
    "label": answer.label,
    "accepted_values": _accepted_values(answer),
    "kind": answer.kind.value,
  }
  if answer.unit:
    data["unit"] = answer.unit
  return data


def export_iframe_variants(
    quizzes,
    *,
    variations: int,
    output_dir: str | Path,
    base_seed: int | None = None,
) -> list[Path]:
  """Write one bare HTML/YAML fragment for each question and variation.

  The output directory is intentionally recreated on every run. It is a
  generated artifact, so stale question folders cannot remain after the source
  YAML changes.
  """
  if variations < 1:
    raise ValueError("iframe variations must be at least 1")
  if len(quizzes) != 1:
    raise ValueError("Iframe export supports exactly one quiz YAML document.")

  quiz = quizzes[0]
  if any(isinstance(question, QuestionGroup) for question in quiz.questions):
    raise ValueError("Iframe export does not support question groups.")

  destination = Path(output_dir)
  if destination.exists():
    if not destination.is_dir():
      raise ValueError(f"Iframe output path is not a directory: {destination}")
    shutil.rmtree(destination)
  destination.mkdir(parents=True)

  seed_rng = random.Random(base_seed)
  written: list[Path] = []
  for question_index, question in enumerate(quiz.questions, start=1):
    question_dir = destination / f"q{question_index:02d}"
    question_dir.mkdir()

    for variation_index in range(1, variations + 1):
      instance = question.instantiate(rng_seed=seed_rng.randint(0, 2**31 - 1))
      if isinstance(instance, list):
        raise ValueError("Iframe export does not support questions with multiple instances.")

      for answer_index, answer in enumerate(instance.answers, start=1):
        answer.key = f"q{question_index:02d}-v{variation_index:03d}-a{answer_index:02d}"

      payload = {
        "question_html": _clean_html_fragment(
          instance.body.render(ca.OutputFormat.IFRAME_HTML)
        ),
        "answer": [_answer_data(answer) for answer in instance.answers],
        "explanation_html": _clean_html_fragment(
          instance.explanation.render(ca.OutputFormat.IFRAME_HTML)
        ),
      }
      output_path = question_dir / f"v{variation_index:03d}.yaml"
      with output_path.open("w", encoding="utf-8") as handle:
        yaml.dump(
          payload,
          handle,
          Dumper=_IframeYamlDumper,
          allow_unicode=True,
          sort_keys=False,
        )
      written.append(output_path)

  return written
