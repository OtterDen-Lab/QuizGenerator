from __future__ import annotations

import zipfile

from lxml import etree

from QuizGenerator.generation.question import QuestionRegistry
from QuizGenerator.generation.quiz import Quiz
from QuizGenerator.qti_export import export_qti_package


def _quiz():
  QuestionRegistry.load_premade_questions()
  return Quiz.from_exam_dicts([{
    "name": "QTI test",
    "questions": {2: {"Evaluation": {"class": "math130.FunctionEvaluation"}}},
  }])[0]


def test_qti_export_writes_deterministic_pick_one_package(tmp_path):
  first = export_qti_package(_quiz(), variations=2, output_path=tmp_path / "first.zip", base_seed=12)
  second = export_qti_package(_quiz(), variations=2, output_path=tmp_path / "second.zip", base_seed=12)
  with zipfile.ZipFile(first) as archive:
    item_name = next(name for name in archive.namelist() if name.endswith(".xml") and name != "imsmanifest.xml")
    assessment = etree.fromstring(archive.read(item_name))
    assert len(assessment.xpath('//*[local-name()="item"]')) == 2
    first_items = [archive.read(item_name)]
  with zipfile.ZipFile(second) as archive:
    item_name = next(name for name in archive.namelist() if name.endswith(".xml") and name != "imsmanifest.xml")
    second_items = [archive.read(item_name)]
  assert first_items == second_items


def test_qti_export_supports_canvas_dropdown_answers(tmp_path):
  QuestionRegistry.load_premade_questions()
  quiz = Quiz.from_exam_dicts([{
    "name": "Dropdown test",
    "questions": {
      1: {"VSFS": {"class": "cst334.VSFS_states", "num_steps": 3}},
    },
  }])[0]

  package = export_qti_package(
    quiz,
    variations=1,
    output_path=tmp_path / "dropdown.zip",
    base_seed=7,
  )

  with zipfile.ZipFile(package) as archive:
    item_name = next(name for name in archive.namelist() if name.endswith(".xml") and name != "imsmanifest.xml")
    item = etree.fromstring(archive.read(item_name))
  assert item.xpath('//*[local-name()="fieldentry" and text()="multiple_dropdowns_question"]')
  assert item.xpath('//*[local-name()="response_lid"]/@ident') == ["response_answer_1"]
  assert "[answer_1]" in "".join(item.xpath('//*[local-name()="mattext"]//text()'))
