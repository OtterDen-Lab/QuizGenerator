from __future__ import annotations

import re
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


def test_qti_export_uses_single_bracket_blank_tokens(tmp_path):
  package = export_qti_package(
    _quiz(),
    variations=1,
    output_path=tmp_path / "blank.zip",
    base_seed=12,
  )

  with zipfile.ZipFile(package) as archive:
    item_name = next(name for name in archive.namelist() if name.endswith(".xml") and name != "imsmanifest.xml")
    item = etree.fromstring(archive.read(item_name))

  body = "".join(item.xpath('//*[local-name()="presentation"]/*[local-name()="material"]//*[local-name()="mattext"]//text()'))
  assert "[answer_1]" in body
  assert "[[answer_1]]" not in body
  assert item.xpath('//*[local-name()="fieldentry" and text()="fill_in_multiple_blanks_question"]')
  assert item.xpath('//*[local-name()="response_lid"]/@ident') == ["response_answer_1"]
  assert item.xpath('//*[local-name()="response_label"]/@ident') == ["response_answer_1_1"]
  assert item.xpath('//*[local-name()="varequal"]/@respident') == ["response_answer_1"]
  assert item.xpath('//*[local-name()="varequal"]/text()') == ["response_answer_1_1"]
  feedback = item.xpath('//*[local-name()="itemfeedback" and @ident="general_fb"]//*[local-name()="mattext"]//text()')
  assert "So the value of the function" in "".join(feedback)


def test_qti_export_links_every_mlfq_blank_to_a_scored_response(tmp_path):
  QuestionRegistry.load_premade_questions()
  quiz = Quiz.from_exam_dicts([{
    "name": "MLFQ QTI test",
    "questions": {
      1: {"MLFQ": {"class": "cst334.MLFQQuestion", "num_jobs": 3, "num_queues": 2}},
    },
  }])[0]

  package = export_qti_package(
    quiz,
    variations=1,
    output_path=tmp_path / "mlfq.zip",
    base_seed=12,
  )

  with zipfile.ZipFile(package) as archive:
    item_name = next(name for name in archive.namelist() if name.endswith(".xml") and name != "imsmanifest.xml")
    item = etree.fromstring(archive.read(item_name))
    manifest = etree.fromstring(archive.read("imsmanifest.xml"))
    archive_names = set(archive.namelist())

  body = "".join(item.xpath('//*[local-name()="presentation"]/*[local-name()="material"]//*[local-name()="mattext"]//text()'))
  blank_ids = re.findall(r"\[([^\[\]]+)\]", body)
  response_ids = item.xpath('//*[local-name()="response_lid"]/@ident')
  scored_responses = item.xpath('//*[local-name()="varequal"]/@respident')
  scored_choice_ids = item.xpath('//*[local-name()="varequal"]/text()')

  assert len(blank_ids) == 3
  assert response_ids == [f"response_{blank_id}" for blank_id in blank_ids]
  assert scored_responses == response_ids
  assert all(scored_choice_ids)

  explanation_html = "".join(
    item.xpath('//*[local-name()="itemfeedback" and @ident="general_fb"]//*[local-name()="mattext"]//text()')
  )
  image_sources = re.findall(r'<img src="([^"]+)"', explanation_html)
  assert image_sources
  assessment_folder = item_name.rsplit("/", 1)[0]
  manifest_files = manifest.xpath('//*[local-name()="file"]/@href')
  for source in image_sources:
    archive_path = f"{assessment_folder}/{source}"
    assert archive_path in archive_names
    assert archive_path in manifest_files
