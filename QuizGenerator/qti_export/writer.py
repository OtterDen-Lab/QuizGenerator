"""Canvas Classic Quiz QTI 1.2 package writer.

Canvas imports QTI 2.x choice interactions as generic multiple-answer items.
Its Classic Quiz importer requires Canvas' QTI 1.2 metadata for dropdowns.
"""
from __future__ import annotations

import hashlib
import random
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from lxml import etree

import QuizGenerator.generation.contentast as ca
from QuizGenerator.generation.question import QuestionGroup

from .adapter import ExportedQuestion, QtiCompatibilityError, adapt_instance

QTI = "http://www.imsglobal.org/xsd/ims_qtiasiv1p2"
IMSCP = "http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1"


class QtiExportError(ValueError):
  """A user-facing Canvas QTI export failure."""


def _id(*parts: object) -> str:
  return "g" + hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:31]


def _metadata(parent, label: str, value: object) -> None:
  field = etree.SubElement(parent, "qtimetadatafield")
  etree.SubElement(field, "fieldlabel").text = label
  etree.SubElement(field, "fieldentry").text = str(value)


def _mattext(parent, value: str, *, html: bool = False) -> None:
  element = etree.SubElement(parent, "mattext", **({"texttype": "text/html"} if html else {}))
  element.text = etree.CDATA(value) if html else value


def _accepted_values(answer: ca.Answer) -> list[str]:
  values = [str(payload["answer_text"]) for payload in answer.get_for_canvas() if payload.get("answer_weight", 0) > 0]
  return list(dict.fromkeys(values)) or [str(answer.value)]


class _QtiImageAssets:
  """Store rendered images beside the assessment and return QTI-relative URLs."""

  def __init__(self, assessment_folder: Path):
    self.assessment_folder = assessment_folder
    self.relative_paths: list[str] = []

  @staticmethod
  def _extension(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
      return "png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
      return "jpg"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
      return "gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
      return "webp"
    return "png"

  def add(self, image_data) -> str:
    position = image_data.tell()
    try:
      image_data.seek(0)
      image_bytes = image_data.read()
    finally:
      image_data.seek(position)

    image_hash = hashlib.sha256(image_bytes).hexdigest()
    relative_path = f"assets/{image_hash}.{self._extension(image_bytes)}"
    if relative_path not in self.relative_paths:
      destination = self.assessment_folder / relative_path
      destination.parent.mkdir(exist_ok=True)
      destination.write_bytes(image_bytes)
      self.relative_paths.append(relative_path)
    return relative_path


def _question_type(question: ExportedQuestion) -> str:
  return {
    ca.Answer.CanvasAnswerKind.MULTIPLE_ANSWER: "multiple_choice_question",
    ca.Answer.CanvasAnswerKind.MULTIPLE_DROPDOWN: "multiple_dropdowns_question",
    ca.Answer.CanvasAnswerKind.MATCHING: "matching_question",
  }.get(question.answer_kind, "fill_in_multiple_blanks_question")


def _response(parent, response_id: str, options: list[str], correct: str) -> tuple[str, str]:
  lid = etree.SubElement(parent, "response_lid", ident=response_id)
  material = etree.SubElement(lid, "material")
  _mattext(material, response_id.removeprefix("response_"))
  choices = etree.SubElement(lid, "render_choice")
  correct_id = ""
  for index, option in enumerate(dict.fromkeys(options), 1):
    option_id = f"{response_id}_{index}"
    label = etree.SubElement(choices, "response_label", ident=option_id)
    material = etree.SubElement(label, "material")
    _mattext(material, option)
    if option == correct:
      correct_id = option_id
  return response_id, correct_id


def _item(question: ExportedQuestion, number: int) -> etree._Element:
  item = etree.Element("item", ident=_id("item", number), title=question.title)
  metadata = etree.SubElement(etree.SubElement(item, "itemmetadata"), "qtimetadata")
  _metadata(metadata, "question_type", _question_type(question))
  _metadata(metadata, "points_possible", question.points)
  _metadata(metadata, "assessment_question_identifierref", _id("assessment-question", number))
  presentation = etree.SubElement(item, "presentation")
  body_html = question.body_html
  response_ids: list[tuple[str, str]] = []

  if question.answer_kind == ca.Answer.CanvasAnswerKind.MULTIPLE_DROPDOWN:
    for index, answer in enumerate(question.answers, 1):
      blank = f"answer_{index}"
      body_html = body_html.replace(answer.key, blank)
      response_ids.append(_response(presentation, f"response_{blank}", _accepted_values(answer) + [str(x) for x in (answer.baffles or [])], str(answer.value)))
  else:
    for index, answer in enumerate(question.answers, 1):
      response_id = f"response_answer_{index}"
      if question.answer_kind == ca.Answer.CanvasAnswerKind.MULTIPLE_ANSWER:
        response_ids.append(_response(presentation, response_id, _accepted_values(answer) + [str(x) for x in (answer.baffles or [])], str(answer.value)))
      else:
        # Answer.render_html() already wraps blank keys in square brackets.
        # Replace that complete token so the Canvas placeholder remains
        # [answer_N], rather than becoming [[answer_N]].
        body_html = body_html.replace(f"[{answer.key}]", f"[answer_{index}]")
        # Canvas resolves [answer_N] by finding this response_lid.  Without
        # it, Canvas imports the placeholder as ordinary text.
        accepted_values = _accepted_values(answer)
        response_ids.append(_response(
          presentation,
          response_id,
          accepted_values + [str(x) for x in (answer.baffles or [])],
          accepted_values[0],
        ))
  material = etree.Element("material")
  _mattext(material, body_html, html=True)
  presentation.insert(0, material)

  processing = etree.SubElement(item, "resprocessing")
  outcomes = etree.SubElement(processing, "outcomes")
  etree.SubElement(outcomes, "decvar", varname="SCORE", vartype="Decimal", minvalue="0", maxvalue="100")
  for response_id, correct in response_ids:
    condition = etree.SubElement(processing, "respcondition")
    conditionvar = etree.SubElement(condition, "conditionvar")
    etree.SubElement(conditionvar, "varequal", respident=response_id).text = correct
    etree.SubElement(condition, "setvar", varname="SCORE", action="Add").text = str(100 / len(response_ids))
  if question.explanation_html:
    condition = etree.SubElement(processing, "respcondition", attrib={"continue": "Yes"})
    conditionvar = etree.SubElement(condition, "conditionvar")
    etree.SubElement(conditionvar, "other")
    etree.SubElement(
      condition,
      "displayfeedback",
      feedbacktype="Response",
      linkrefid="general_fb",
    )
    feedback = etree.SubElement(item, "itemfeedback", ident="general_fb")
    material = etree.SubElement(etree.SubElement(feedback, "flow_mat"), "material")
    _mattext(material, question.explanation_html, html=True)
  return item


def export_qti_package(
    quiz,
    *,
    variations: int,
    output_path: str | Path,
    base_seed: int | None = None,
    date_stamp: bool = False,
) -> Path:
  """Write a Canvas Classic Quiz-compatible QTI 1.2 package."""
  if variations < 1:
    raise QtiExportError("qti variations must be at least 1")
  destination = Path(output_path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  seed_rng = random.Random(0 if base_seed is None else base_seed)
  title_suffix = f" ({datetime.now().strftime('%B %Y')})" if date_stamp else ""
  assessment_title = f"{quiz.name}{title_suffix}"
  assessment_id = _id(assessment_title, base_seed, variations)
  with tempfile.TemporaryDirectory(prefix="quizgen-canvas-qti-") as temporary:
    root = Path(temporary)
    folder = root / assessment_id
    folder.mkdir()
    image_assets = _QtiImageAssets(folder)
    try:
      exported: list[ExportedQuestion] = []
      for source in quiz.questions:
        questions = source.questions if isinstance(source, QuestionGroup) else [source]
        for question in questions:
          for variation in range(1, variations + 1):
            instance = question.instantiate(rng_seed=seed_rng.randint(0, 2**31 - 1))
            if isinstance(instance, list):
              raise QtiCompatibilityError(f"{question.name}: multi-instance questions are unsupported")
            exported.append(adapt_instance(
              question,
              instance,
              title=f"{question.name}{title_suffix} variation {variation}",
              image_upload=image_assets.add,
            ))
    except QtiCompatibilityError as exc:
      raise QtiExportError(str(exc)) from exc
    document = etree.Element("questestinterop", nsmap={None: QTI})
    assessment = etree.SubElement(document, "assessment", ident=assessment_id, title=assessment_title)
    qtimetadata = etree.SubElement(assessment, "qtimetadata")
    _metadata(qtimetadata, "cc_maxattempts", "1")
    section = etree.SubElement(assessment, "section", ident="root_section")
    for number, question in enumerate(exported, 1):
      section.append(_item(question, number))
    item_path = folder / f"{assessment_id}.xml"
    item_path.write_bytes(etree.tostring(document, encoding="UTF-8", xml_declaration=True, pretty_print=True))
    manifest = etree.Element("manifest", nsmap={None: IMSCP}, identifier=_id("manifest", assessment_id))
    resources = etree.SubElement(manifest, "resources")
    resource = etree.SubElement(resources, "resource", identifier=assessment_id, type="imsqti_xmlv1p2")
    etree.SubElement(resource, "file", href=f"{assessment_id}/{assessment_id}.xml")
    for relative_path in image_assets.relative_paths:
      etree.SubElement(resource, "file", href=f"{assessment_id}/{relative_path}")
    (root / "imsmanifest.xml").write_bytes(etree.tostring(manifest, encoding="UTF-8", xml_declaration=True, pretty_print=True))
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
      for file in root.rglob("*"):
        if file.is_file():
          archive.write(file, file.relative_to(root).as_posix())
  return destination
