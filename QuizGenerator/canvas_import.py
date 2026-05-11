from __future__ import annotations

import base64
import copy
import html
import logging
import mimetypes
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

log = logging.getLogger(__name__)

_YAML_WIDTH = 88


_FILE_ID_RE = re.compile(r"/files/(\d+)(?:/|$)")
_BLANK_TOKEN_RE = re.compile(r"\[([^\[\]]+)\]")
_INLINE_MATH_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.DOTALL)


@dataclass(frozen=True)
class CanvasQuizRef:
  quiz_id: int
  title: str


def sanitize_filename(name: str) -> str:
  sanitized = re.sub(r"[^\w\-]+", "_", name.strip())
  sanitized = sanitized.strip("_")
  return sanitized or "quiz"


def resolve_canvas_quiz(canvas_course, *, quiz_id: int | None = None, assignment_id: int | None = None):
  course = canvas_course.course

  if quiz_id is not None:
    return course.get_quiz(int(quiz_id))

  if assignment_id is None:
    raise ValueError("Either quiz_id or assignment_id is required.")

  assignment_id = int(assignment_id)
  for quiz in course.get_quizzes():
    if getattr(quiz, "assignment_id", None) == assignment_id:
      return quiz

  raise ValueError(
    f"Could not find a classic quiz for assignment_id={assignment_id}. "
    "Make sure the assignment belongs to a classic quiz."
  )


def export_canvas_quiz_to_yaml(
  canvas_course,
  *,
  quiz_id: int | None = None,
  assignment_id: int | None = None,
  output_path: str | os.PathLike[str],
  flatten_groups: bool = True,
) -> Path:
  quiz = resolve_canvas_quiz(canvas_course, quiz_id=quiz_id, assignment_id=assignment_id)
  output_path = Path(output_path).expanduser().resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)

  image_dir = output_path.parent / f"{output_path.stem}_assets"
  image_dir.mkdir(parents=True, exist_ok=True)

  spec = build_canvas_quiz_spec(
    canvas_course,
    quiz,
    image_dir=image_dir,
    flatten_groups=flatten_groups,
  )
  yaml_text = yaml.safe_dump(
    spec,
    sort_keys=False,
    allow_unicode=True,
    width=_YAML_WIDTH,
  )
  output_path.write_text(yaml_text, encoding="utf-8")
  return output_path


def build_canvas_quiz_spec(
  canvas_course,
  quiz,
  *,
  image_dir: Path,
  flatten_groups: bool = True,
) -> dict[str, Any]:
  quiz_id = int(getattr(quiz, "id"))
  quiz_title = str(getattr(quiz, "title", f"Canvas Quiz {quiz_id}"))
  practice = str(getattr(quiz, "quiz_type", "")).lower() == "practice_quiz"
  description = _plain_text_from_html(getattr(quiz, "description", None))

  questions = list(quiz.get_questions())
  question_specs: list[dict[str, Any]] = []
  seen_groups: set[int] = set()

  for question in questions:
    group_id = getattr(question, "quiz_group_id", None)
    if flatten_groups and group_id is not None:
      group_id = int(group_id)
      if group_id in seen_groups:
        continue
      seen_groups.add(group_id)

    question_specs.append(
      _serialize_question(canvas_course, quiz, question, image_dir=image_dir)
    )

  spec: dict[str, Any] = {
    "name": quiz_title,
    "practice": practice,
    "yaml_id": f"canvas-{quiz_id}",
    "question_order": "yaml",
    "questions": question_specs,
  }

  if description:
    spec["description"] = description

  return spec


def _serialize_question(canvas_course, quiz, question, *, image_dir: Path) -> dict[str, Any]:
  question_type = str(getattr(question, "question_type", "")).lower()
  question_name = str(getattr(question, "question_name", f"Question {getattr(question, 'id', '?')}"))
  points = float(getattr(question, "points_possible", 0) or 0)

  group_id = getattr(question, "quiz_group_id", None)
  if group_id is not None:
    try:
      group = quiz.get_quiz_group(int(group_id))
      if getattr(group, "name", None):
        question_name = str(group.name)
      if getattr(group, "question_points", None) is not None:
        points = float(group.question_points)
    except Exception as exc:
      log.warning(f"Failed to resolve quiz group {group_id}: {exc}")

  blank_answer_specs = _build_blank_answer_specs(question)

  body_nodes = _html_to_nodes(
    getattr(question, "question_text", "") or "",
    canvas_course=canvas_course,
    quiz=quiz,
    image_dir=image_dir,
    blank_answer_specs=blank_answer_specs,
  )

  if question_type not in {"fill_in_multiple_blanks_question", "multiple_dropdowns_question"}:
    answer_nodes, display_nodes = _build_answer_nodes(
      canvas_course,
      quiz,
      question,
    )

    if display_nodes:
      body_nodes.append({"section": {"children": display_nodes}})

    if answer_nodes:
      body_nodes.append({"only_html": {"children": answer_nodes}})

  explanation_nodes = _combine_comments(question)

  yaml_spec: dict[str, Any] = {
    "version": 1,
    "body": body_nodes,
    "explanation": explanation_nodes,
  }

  if question_type in {"matching_question"}:
    matching_spec = _build_matching_spec(question)
    if matching_spec is not None:
      yaml_spec["body"].append({"matching": matching_spec})

  serialized = {
    "name": question_name,
    "points": points,
    "class": "FromYaml",
    "kwargs": {
      "yaml_spec": yaml_spec,
    },
  }

  source_id = getattr(question, "id", None)
  if source_id is not None:
    serialized["question_id"] = f"canvas:{quiz.id}:{source_id}"

  return serialized


def _build_blank_answer_specs(question) -> dict[str, dict[str, Any]]:
  question_type = str(getattr(question, "question_type", "")).lower()
  if question_type not in {"fill_in_multiple_blanks_question", "multiple_dropdowns_question"}:
    return {}

  answers = list(getattr(question, "answers", []) or [])
  grouped: dict[str, list[tuple[str, bool]]] = defaultdict(list)
  for answer in answers:
    blank_id = str(answer.get("blank_id", "blank"))
    grouped[blank_id].append((_answer_text(answer), int(answer.get("answer_weight", 0) or 0) > 0))

  blank_specs: dict[str, dict[str, Any]] = {}
  for blank_id, rows in grouped.items():
    correct_values = [text for text, is_correct in rows if is_correct]
    incorrect_values = [text for text, is_correct in rows if not is_correct]
    if not correct_values:
      correct_values = [rows[0][0]]

    parsed_value = _coerce_answer_values(correct_values)
    node_type = _infer_scalar_answer_type(parsed_value)
    blank_length = max(
      5,
      max((len(str(value)) for value in [*correct_values, *incorrect_values] if value), default=0),
    )
    blank_length = min(blank_length, 16)
    answer_spec: dict[str, Any] = {
      "type": node_type,
      "value": parsed_value,
      "label": "",
      "key": blank_id,
      "correct": True,
      "blank_length": blank_length,
    }
    if incorrect_values:
      answer_spec["baffles"] = incorrect_values

    blank_specs[blank_id] = {"answer": answer_spec}

  return blank_specs


def _build_answer_nodes(canvas_course, quiz, question) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  question_type = str(getattr(question, "question_type", "")).lower()
  answers = list(getattr(question, "answers", []) or [])

  if not answers:
    return [], []

  if question_type in {"multiple_choice_question", "multiple_answers_question"}:
    display_nodes = _choice_display_nodes(answers)
    answer_nodes = []
    for idx, answer in enumerate(answers):
      text = _answer_text(answer)
      weight = int(answer.get("answer_weight", 0) or 0)
      answer_nodes.append(
        {
          "answer": {
            "type": "multiple_choice",
            "value": text,
            "correct": weight > 0,
            "key": f"{getattr(question, 'id', 'q')}:{idx}",
          }
        }
      )
    return answer_nodes, display_nodes

  if question_type in {"fill_in_multiple_blanks_question", "multiple_dropdowns_question"}:
    grouped: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for answer in answers:
      blank_id = str(answer.get("blank_id", "blank"))
      grouped[blank_id].append((_answer_text(answer), int(answer.get("answer_weight", 0) or 0) > 0))

    answer_nodes = []
    for blank_id, rows in grouped.items():
      correct_values = [text for text, is_correct in rows if is_correct]
      incorrect_values = [text for text, is_correct in rows if not is_correct]
      if not correct_values:
        correct_values = [rows[0][0]]
      parsed_value = _coerce_answer_values(correct_values)
      node_type = _infer_scalar_answer_type(parsed_value)
      node: dict[str, Any] = {
        "answer": {
          "type": node_type,
          "value": parsed_value,
          "label": blank_id,
          "key": blank_id,
          "correct": True,
        }
      }
      if incorrect_values:
        node["answer"]["baffles"] = incorrect_values
      answer_nodes.append(node)

    return answer_nodes, []

  if question_type in {"matching_question"}:
    return [], []

  if question_type in {"numerical_question", "calculated_question"}:
    correct_answers = [answer for answer in answers if int(answer.get("answer_weight", 0) or 0) > 0]
    if not correct_answers:
      return [], []
    answer_text = _answer_text(correct_answers[0])
    node_type, value = _numeric_answer_type(answer_text)
    answer_node = {
      "answer": {
        "type": node_type,
        "value": value,
        "label": "Answer",
        "correct": True,
      }
    }
    if question_type == "calculated_question":
      answer_node["answer"]["pdf_only"] = True
    return [answer_node], []

  if question_type in {"short_answer_question", "essay_question"}:
    return [], []

  # Fallback: preserve the choices as visible content but do not invent an answer form.
  display_nodes = _choice_display_nodes(answers)
  return [], display_nodes


def _build_matching_spec(question) -> dict[str, Any] | None:
  answers = list(getattr(question, "answers", []) or [])
  pairs: list[list[str]] = []
  distractors: list[str] = []

  for answer in answers:
    left = answer.get("answer_match_left")
    right = answer.get("answer_match_right")
    if left is not None and right is not None:
      pairs.append([_answer_text({"answer_text": left}), _answer_text({"answer_text": right})])
      continue
    incorrect = answer.get("matching_answer_incorrect_matches")
    if incorrect:
      if isinstance(incorrect, str):
        distractors.extend([item.strip() for item in incorrect.splitlines() if item.strip()])
      else:
        distractors.extend([str(item) for item in incorrect])

  if not pairs:
    return None

  spec: dict[str, Any] = {"pairs": pairs}
  if distractors:
    spec["distractors"] = distractors
  return spec


def _choice_display_nodes(answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
  if not answers:
    return []

  lines = []
  for idx, answer in enumerate(answers):
    letter = chr(ord("A") + idx)
    lines.append(f"{letter}. {_answer_text(answer)}")
  return [{"paragraph": {"lines": [line]}} for line in lines]


def _combine_comments(question) -> list[dict[str, Any]]:
  comments = []
  for label, attr in (
    ("Correct feedback", "correct_comments_html"),
    ("Incorrect feedback", "incorrect_comments_html"),
    ("General feedback", "neutral_comments_html"),
  ):
    value = getattr(question, attr, None)
    if not value:
      continue
    comments.append({"paragraph": {"lines": [f"{label}: {_plain_text_from_html(value)}"]}})
  return comments


class _CanvasHtmlParser(HTMLParser):
  BLOCK_TAGS = {
    "p", "div", "section", "article", "blockquote", "header", "footer",
    "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol",
  }

  def __init__(self, *, canvas_course, quiz, image_dir: Path, blank_answer_specs: dict[str, dict[str, Any]] | None = None):
    super().__init__(convert_charrefs=True)
    self.canvas_course = canvas_course
    self.quiz = quiz
    self.image_dir = image_dir
    self.blank_answer_specs = blank_answer_specs or {}
    self.blank_pattern = self._build_blank_pattern(self.blank_answer_specs)
    self.nodes: list[dict[str, Any]] = []
    self.text_parts: list[Any] = []
    self.pre_parts: list[str] = []
    self.in_pre = False
    self.skip_depth = 0
    self.list_stack: list[str] = []
    self.table_stack: list[_TableContext] = []

  def flush_text(self) -> None:
    if self.in_pre:
      return
    text_parts = self.text_parts
    self.text_parts = []
    if not text_parts:
      return
    node = _parts_to_node(text_parts)
    if node is not None:
      self._append_node(node)

  def _append_node(self, node: dict[str, Any]) -> None:
    if self.table_stack and self.table_stack[-1].current_cell is not None:
      self.table_stack[-1].current_cell.append(node)
      return
    self.nodes.append(node)

  def _current_buffer(self) -> list[Any]:
    if self.table_stack and self.table_stack[-1].current_cell is not None:
      return self.table_stack[-1].current_cell
    return self.text_parts

  @staticmethod
  def _build_blank_pattern(blank_answer_specs: dict[str, dict[str, Any]]) -> re.Pattern | None:
    if not blank_answer_specs:
      return None
    tokens = sorted((re.escape(f"[{blank_id}]") for blank_id in blank_answer_specs), key=len, reverse=True)
    return re.compile("(" + "|".join(tokens) + ")")

  def _append_text(self, data: str) -> None:
    if not data:
      return
    if self.in_pre:
      self.pre_parts.append(data)
      return

    if self.blank_pattern is None:
      self._current_buffer().append(data)
      return

    cursor = 0
    buffer = self._current_buffer()
    for match in self.blank_pattern.finditer(data):
      start, end = match.span()
      if start > cursor:
        buffer.append(data[cursor:start])
      token = match.group(0)
      blank_id = token[1:-1]
      spec = self.blank_answer_specs.get(blank_id)
      if spec is not None:
        buffer.append(copy.deepcopy(spec))
      else:
        buffer.append(token)
      cursor = end
    if cursor < len(data):
      buffer.append(data[cursor:])

  def _flush_table_cell(self) -> None:
    if not self.table_stack:
      return
    table = self.table_stack[-1]
    if table.current_cell is None:
      return
    cell_parts = table.current_cell
    table.current_cell = None
    cell_node = _parts_to_node(cell_parts)
    if cell_node is None:
      cell_node = ""
    table.current_row.append(cell_node)

  def _flush_table_row(self) -> None:
    if not self.table_stack:
      return
    table = self.table_stack[-1]
    if table.current_row is None:
      return
    if table.current_row:
      if table.current_row_is_header and table.headers is None:
        table.headers = list(table.current_row)
      else:
        table.rows.append(list(table.current_row))
    table.current_row = None
    table.current_row_is_header = False

  def _flush_table(self) -> None:
    if not self.table_stack:
      return
    self._flush_table_cell()
    self._flush_table_row()
    table = self.table_stack.pop()
    if table.rows or table.headers is not None:
      table_spec: dict[str, Any] = {"rows": table.rows}
      if table.headers is not None:
        table_spec["headers"] = table.headers
      if table.alignments:
        table_spec["alignments"] = table.alignments
      if table.padding:
        table_spec["padding"] = table.padding
      if table.hide_rules:
        table_spec["hide_rules"] = table.hide_rules
      self.nodes.append({"table": table_spec})

  def _start_table(self) -> None:
    self.flush_text()
    self.table_stack.append(_TableContext())

  def handle_starttag(self, tag, attrs):
    tag = tag.lower()
    if tag in {"script", "style"}:
      self.skip_depth += 1
      return
    if self.skip_depth:
      return

    attrs_dict = dict(attrs)

    if tag == "table":
      self._start_table()
      return

    if self.table_stack:
      table = self.table_stack[-1]
      if tag == "tr":
        self._flush_table_cell()
        self._flush_table_row()
        table.current_row = []
        return
      if tag in {"td", "th"}:
        table.current_cell = []
        if table.current_row is None:
          table.current_row = []
        if tag == "th":
          table.current_row_is_header = True
        return

    if tag == "br":
      self._current_buffer().append("\n")
    elif tag in {"strong", "b"}:
      self._current_buffer().append("**")
    elif tag in {"em", "i"}:
      self._current_buffer().append("*")
    elif tag == "code" and not self.in_pre:
      self._current_buffer().append("`")
    elif tag == "pre":
      self.flush_text()
      self.in_pre = True
      self.pre_parts = []
    elif tag == "li":
      self.flush_text()
      self._current_buffer().append("- ")
    elif tag in {"ul", "ol"}:
      self.list_stack.append(tag)
    elif tag == "img":
      self.flush_text()
      src = attrs_dict.get("src")
      if not src:
        return
      path = _download_image(self.canvas_course, src, self.image_dir)
      if path is None:
        return
      picture_spec = {"path": str(path)}
      caption = attrs_dict.get("alt") or attrs_dict.get("title")
      if caption:
        picture_spec["caption"] = caption
      width = attrs_dict.get("width")
      if width:
        picture_spec["width"] = width
      self._append_node({"picture": picture_spec})
    elif tag in self.BLOCK_TAGS:
      self.flush_text()

  def handle_endtag(self, tag):
    tag = tag.lower()
    if tag in {"script", "style"}:
      self.skip_depth = max(0, self.skip_depth - 1)
      return
    if self.skip_depth:
      return

    if tag == "table":
      self._flush_table()
      return

    if self.table_stack:
      if tag in {"td", "th"}:
        self._flush_table_cell()
        return
      if tag == "tr":
        self._flush_table_cell()
        self._flush_table_row()
        return

    if tag in self.BLOCK_TAGS:
      self.flush_text()
    if tag in {"strong", "b"}:
      self._current_buffer().append("**")
    elif tag in {"em", "i"}:
      self._current_buffer().append("*")
    elif tag == "code" and not self.in_pre:
      self._current_buffer().append("`")
    elif tag == "pre":
      self.in_pre = False
      text = _normalize_text("".join(self.pre_parts))
      self.pre_parts = []
      if text:
        self._append_node({"code": {"content": text}})
    elif tag in {"ul", "ol"}:
      if self.list_stack:
        self.list_stack.pop()

  def handle_data(self, data):
    if self.skip_depth:
      return
    if self.in_pre:
      self.pre_parts.append(data)
    else:
      self._append_text(data)

  def close(self):
    super().close()
    self.flush_text()
    while self.table_stack:
      self._flush_table()
    if self.in_pre and self.pre_parts:
      text = _normalize_text("".join(self.pre_parts))
      if text:
        self.nodes.append({"code": {"content": text}})
      self.pre_parts = []
      self.in_pre = False


@dataclass
class _TableContext:
  headers: list[Any] | None = None
  rows: list[list[Any]] = None  # type: ignore[assignment]
  current_row: list[Any] | None = None
  current_row_is_header: bool = False
  current_cell: list[Any] | None = None
  alignments: list[str] | None = None
  padding: bool = False
  hide_rules: bool = False

  def __post_init__(self):
    if self.rows is None:
      self.rows = []


def _parts_to_node(parts: list[Any]) -> dict[str, Any] | str | None:
  flattened: list[Any] = []
  for part in parts:
    if part is None:
      continue
    if isinstance(part, str) and part == "":
      continue
    if isinstance(part, str):
      flattened.extend(_inline_math_to_parts(part))
    else:
      flattened.append(part)

  if not flattened:
    return None

  if len(flattened) == 1:
    single = flattened[0]
    if isinstance(single, str):
      math_node = _maybe_equation_node(single)
      if math_node is not None:
        return math_node
    return single

  return {"paragraph": {"lines": flattened}}


def _inline_math_to_parts(text: str) -> list[Any]:
  if not text:
    return []

  parts: list[Any] = []
  cursor = 0
  for match in _INLINE_MATH_RE.finditer(text):
    start, end = match.span()
    if start > cursor:
      parts.append(text[cursor:start])
    latex = next((group for group in match.groups() if group is not None), "")
    latex = _strip_math_wrappers(latex.strip())
    inline_expr = match.group(2) is not None or match.group(3) is not None
    parts.append(
      {
        "equation": {
          "latex": latex.strip(),
          "inline": inline_expr,
        }
      }
    )
    cursor = end
  if cursor < len(text):
    parts.append(text[cursor:])
  return parts


def _maybe_equation_node(text: str) -> dict[str, Any] | None:
  stripped = _normalize_text(text)
  if not stripped:
    return None

  stripped = _strip_math_wrappers(stripped)

  if not stripped:
    return None

  if _looks_like_latex_math(stripped):
    return {"equation": {"latex": stripped, "inline": False}}
  return None


def _looks_like_latex_math(text: str) -> bool:
  if any(token in text for token in (r"\left", r"\right", r"\frac", r"\begin{", r"\end{", r"\sum", r"\nabla", r"\cdot", r"\times", r"\int", r"\sqrt")):
    return True
  if re.search(r"\\[A-Za-z]+", text):
    return True
  if re.search(r"[A-Za-z_]\s*\^\s*[{(]?", text):
    return True
  if re.search(r"[=+\-*/]", text) and re.search(r"[A-Za-z0-9]", text):
    return True
  return False


def _strip_math_wrappers(text: str) -> str:
  stripped = text.strip()
  if not stripped:
    return ""

  dollar_match = re.match(r"^\$+\s*(.*?)\s*\$+$", stripped, flags=re.DOTALL)
  if dollar_match:
    stripped = dollar_match.group(1).strip()

  if stripped.startswith(r"\(") and stripped.endswith(r"\)"):
    stripped = stripped[2:-2].strip()
  elif stripped.startswith(r"\[") and stripped.endswith(r"\]"):
    stripped = stripped[2:-2].strip()

  if stripped.startswith(r"\displaystyle "):
    stripped = stripped[len(r"\displaystyle "):].strip()
  elif stripped == r"\displaystyle":
    stripped = ""

  return stripped


def _html_to_nodes(
  html_text: str,
  *,
  canvas_course,
  quiz,
  image_dir: Path,
  blank_answer_specs: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
  if not html_text:
    return []
  parser = _CanvasHtmlParser(
    canvas_course=canvas_course,
    quiz=quiz,
    image_dir=image_dir,
    blank_answer_specs=blank_answer_specs,
  )
  parser.feed(str(html_text))
  parser.close()
  return parser.nodes


def _plain_text_from_html(html_text: str | None) -> str:
  if not html_text:
    return ""
  text = re.sub(r"<\s*br\s*/?\s*>", "\n", str(html_text), flags=re.IGNORECASE)
  text = re.sub(r"<[^>]+>", " ", text)
  return _normalize_text(html.unescape(text))


def _normalize_text(text: str) -> str:
  lines = [line.strip() for line in text.splitlines()]
  cleaned = "\n".join(line for line in lines if line)
  return re.sub(r"[ \t]+", " ", cleaned).strip()


def _answer_text(answer: dict[str, Any]) -> str:
  for key in ("answer_text", "text", "answer", "value"):
    value = answer.get(key)
    if value is not None:
      return str(value)
  return ""


def _coerce_answer_values(values: list[str]) -> str | list[str]:
  if len(values) == 1:
    return values[0]
  return values


def _infer_scalar_answer_type(value: str | list[str]) -> str:
  if isinstance(value, list):
    return "string"
  if re.fullmatch(r"[+-]?\d+", value or ""):
    return "int"
  if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?", value or ""):
    return "float"
  return "string"


def _numeric_answer_type(answer_text: str) -> tuple[str, int | float | str]:
  text = answer_text.strip()
  if re.fullmatch(r"[+-]?\d+", text):
    return "int", int(text)
  try:
    return "float", float(text)
  except ValueError:
    return "string", text


def _download_image(canvas_course, src: str, image_dir: Path) -> Path | None:
  image_dir.mkdir(parents=True, exist_ok=True)
  src = str(src).strip()

  if src.startswith("data:"):
    match = re.match(r"data:(?P<mime>[^;]+);base64,(?P<data>.+)", src, re.DOTALL)
    if not match:
      return None
    mime = match.group("mime")
    data = base64.b64decode(match.group("data"))
    ext = mimetypes.guess_extension(mime) or ".bin"
    path = image_dir / f"image-{abs(hash(src)) & 0xffffffff:08x}{ext}"
    path.write_bytes(data)
    return path.resolve()

  parsed = urlparse(src)
  file_id = None
  if parsed.path:
    match = _FILE_ID_RE.search(parsed.path)
    if match:
      file_id = int(match.group(1))

  if file_id is not None:
    try:
      canvas_file = canvas_course.course.get_file(file_id)
      filename = getattr(canvas_file, "filename", None) or getattr(canvas_file, "display_name", None) or f"file-{file_id}"
      path = _build_image_path(image_dir, str(filename), content_type=getattr(canvas_file, "content_type", None))
      canvas_file.download(str(path))
      return path.resolve()
    except Exception as exc:
      log.warning(f"Failed to download Canvas file {file_id}: {exc}")
      return None

  if parsed.scheme in {"http", "https"}:
    try:
      response = canvas_course.course._requester.request("GET", _url=src)
      filename = os.path.basename(parsed.path) or f"image-{abs(hash(src)) & 0xffffffff:08x}"
      path = _build_image_path(
        image_dir,
        filename,
        content_type=getattr(response, "headers", {}).get("Content-Type"),
      )
      path.write_bytes(response.content)
      return path.resolve()
    except Exception as exc:
      log.warning(f"Failed to download image {src}: {exc}")
      return None

  return None


def _build_image_path(image_dir: Path, filename: str, *, content_type: str | None = None) -> Path:
  raw_name = Path(str(filename))
  stem = sanitize_filename(raw_name.stem)
  suffix = raw_name.suffix
  if not suffix:
    suffix = _guess_image_extension(content_type)
  return image_dir / f"{stem}{suffix}"


def _guess_image_extension(content_type: str | None) -> str:
  if content_type:
    mime = content_type.split(";", 1)[0].strip().lower()
    if mime:
      ext = mimetypes.guess_extension(mime)
      if ext:
        return ext
      if mime == "image/jpeg":
        return ".jpg"
      if mime == "image/svg+xml":
        return ".svg"
      if mime == "image/webp":
        return ".webp"
  return ".png"
