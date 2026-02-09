from __future__ import annotations

import base64
import io
import math
import os
import random
import re
from typing import Any

import numpy as np
import yaml

import QuizGenerator.contentast as ca
from QuizGenerator.question import QuestionContext


_TEMPLATE_RE = re.compile(r"\{\{(.*?)\}\}")
_EXPR_ONLY_RE = re.compile(r"^\s*\{\{(.*?)\}\}\s*$")
_SPEC_CACHE: dict[str, dict[str, Any]] = {}


def load_question_spec(
  *,
  yaml_path: str | None = None,
  yaml_text: str | None = None,
  yaml_spec: dict[str, Any] | None = None,
  base_dir: str | None = None,
) -> dict[str, Any]:
  if yaml_spec is not None:
    return yaml_spec

  if yaml_path is not None:
    resolved = _resolve_path(yaml_path, base_dir=base_dir)
    cached = _SPEC_CACHE.get(resolved)
    if cached is not None:
      return cached
    with open(resolved, "r") as fid:
      spec = yaml.safe_load(fid)
    _SPEC_CACHE[resolved] = spec
    return spec

  if yaml_text is not None:
    return yaml.safe_load(yaml_text)

  raise ValueError("Must provide yaml_path, yaml_text, or yaml_spec for a YAML question.")


def _resolve_path(path: str, base_dir: str | None = None) -> str:
  if os.path.isabs(path):
    return path
  if base_dir:
    return os.path.abspath(os.path.join(base_dir, path))
  return os.path.abspath(path)


def apply_context_spec(spec: dict[str, Any], context: QuestionContext | dict[str, Any]) -> None:
  ctx_spec = spec.get("context", {}) or {}
  vars_spec = ctx_spec.get("vars", {}) or {}
  derived_spec = ctx_spec.get("derived", {}) or {}

  for name, var_def in vars_spec.items():
    context[name] = _evaluate_var_def(var_def, context)

  if isinstance(derived_spec, list):
    for entry in derived_spec:
      _apply_derived_entry(entry, context)
    return

  for name, expr in derived_spec.items():
    if expr is None:
      continue
    if isinstance(expr, dict) and "python" in expr:
      result = _exec_python(expr.get("python", ""), context)
      context[name] = result
    else:
      context[name] = _eval_expr(str(expr), context)


def parse_question_templates(spec: dict[str, Any]) -> dict[str, ca.Element]:
  body_nodes = spec.get("body", []) or []
  explanation_nodes = spec.get("explanation", []) or []
  body = _parse_section(body_nodes)
  explanation = _parse_section(explanation_nodes)
  return {"body": body, "explanation": explanation}


def _parse_section(nodes: Any) -> ca.Section:
  if isinstance(nodes, dict):
    nodes = nodes.get("children", nodes.get("nodes", []))
  elements = []
  for node in _normalize_nodes(nodes):
    element = _parse_node(node)
    if element is None:
      continue
    elements.append(element)
  return ca.Section(elements)


def _normalize_nodes(nodes: Any) -> list[Any]:
  if nodes is None:
    return []
  if isinstance(nodes, list):
    return nodes
  return [nodes]


def _parse_node(node) -> ca.Element | ca.TemplateElement | None:
  if node is None:
    return None
  if isinstance(node, str):
    return _parse_text(node)
  if isinstance(node, (int, float, bool)):
    return ca.Text(str(node))
  if isinstance(node, list):
    return _parse_section(node)
  if not isinstance(node, dict):
    return ca.Text(str(node))

  if "equation" in node and "inline" in node and len(node.keys()) <= 2:
    return _parse_equation({"latex": node["equation"], "inline": node.get("inline", False)}, inline=False)

  if len(node.keys()) == 1:
    key, value = next(iter(node.items()))
    parser = ca.get_yaml_node_parser(key)
    if parser is not None:
      return parser(value)

  raise ValueError(f"Unknown YAML AST node: {node}")


def _parse_paragraph(lines: Any) -> ca.Paragraph:
  if isinstance(lines, dict):
    lines = lines.get("lines", lines.get("value", []))
  if isinstance(lines, (str, int, float, bool)):
    lines = [lines]
  elements = []
  for item in lines or []:
    elements.append(_parse_inline(item))
  return ca.Paragraph(elements)


def _parse_inline(item: Any) -> ca.Element | ca.TemplateElement | None:
  if isinstance(item, str):
    return _parse_text(item)
  if isinstance(item, (int, float, bool)):
    return ca.Text(str(item))
  if isinstance(item, dict):
    if "equation" in item:
      return _parse_equation(item["equation"], inline=item.get("inline", True))
    if "text" in item:
      return _parse_text(item["text"])
  return _parse_node(item)


def _parse_text(text_value: Any) -> ca.Element:
  if text_value is None:
    return ca.Text("")
  if isinstance(text_value, dict):
    text_value = text_value.get("value", "")
  if not isinstance(text_value, str):
    return ca.Text(str(text_value))
  if "{{" in text_value:
    return ca.Expr(lambda ctx, t=text_value: _render_template(t, ctx))
  return ca.Text(text_value)


def _parse_equation(spec: Any, *, inline: bool) -> ca.Element:
  if isinstance(spec, dict):
    latex = spec.get("latex", "")
    inline = spec.get("inline", inline)
  else:
    latex = spec

  if not isinstance(latex, str):
    latex = str(latex)

  if "{{" in latex:
    return ca.Expr(lambda ctx, s=latex, inline=inline: ca.Equation(_render_template(s, ctx), inline=inline))
  return ca.Equation(latex, inline=inline)


def _parse_code(spec: Any) -> ca.Element:
  if isinstance(spec, dict):
    content = spec.get("content", spec.get("value", ""))
  else:
    content = spec

  if not isinstance(content, str):
    content = str(content)

  if "{{" in content:
    return ca.Expr(lambda ctx, s=content: ca.Code(_render_template(s, ctx)))
  return ca.Code(content)


def _parse_table(spec: dict[str, Any]) -> ca.Table:
  headers = spec.get("headers")
  rows = spec.get("rows") or spec.get("data") or []
  alignments = spec.get("alignments")
  padding = spec.get("padding", False)
  hide_rules = spec.get("hide_rules", False)

  parsed_headers = None
  if headers is not None:
    parsed_headers = [_parse_inline(h) for h in headers]

  parsed_rows = []
  for row in rows:
    parsed_row = [_parse_inline(cell) for cell in row]
    parsed_rows.append(parsed_row)

  return ca.Table(
    headers=parsed_headers,
    data=parsed_rows,
    alignments=alignments,
    padding=padding,
    hide_rules=hide_rules
  )


def _parse_answer_block(spec: Any) -> ca.AnswerBlock:
  if isinstance(spec, dict):
    spec = spec.get("answers", [])
  if not isinstance(spec, list):
    raise ValueError("answer_block must be a list of answer nodes.")
  answers = []
  for item in spec:
    element = _parse_node(item)
    if element is None:
      continue
    answers.append(element)
  return ca.AnswerBlock(answers)


def _parse_when(spec: dict[str, Any]) -> ca.When:
  condition = spec.get("cond") if isinstance(spec, dict) else None
  if condition is None:
    condition = spec.get("condition")
  then_branch = spec.get("then") if isinstance(spec, dict) else None
  else_branch = spec.get("else") if isinstance(spec, dict) else None

  if isinstance(condition, str):
    cond_fn = lambda ctx, expr=condition: bool(_eval_expr(expr, ctx))
  else:
    cond_fn = bool(condition)

  then_node = _parse_branch(then_branch)
  else_node = _parse_branch(else_branch)
  return ca.When(cond_fn, then_node, else_node)


def _parse_branch(branch: Any) -> ca.Element | ca.TemplateElement | None:
  if branch is None:
    return None
  if isinstance(branch, list):
    return _parse_section(branch)
  return _parse_node(branch)


def _parse_matrix(spec: Any) -> ca.TemplateElement:
  if isinstance(spec, dict):
    data_spec = spec.get("data", [])
    bracket_type = spec.get("bracket_type", spec.get("bracket", "p"))
    inline = bool(spec.get("inline", False))
    name = spec.get("name")
  else:
    data_spec = spec
    bracket_type = "p"
    inline = False
    name = None

  def builder(ctx):
    data = _normalize_matrix_data(data_spec)
    evaluated = []
    for row in data:
      evaluated.append([_eval_field(cell, ctx) for cell in row])
    name_value = _eval_field(name, ctx) if name is not None else None
    return ca.Matrix(evaluated, bracket_type=bracket_type, inline=inline, name=name_value)

  return ca.Expr(builder)


def _normalize_matrix_data(data_spec: Any) -> list[list[Any]]:
  if not isinstance(data_spec, list):
    raise ValueError("matrix data must be a list.")
  if not data_spec:
    return []
  if not isinstance(data_spec[0], list):
    return [[item] for item in data_spec]
  return data_spec


def _parse_math_expression(spec: Any) -> ca.TemplateElement:
  if isinstance(spec, dict):
    parts_spec = spec.get("parts", []) or []
    inline = bool(spec.get("inline", False))
  else:
    parts_spec = spec or []
    inline = False

  parsed_parts = []
  for part in parts_spec:
    if isinstance(part, (str, int, float, bool)):
      parsed_parts.append(part)
    else:
      parsed_parts.append(_parse_inline(part))

  def builder(ctx):
    resolved_parts = []
    for part in parsed_parts:
      if isinstance(part, (str, int, float, bool)):
        if isinstance(part, str):
          resolved_parts.append(_eval_field(part, ctx))
        else:
          resolved_parts.append(part)
        continue
      if part is None:
        continue
      resolved = ca.resolve_template(part, ctx)
      resolved_parts.append(resolved)
    return ca.MathExpression(resolved_parts, inline=inline)

  return ca.Expr(builder)


def _parse_picture(spec: Any) -> ca.TemplateElement:
  if isinstance(spec, dict):
    path = spec.get("path") or spec.get("file")
    data_b64 = spec.get("data")
    caption = spec.get("caption")
    width = spec.get("width")
  else:
    path = spec
    data_b64 = None
    caption = None
    width = None

  def builder(ctx):
    img_data = None
    if data_b64:
      img_bytes = base64.b64decode(str(data_b64))
      img_data = io.BytesIO(img_bytes)
    elif path:
      resolved = str(path)
      base_dir = None
      if isinstance(ctx, dict):
        base_dir = ctx.get("yaml_base_dir")
      elif hasattr(ctx, "data"):
        base_dir = ctx.data.get("yaml_base_dir")
      if base_dir and not os.path.isabs(resolved):
        resolved = os.path.join(str(base_dir), resolved)
      with open(resolved, "rb") as handle:
        img_bytes = handle.read()
      img_data = io.BytesIO(img_bytes)
    if img_data is None:
      raise ValueError("picture node requires 'path' or 'data'.")
    caption_value = _eval_field(caption, ctx) if caption is not None else None
    width_value = _eval_field(width, ctx) if width is not None else None
    return ca.Picture(img_data, caption=caption_value, width=width_value)

  return ca.Expr(builder)


def _parse_line_break(spec: Any) -> ca.Element:
  return ca.LineBreak()


def _parse_only_latex(spec: Any) -> ca.Element:
  nodes = spec
  if isinstance(spec, dict):
    nodes = spec.get("children", spec.get("nodes", spec.get("value", [])))
  elements = []
  for node in _normalize_nodes(nodes):
    element = _parse_node(node)
    if element is None:
      continue
    elements.append(element)
  return ca.OnlyLatex(elements)


def _parse_only_html(spec: Any) -> ca.Element:
  nodes = spec
  if isinstance(spec, dict):
    nodes = spec.get("children", spec.get("nodes", spec.get("value", [])))
  elements = []
  for node in _normalize_nodes(nodes):
    element = _parse_node(node)
    if element is None:
      continue
    elements.append(element)
  return ca.OnlyHtml(elements)


def _parse_choose(spec: Any) -> ca.TemplateElement:
  if isinstance(spec, dict):
    options = spec.get("options", []) or []
    weights = spec.get("weights")
  else:
    options = spec or []
    weights = None

  parsed_options = []
  for option in _normalize_nodes(options):
    element = _parse_node(option)
    if element is None:
      continue
    parsed_options.append(element)
  parsed_weights = None
  if weights is not None:
    parsed_weights = list(weights)
  return ca.Choose(parsed_options, weights=parsed_weights)


def _parse_table_group(spec: Any) -> ca.Element:
  if isinstance(spec, dict):
    tables = spec.get("tables", []) or []
  else:
    tables = spec or []

  group = ca.TableGroup()
  for entry in tables:
    label = None
    table_spec = entry
    if isinstance(entry, dict) and "table" in entry:
      label = entry.get("label")
      table_spec = entry.get("table")
    if isinstance(table_spec, dict) and "table" in table_spec:
      table = _parse_node(table_spec)
    else:
      table = _parse_table(table_spec)
    if table is None:
      continue
    label_element = _parse_inline(label) if label is not None else None
    group.tables.append((label_element, table))
  return group


def _parse_repeated_problem_part(spec: Any) -> ca.Element:
  if isinstance(spec, dict):
    parts = spec.get("parts", spec.get("subparts", [])) or []
  else:
    parts = spec or []

  parsed_parts = []
  for part in parts:
    if isinstance(part, (list, tuple)):
      parsed_parts.append([_parse_inline(item) for item in part])
    else:
      parsed_parts.append(_parse_inline(part))
  return ca.RepeatedProblemPart(parsed_parts)


def _parse_answer(spec: dict[str, Any]) -> ca.TemplateElement:
  if not isinstance(spec, dict):
    raise ValueError("answer node must be a mapping.")

  answer_type = str(spec.get("type", "string")).lower()
  strict = bool(spec.get("strict", False))
  require_prefix = bool(spec.get("require_prefix", False))

  if answer_type.endswith("_strict"):
    strict = True
    answer_type = answer_type.replace("_strict", "")

  def builder(ctx):
    value = _eval_field(spec.get("value"), ctx)
    label = _eval_field(spec.get("label", ""), ctx)
    unit = _eval_field(spec.get("unit", ""), ctx)
    length = _eval_field(spec.get("length"), ctx)
    blank_length = _eval_field(spec.get("blank_length", 5), ctx)
    baffles = _eval_field(spec.get("baffles"), ctx)
    pdf_only = bool(_eval_field(spec.get("pdf_only", False), ctx))
    order_matters = bool(_eval_field(spec.get("order_matters", True), ctx))

    if answer_type in {"binary", "hex", "decimal"} and strict:
      strict_map = {
        "binary": ca.AnswerTypes.BinaryStrict,
        "hex": ca.AnswerTypes.HexStrict,
        "decimal": ca.AnswerTypes.DecimalStrict,
      }
      answer_cls = strict_map[answer_type]
      return answer_cls(
        value=value,
        length=length,
        label=label,
        unit=unit,
        blank_length=blank_length,
        pdf_only=pdf_only,
        require_prefix=require_prefix
      )

    if answer_type == "binary":
      return ca.AnswerTypes.Binary(
        value=value,
        length=length,
        label=label,
        unit=unit,
        blank_length=blank_length,
        pdf_only=pdf_only
      )
    if answer_type == "hex":
      return ca.AnswerTypes.Hex(
        value=value,
        length=length,
        label=label,
        unit=unit,
        blank_length=blank_length,
        pdf_only=pdf_only
      )
    if answer_type == "decimal":
      return ca.AnswerTypes.Decimal(
        value=value,
        length=length,
        label=label,
        unit=unit,
        blank_length=blank_length,
        pdf_only=pdf_only
      )
    if answer_type == "int":
      return ca.AnswerTypes.Int(
        value=value,
        label=label,
        unit=unit,
        blank_length=blank_length,
        pdf_only=pdf_only
      )
    if answer_type == "float":
      return ca.AnswerTypes.Float(
        value=value,
        label=label,
        unit=unit,
        blank_length=blank_length,
        pdf_only=pdf_only
      )
    if answer_type == "string":
      return ca.AnswerTypes.String(
        value=value,
        label=label,
        unit=unit,
        blank_length=blank_length,
        pdf_only=pdf_only
      )
    if answer_type == "list":
      return ca.AnswerTypes.List(
        value=value,
        label=label,
        unit=unit,
        blank_length=blank_length,
        pdf_only=pdf_only,
        order_matters=order_matters
      )
    if answer_type == "vector":
      return ca.AnswerTypes.Vector(
        value=value,
        label=label,
        unit=unit,
        blank_length=blank_length,
        pdf_only=pdf_only
      )
    if answer_type == "matrix":
      matrix_value = value
      if isinstance(matrix_value, list):
        matrix_value = np.array(matrix_value)
      return ca.AnswerTypes.Matrix(
        value=matrix_value,
        label=label,
        pdf_only=pdf_only
      )
    if answer_type == "open_ended":
      return ca.AnswerTypes.OpenEnded(
        value=value,
        label=label,
        unit=unit,
        blank_length=blank_length,
        pdf_only=pdf_only
      )
    if answer_type == "dropdown":
      return ca.Answer.dropdown(
        value=value,
        baffles=baffles,
        label=label,
        unit=unit,
        blank_length=blank_length,
        pdf_only=pdf_only
      )
    if answer_type == "multiple_choice":
      return ca.Answer.multiple_choice(
        key="",
        value=value,
        baffles=baffles,
        label=label,
        unit=unit,
        blank_length=blank_length,
        pdf_only=pdf_only
      )

    raise ValueError(f"Unknown answer type '{answer_type}'.")

  return ca.Expr(builder)


def _evaluate_var_def(var_def: Any, context: QuestionContext | dict[str, Any]) -> Any:
  if isinstance(var_def, dict):
    var_type = str(var_def.get("type", "expr")).lower()
    if var_type == "int":
      min_val = _eval_field(var_def.get("min", 0), context)
      max_val = _eval_field(var_def.get("max", 1), context)
      rng = _get_rng(context)
      return rng.randint(int(min_val), int(max_val))
    if var_type == "float":
      min_val = _eval_field(var_def.get("min", 0.0), context)
      max_val = _eval_field(var_def.get("max", 1.0), context)
      rng = _get_rng(context)
      return rng.uniform(float(min_val), float(max_val))
    if var_type in {"choice", "weighted_choice"}:
      options = _eval_field(var_def.get("options", []), context)
      weights = _eval_field(var_def.get("weights"), context)
      rng = _get_rng(context)
      if weights is not None:
        return rng.choices(list(options), weights=list(weights), k=1)[0]
      return rng.choice(list(options))
    if var_type == "bool":
      rng = _get_rng(context)
      return rng.choice([True, False])
    if var_type == "literal":
      return var_def.get("value")
    expr = var_def.get("expr")
    if expr is not None:
      return _eval_expr(str(expr), context)
    return var_def.get("value")
  return var_def


def _eval_field(value: Any, context: QuestionContext | dict[str, Any]) -> Any:
  if isinstance(value, str):
    expr_match = _EXPR_ONLY_RE.match(value)
    if expr_match:
      return _eval_expr(expr_match.group(1).strip(), context)
    if "{{" in value:
      return _render_template(value, context)
    return value
  if isinstance(value, list):
    return [_eval_field(item, context) for item in value]
  if isinstance(value, dict):
    return {k: _eval_field(v, context) for k, v in value.items()}
  return value


def _eval_expr(expr: str, context: QuestionContext | dict[str, Any]) -> Any:
  env = _build_eval_env(context)
  return eval(expr, {"__builtins__": {}}, env)


def _exec_python(code: str, context: QuestionContext | dict[str, Any]) -> Any:
  env = _build_eval_env(context)

  def set_value(name, value):
    if hasattr(context, "data"):
      context.data[name] = value
    elif isinstance(context, dict):
      context[name] = value
    else:
      setattr(context, name, value)
    return value

  env["set"] = set_value
  env["context"] = context
  exec(code, {"__builtins__": {}}, env)
  return env.get("result")


def _apply_derived_entry(entry: Any, context: QuestionContext | dict[str, Any]) -> None:
  if entry is None:
    return
  if not isinstance(entry, dict):
    raise ValueError("derived list entries must be mappings.")

  name = entry.get("name")
  if "expr" in entry:
    if name is None:
      raise ValueError("derived expr entries must include a name.")
    context[name] = _eval_expr(str(entry.get("expr", "")), context)
    return

  if "python" in entry:
    result = _exec_python(entry.get("python", ""), context)
    if name is not None:
      context[name] = result
    return

  raise ValueError("derived entries must include 'expr' or 'python'.")


def _render_template(text: str, context: QuestionContext | dict[str, Any]) -> str:
  def repl(match: re.Match) -> str:
    expr = match.group(1).strip()
    return str(_eval_expr(expr, context))
  return _TEMPLATE_RE.sub(repl, text)


def _get_rng(context: QuestionContext | dict[str, Any]) -> random.Random:
  rng = getattr(context, "rng", None)
  if rng is None and hasattr(context, "get"):
    rng = context.get("rng")
  if rng is None:
    raise ValueError("Context does not contain rng.")
  return rng


def _build_eval_env(context: QuestionContext | dict[str, Any]) -> dict[str, Any]:
  rng = _get_rng(context)

  def randint(a, b):
    return rng.randint(int(a), int(b))

  def choice(seq):
    return rng.choice(list(seq))

  def weighted_choice(seq, weights):
    return rng.choices(list(seq), weights=list(weights), k=1)[0]

  def uniform(a, b):
    return rng.uniform(float(a), float(b))

  def sample(seq, k):
    return rng.sample(list(seq), int(k))

  def bin_fmt(value, width=None):
    if width is None:
      return format(int(value), "b")
    return format(int(value), f"0{int(width)}b")

  def hex_fmt(value, width=None, upper=True):
    fmt = "X" if upper else "x"
    if width is None:
      return format(int(value), fmt)
    return format(int(value), f"0{int(width)}{fmt}")

  def var(name):
    if hasattr(context, "data"):
      return context.data[name]
    if isinstance(context, dict):
      return context[name]
    return getattr(context, name)

  env = {
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "int": int,
    "float": float,
    "str": str,
    "len": len,
    "math": math,
    "ceil": math.ceil,
    "floor": math.floor,
    "log2": math.log2,
    "randint": randint,
    "choice": choice,
    "weighted_choice": weighted_choice,
    "uniform": uniform,
    "sample": sample,
    "bin": bin_fmt,
    "hex": hex_fmt,
    "var": var,
  }

  if hasattr(context, "data"):
    env.update(context.data)
  elif isinstance(context, dict):
    env.update(context)

  return env


ca.register_yaml_node(
  "paragraph",
  _parse_paragraph,
  form_spec={
    "title": "Paragraph",
    "fields": [
      {"name": "lines", "type": "inline_list", "required": True},
    ],
  }
)
ca.register_yaml_node(
  "text",
  _parse_text,
  form_spec={
    "title": "Text",
    "fields": [
      {"name": "value", "type": "templated_string", "required": True},
    ],
  }
)
ca.register_yaml_node(
  "equation",
  lambda spec: _parse_equation(spec, inline=False),
  form_spec={
    "title": "Equation",
    "fields": [
      {"name": "latex", "type": "templated_string", "required": True},
      {"name": "inline", "type": "bool", "default": False},
    ],
  }
)
ca.register_yaml_node(
  "code",
  _parse_code,
  form_spec={
    "title": "Code Block",
    "fields": [
      {"name": "content", "type": "templated_string", "required": True},
    ],
  }
)
ca.register_yaml_node(
  "table",
  _parse_table,
  form_spec={
    "title": "Table",
    "fields": [
      {"name": "headers", "type": "inline_list"},
      {"name": "rows", "type": "inline_grid", "required": True},
      {
        "name": "alignments",
        "type": "list",
        "item_type": "enum",
        "options": ["left", "center", "right"]
      },
      {"name": "padding", "type": "bool", "default": False},
      {"name": "hide_rules", "type": "bool", "default": False},
    ],
  }
)
ca.register_yaml_node(
  "answer",
  _parse_answer,
  form_spec={
    "title": "Answer",
    "fields": [
      {
        "name": "type",
        "type": "enum",
        "required": True,
        "options": [
          "int", "float", "string",
          "binary", "hex", "decimal",
          "list", "vector", "matrix",
          "open_ended",
          "dropdown", "multiple_choice",
        ],
      },
      {"name": "value", "type": "expr", "required": True},
      {"name": "label", "type": "templated_string"},
      {"name": "unit", "type": "templated_string"},
      {"name": "length", "type": "expr"},
      {"name": "blank_length", "type": "expr", "default": 5},
      {"name": "strict", "type": "bool", "default": False},
      {"name": "require_prefix", "type": "bool", "default": False},
      {"name": "baffles", "type": "expr_list"},
      {"name": "pdf_only", "type": "bool", "default": False},
      {"name": "order_matters", "type": "bool", "default": True},
    ],
  }
)
ca.register_yaml_node(
  "answer_block",
  _parse_answer_block,
  form_spec={
    "title": "Answer Block",
    "fields": [
      {"name": "answers", "type": "node_list", "required": True, "node_filter": ["answer"]},
    ],
  }
)
ca.register_yaml_node(
  "when",
  _parse_when,
  form_spec={
    "title": "Conditional",
    "fields": [
      {"name": "cond", "type": "expr", "required": True},
      {"name": "then", "type": "node", "required": True},
      {"name": "else", "type": "node"},
    ],
  }
)
ca.register_yaml_node(
  "section",
  _parse_section,
  form_spec={
    "title": "Section",
    "fields": [
      {"name": "children", "type": "node_list", "required": True},
    ],
  }
)
ca.register_yaml_node(
  "matrix",
  _parse_matrix,
  form_spec={
    "title": "Matrix",
    "fields": [
      {"name": "data", "type": "inline_grid", "required": True},
      {
        "name": "bracket_type",
        "type": "enum",
        "options": ["p", "b", "v", "B", "V"],
        "default": "p",
      },
      {"name": "inline", "type": "bool", "default": False},
      {"name": "name", "type": "templated_string"},
    ],
  }
)
ca.register_yaml_node(
  "math_expression",
  _parse_math_expression,
  form_spec={
    "title": "Math Expression",
    "fields": [
      {"name": "parts", "type": "inline_list", "required": True},
      {"name": "inline", "type": "bool", "default": False},
    ],
  }
)
ca.register_yaml_node(
  "picture",
  _parse_picture,
  form_spec={
    "title": "Picture",
    "fields": [
      {"name": "path", "type": "string"},
      {"name": "data", "type": "string"},
      {"name": "caption", "type": "templated_string"},
      {"name": "width", "type": "templated_string"},
    ],
  }
)
ca.register_yaml_node(
  "line_break",
  _parse_line_break,
  form_spec={
    "title": "Line Break",
    "fields": [],
  }
)
ca.register_yaml_node(
  "only_latex",
  _parse_only_latex,
  form_spec={
    "title": "LaTeX Only",
    "fields": [
      {"name": "children", "type": "node_list", "required": True},
    ],
  }
)
ca.register_yaml_node(
  "only_html",
  _parse_only_html,
  form_spec={
    "title": "HTML Only",
    "fields": [
      {"name": "children", "type": "node_list", "required": True},
    ],
  }
)
ca.register_yaml_node(
  "choose",
  _parse_choose,
  form_spec={
    "title": "Choose",
    "fields": [
      {"name": "options", "type": "node_list", "required": True},
      {"name": "weights", "type": "list"},
    ],
  }
)
ca.register_yaml_node(
  "table_group",
  _parse_table_group,
  form_spec={
    "title": "Table Group",
    "fields": [
      {"name": "tables", "type": "node_list", "required": True},
    ],
  }
)
ca.register_yaml_node(
  "repeated_problem_part",
  _parse_repeated_problem_part,
  form_spec={
    "title": "Repeated Problem Part",
    "fields": [
      {"name": "parts", "type": "node_list", "required": True},
    ],
  }
)
