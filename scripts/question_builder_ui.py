from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
from typing import Any

import yaml


BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
import sys
if str(BASE_DIR) not in sys.path:
  sys.path.insert(0, str(BASE_DIR))

import QuizGenerator.contentast as ca
import QuizGenerator.yaml_question  # registers YAML nodes
from QuizGenerator.question import QuestionContext, QuestionRegistry, Question


HTML_PATH = BASE_DIR / "documentation" / "question_builder_ui.html"


class QuestionBuilderHandler(SimpleHTTPRequestHandler):
  def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
    data = json.dumps(payload).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)

  def _send_text(
    self,
    text: str,
    status: HTTPStatus = HTTPStatus.OK,
    content_type: str = "text/plain; charset=utf-8"
  ) -> None:
    data = text.encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", content_type)
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)

  def _load_html(self) -> str | None:
    if not HTML_PATH.exists():
      return None
    return HTML_PATH.read_text(encoding="utf-8")

  def _resolve_save_path(self, path_text: str | None) -> pathlib.Path:
    target = path_text or "out/question.yaml"
    path = pathlib.Path(target)
    if not path.is_absolute():
      path = BASE_DIR / path
    path = path.resolve()
    if BASE_DIR not in path.parents and path != BASE_DIR:
      raise ValueError("Save path must be within the repository.")
    return path

  def do_GET(self) -> None:
    if self.path == "/" or self.path.startswith("/index"):
      html = self._load_html()
      if html is None:
        self._send_text("question_builder_ui.html not found.", status=HTTPStatus.NOT_FOUND)
        return
      self._send_text(html, content_type="text/html; charset=utf-8")
      return

    if self.path == "/form_spec":
      specs = ca.list_yaml_nodes()
      self._send_json({"nodes": specs, "version": 1})
      return

    if self.path == "/premade_list":
      items = QuestionRegistry.list_registered()
      self._send_json({"items": items})
      return

    return super().do_GET()

  def do_POST(self) -> None:
    if self.path not in {"/to_yaml", "/from_yaml", "/preview", "/save_yaml", "/preview_premade"}:
      self._send_text("Not found.", status=HTTPStatus.NOT_FOUND)
      return

    content_length = int(self.headers.get("Content-Length", "0"))
    raw = self.rfile.read(content_length)
    try:
      payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
      self._send_json({"error": f"Invalid JSON: {exc}"}, status=HTTPStatus.BAD_REQUEST)
      return

    if self.path == "/to_yaml":
      spec = payload.get("spec")
      if not isinstance(spec, dict):
        self._send_json({"error": "Payload must include a 'spec' object."}, status=HTTPStatus.BAD_REQUEST)
        return

      yaml_text = yaml.safe_dump(spec, sort_keys=False)
      self._send_json({"yaml": yaml_text})
      return

    if self.path == "/from_yaml":
      yaml_text = payload.get("yaml")
      if not isinstance(yaml_text, str):
        self._send_json({"error": "Payload must include a 'yaml' string."}, status=HTTPStatus.BAD_REQUEST)
        return
      try:
        spec = yaml.safe_load(yaml_text)
      except yaml.YAMLError as exc:
        self._send_json({"error": f"Invalid YAML: {exc}"}, status=HTTPStatus.BAD_REQUEST)
        return
      if not isinstance(spec, dict):
        self._send_json({"error": "YAML must define a mapping at the root."}, status=HTTPStatus.BAD_REQUEST)
        return
      self._send_json({"spec": spec})
      return

    if self.path == "/preview":
      spec = payload.get("spec")
      yaml_text = payload.get("yaml")
      show_answers = bool(payload.get("show_answers", False))
      seed = payload.get("seed")
      rng_seed = None if seed in (None, "") else int(seed)

      if isinstance(spec, dict):
        spec_dict = spec
      elif isinstance(yaml_text, str):
        try:
          spec_dict = yaml.safe_load(yaml_text)
        except yaml.YAMLError as exc:
          self._send_json({"error": f"Invalid YAML: {exc}"}, status=HTTPStatus.BAD_REQUEST)
          return
      else:
        self._send_json({"error": "Provide 'spec' or 'yaml' for preview."}, status=HTTPStatus.BAD_REQUEST)
        return

      if not isinstance(spec_dict, dict):
        self._send_json({"error": "Spec must be a mapping."}, status=HTTPStatus.BAD_REQUEST)
        return

      ctx = QuestionContext(rng_seed=rng_seed, rng=random.Random(rng_seed))
      try:
        import QuizGenerator.yaml_question as yaml_question
        yaml_question.apply_context_spec(spec_dict, ctx)
        templates = yaml_question.parse_question_templates(spec_dict)
        body = ca.resolve_template(templates.get("body"), ctx)
        explanation = ca.resolve_template(templates.get("explanation"), ctx)
        if body is None:
          body = ca.Section()
        if explanation is None:
          explanation = ca.Section()
        body_html = body.render("html", show_answers=show_answers)
        explanation_html = explanation.render("html", show_answers=show_answers)
      except Exception as exc:
        self._send_json({"error": f"Preview failed: {type(exc).__name__}: {exc}"}, status=HTTPStatus.BAD_REQUEST)
        return

      self._send_json({
        "body_html": body_html,
        "explanation_html": explanation_html,
        "context": dict(ctx.data),
      })
      return

    if self.path == "/preview_premade":
      question_class = payload.get("class")
      kwargs = payload.get("kwargs") or {}
      show_answers = bool(payload.get("show_answers", False))
      seed = payload.get("seed")
      rng_seed = None if seed in (None, "") else int(seed)

      if not isinstance(question_class, str) or not question_class.strip():
        self._send_json({"error": "Provide a 'class' for preview."}, status=HTTPStatus.BAD_REQUEST)
        return
      if not isinstance(kwargs, dict):
        self._send_json({"error": "'kwargs' must be a mapping."}, status=HTTPStatus.BAD_REQUEST)
        return

      try:
        QuestionRegistry.load_premade_questions()
        if "topic" in kwargs and isinstance(kwargs["topic"], str):
          kwargs["topic"] = Question.Topic.from_string(kwargs["topic"])
        question = QuestionRegistry.create(question_class, **kwargs)
        instance = question.instantiate(rng_seed=rng_seed, max_backoff_attempts=3)
        body_html = instance.body.render("html", show_answers=show_answers)
        explanation_html = instance.explanation.render("html", show_answers=show_answers)
      except Exception as exc:
        self._send_json({"error": f"Preview failed: {type(exc).__name__}: {exc}"}, status=HTTPStatus.BAD_REQUEST)
        return

      self._send_json({
        "body_html": body_html,
        "explanation_html": explanation_html,
      })
      return

    if self.path == "/save_yaml":
      spec = payload.get("spec")
      yaml_text = payload.get("yaml")
      if spec is not None and not isinstance(spec, dict):
        self._send_json({"error": "'spec' must be a mapping."}, status=HTTPStatus.BAD_REQUEST)
        return
      if spec is None and not isinstance(yaml_text, str):
        self._send_json({"error": "Provide 'spec' or 'yaml' to save."}, status=HTTPStatus.BAD_REQUEST)
        return

      try:
        save_path = self._resolve_save_path(payload.get("path"))
      except ValueError as exc:
        self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        return

      if save_path.exists() and save_path.is_dir():
        self._send_json({"error": "Save path points to a directory."}, status=HTTPStatus.BAD_REQUEST)
        return

      if spec is not None:
        yaml_text = yaml.safe_dump(spec, sort_keys=False)

      if yaml_text is None:
        self._send_json({"error": "No YAML content provided."}, status=HTTPStatus.BAD_REQUEST)
        return

      if not yaml_text.endswith("\n"):
        yaml_text += "\n"

      save_path.parent.mkdir(parents=True, exist_ok=True)
      save_path.write_text(yaml_text, encoding="utf-8")
      self._send_json({"ok": True, "path": str(save_path)})
      return


def main() -> None:
  parser = argparse.ArgumentParser(description="Run the YAML question builder UI.")
  parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
  parser.add_argument("--port", default=8787, type=int, help="Port to bind (default: 8787)")
  args = parser.parse_args()

  os.chdir(BASE_DIR)
  with TCPServer((args.host, args.port), QuestionBuilderHandler) as httpd:
    print(f"Question Builder UI running at http://{args.host}:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
  main()
