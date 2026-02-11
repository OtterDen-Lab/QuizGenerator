from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pathlib
import random
import time
import zipfile
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml
try:
  from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
  load_dotenv = None


BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
import sys
if str(BASE_DIR) not in sys.path:
  sys.path.insert(0, str(BASE_DIR))

import QuizGenerator.contentast as ca
import QuizGenerator.yaml_question  # registers YAML nodes
from QuizGenerator import generate as quiz_generate
from QuizGenerator.question import QuestionContext, QuestionRegistry, Question


HTML_PATH = BASE_DIR / "documentation" / "question_builder_ui.html"
ENV_PATH: str | None = None

def _load_env() -> None:
  global ENV_PATH
  if load_dotenv is None:
    return
  repo_env = BASE_DIR / ".env"
  loaded = False
  source = None
  if repo_env.exists():
    loaded = load_dotenv(repo_env)
    source = str(repo_env)
  else:
    source = os.path.join(os.path.expanduser("~"), ".env")
    loaded = load_dotenv(source)
  if source and pathlib.Path(source).exists():
    ENV_PATH = source
  key = os.environ.get("QUIZ_ENCRYPTION_KEY")
  if key:
    fingerprint = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    origin = source if loaded else "environment"
    print(f"QUIZ_ENCRYPTION_KEY loaded from {origin} (sha256:{fingerprint})")
  else:
    if loaded:
      print(f"Loaded {source} (QUIZ_ENCRYPTION_KEY not set)")
    else:
      print("QUIZ_ENCRYPTION_KEY not set; using environment variables")


def _coerce_quiz_spec(spec: dict[str, Any]) -> dict[str, Any]:
  questions = spec.get("questions")
  if not isinstance(questions, dict):
    return spec
  converted: dict[Any, Any] = {}
  for key, value in questions.items():
    new_key = key
    if isinstance(key, str):
      stripped = key.strip()
      try:
        new_key = int(stripped)
      except ValueError:
        try:
          new_key = float(stripped)
        except ValueError:
          new_key = key
    converted[new_key] = value
  spec = dict(spec)
  spec["questions"] = converted
  return spec


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
  if value is None:
    return default
  return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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

  def _get_canvas_interface(self, use_prod: bool) -> Any:
    try:
      from lms_interface.canvas_interface import CanvasInterface
    except Exception as exc:  # pragma: no cover - optional dependency
      raise RuntimeError(f"Canvas interface unavailable: {exc}") from exc
    if ENV_PATH:
      return CanvasInterface(prod=use_prod, env_path=ENV_PATH)
    return CanvasInterface(prod=use_prod)

  @staticmethod
  def _canvas_env_label(canvas_url: str | None) -> str:
    if not canvas_url:
      return "UNKNOWN"
    lowered = canvas_url.lower()
    return "DEV" if ("beta" in lowered or "test" in lowered) else "PROD"

  def do_GET(self) -> None:
    parsed = urlparse(self.path)
    path_only = parsed.path

    if path_only == "/" or path_only.startswith("/index"):
      html = self._load_html()
      if html is None:
        self._send_text("question_builder_ui.html not found.", status=HTTPStatus.NOT_FOUND)
        return
      self._send_text(html, content_type="text/html; charset=utf-8")
      return

    if path_only == "/form_spec":
      specs = ca.list_yaml_nodes()
      self._send_json({"nodes": specs, "version": 1})
      return

    if path_only == "/premade_list":
      items = QuestionRegistry.list_registered()
      self._send_json({"items": items})
      return

    if path_only == "/canvas/courses":
      query = parse_qs(parsed.query)
      use_prod = _parse_bool((query.get("use_prod") or [None])[0], default=False)
      try:
        canvas_interface = self._get_canvas_interface(use_prod)
        courses = canvas_interface.canvas.get_courses(
          enrollment_state="active",
          enrollment_type="teacher",
          include=["term", "favorites"]
        )
        course_list = []
        for course in courses:
          if not hasattr(course, "name"):
            continue
          workflow_state = getattr(course, "workflow_state", None)
          if workflow_state != "available":
            continue
          course_list.append({
            "id": course.id,
            "name": course.name,
            "start_at": getattr(course, "start_at", None),
            "enrollment_term_id": getattr(course, "enrollment_term_id", None),
            "is_favorite": getattr(course, "is_favorite", False),
          })

        course_list.sort(
          key=lambda c: (
            c["is_favorite"],
            c["enrollment_term_id"] or 0,
            c["start_at"] or "",
          ),
          reverse=True
        )

        self._send_json({
          "courses": course_list,
          "environment": self._canvas_env_label(canvas_interface.canvas_url),
          "canvas_url": canvas_interface.canvas_url,
        })
      except Exception as exc:
        self._send_json({"error": f"Failed to fetch courses: {exc}"}, status=HTTPStatus.BAD_REQUEST)
      return

    return super().do_GET()

  def do_POST(self) -> None:
    parsed = urlparse(self.path)
    path_only = parsed.path

    if path_only not in {
      "/to_yaml",
      "/from_yaml",
      "/preview",
      "/save_yaml",
      "/preview_premade",
      "/export_pdf",
      "/canvas/upload",
    }:
      self._send_text("Not found.", status=HTTPStatus.NOT_FOUND)
      return

    content_length = int(self.headers.get("Content-Length", "0"))
    raw = self.rfile.read(content_length)
    try:
      payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
      self._send_json({"error": f"Invalid JSON: {exc}"}, status=HTTPStatus.BAD_REQUEST)
      return

    if path_only == "/to_yaml":
      spec = payload.get("spec")
      if not isinstance(spec, dict):
        self._send_json({"error": "Payload must include a 'spec' object."}, status=HTTPStatus.BAD_REQUEST)
        return

      yaml_text = yaml.safe_dump(_coerce_quiz_spec(spec), sort_keys=False)
      self._send_json({"yaml": yaml_text})
      return

    if path_only == "/from_yaml":
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

    if path_only == "/preview":
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

    if path_only == "/preview_premade":
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

    if path_only == "/save_yaml":
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
        yaml_text = yaml.safe_dump(_coerce_quiz_spec(spec), sort_keys=False)

      if yaml_text is None:
        self._send_json({"error": "No YAML content provided."}, status=HTTPStatus.BAD_REQUEST)
        return

      if not yaml_text.endswith("\n"):
        yaml_text += "\n"

      save_path.parent.mkdir(parents=True, exist_ok=True)
      save_path.write_text(yaml_text, encoding="utf-8")
      self._send_json({"ok": True, "path": str(save_path)})
      return

    if path_only == "/export_pdf":
      spec = payload.get("spec")
      yaml_text = payload.get("yaml")
      num_pdfs = payload.get("num_pdfs", 1)
      use_typst = bool(payload.get("use_typst", True))

      if spec is not None and not isinstance(spec, dict):
        self._send_json({"error": "'spec' must be a mapping."}, status=HTTPStatus.BAD_REQUEST)
        return
      if yaml_text is None and spec is None:
        self._send_json({"error": "Provide 'yaml' or 'spec' for PDF export."}, status=HTTPStatus.BAD_REQUEST)
        return
      if yaml_text is None:
        yaml_text = yaml.safe_dump(_coerce_quiz_spec(spec), sort_keys=False)
      if not isinstance(yaml_text, str):
        self._send_json({"error": "'yaml' must be a string."}, status=HTTPStatus.BAD_REQUEST)
        return

      try:
        save_path = self._resolve_save_path(payload.get("path") or "out/quiz.yaml")
      except ValueError as exc:
        self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        return

      try:
        num_pdfs = int(num_pdfs)
      except (TypeError, ValueError):
        self._send_json({"error": "'num_pdfs' must be an integer."}, status=HTTPStatus.BAD_REQUEST)
        return
      if num_pdfs < 1:
        self._send_json({"error": "'num_pdfs' must be >= 1."}, status=HTTPStatus.BAD_REQUEST)
        return

      if not yaml_text.endswith("\n"):
        yaml_text += "\n"

      save_path.parent.mkdir(parents=True, exist_ok=True)
      save_path.write_text(yaml_text, encoding="utf-8")

      ok, missing = quiz_generate._check_dependencies(
        require_typst=use_typst,
        require_latex=not use_typst
      )
      if not ok:
        self._send_json({"error": "\n".join(missing)}, status=HTTPStatus.BAD_REQUEST)
        return

      start_time = time.time()
      try:
        quiz_generate.generate_quiz(
          str(save_path),
          num_pdfs=num_pdfs,
          use_typst=use_typst
        )
      except quiz_generate.QuizGenError as exc:
        self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        return
      except Exception as exc:
        self._send_json({"error": f"PDF export failed: {type(exc).__name__}: {exc}"}, status=HTTPStatus.BAD_REQUEST)
        return

      output_dir = BASE_DIR / "out"
      pdfs: list[pathlib.Path] = []
      if output_dir.exists():
        for pdf_path in output_dir.glob("*.pdf"):
          try:
            if pdf_path.stat().st_mtime >= start_time - 1:
              pdfs.append(pdf_path)
          except OSError:
            continue

      if not pdfs:
        self._send_json({"error": "PDF export succeeded but no PDF was found."}, status=HTTPStatus.BAD_REQUEST)
        return
      replay_yaml = pathlib.Path(quiz_generate._replay_yaml_path(str(save_path)))
      if not replay_yaml.exists() or replay_yaml.stat().st_mtime < start_time - 1:
        candidates = sorted(output_dir.glob("*_replay.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
        replay_yaml = candidates[0] if candidates else replay_yaml
      if not replay_yaml.exists():
        self._send_json({"error": "Replay YAML not found. PDF export cannot be bundled."}, status=HTTPStatus.BAD_REQUEST)
        return

      archive_bytes = io.BytesIO()
      with zipfile.ZipFile(archive_bytes, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for pdf_path in sorted(pdfs, key=lambda p: p.name):
          try:
            archive.write(pdf_path, arcname=pdf_path.name)
          except OSError:
            continue
        try:
          archive.write(replay_yaml, arcname=replay_yaml.name)
        except OSError:
          pass
      archive_bytes.seek(0)
      zip_payload = archive_bytes.read()

      self.send_response(HTTPStatus.OK)
      self.send_header("Content-Type", "application/zip")
      self.send_header("Content-Disposition", "attachment; filename=\"quiz_pdfs.zip\"")
      self.send_header("Content-Length", str(len(zip_payload)))
      self.end_headers()
      self.wfile.write(zip_payload)
      return

    if path_only == "/canvas/upload":
      yaml_text = payload.get("yaml")
      spec = payload.get("spec")
      course_id = payload.get("course_id")
      num_variations = payload.get("num_variations", 1)
      is_practice_raw = payload.get("practice")
      is_practice = None
      if is_practice_raw is not None:
        if isinstance(is_practice_raw, bool):
          is_practice = is_practice_raw
        else:
          is_practice = _parse_bool(is_practice_raw, default=False)
      use_prod = _parse_bool(payload.get("use_prod"), default=False)
      title_override = payload.get("title")
      delete_assignment_group = bool(payload.get("delete_assignment_group", False))
      quiet = bool(payload.get("quiet", True))

      if yaml_text is None and spec is None:
        self._send_json({"error": "Provide 'yaml' or 'spec' for Canvas upload."}, status=HTTPStatus.BAD_REQUEST)
        return
      if spec is not None and not isinstance(spec, dict):
        self._send_json({"error": "'spec' must be a mapping."}, status=HTTPStatus.BAD_REQUEST)
        return
      if yaml_text is None:
        yaml_text = yaml.safe_dump(_coerce_quiz_spec(spec), sort_keys=False)
      if not isinstance(yaml_text, str):
        self._send_json({"error": "'yaml' must be a string."}, status=HTTPStatus.BAD_REQUEST)
        return

      try:
        course_id = int(course_id)
      except (TypeError, ValueError):
        self._send_json({"error": "'course_id' must be an integer."}, status=HTTPStatus.BAD_REQUEST)
        return

      try:
        num_variations = int(num_variations)
      except (TypeError, ValueError):
        self._send_json({"error": "'num_variations' must be an integer."}, status=HTTPStatus.BAD_REQUEST)
        return
      if num_variations < 1:
        self._send_json({"error": "'num_variations' must be >= 1."}, status=HTTPStatus.BAD_REQUEST)
        return

      if title_override is not None and not isinstance(title_override, str):
        self._send_json({"error": "'title' must be a string."}, status=HTTPStatus.BAD_REQUEST)
        return

      if not yaml_text.endswith("\n"):
        yaml_text += "\n"

      try:
        save_path = self._resolve_save_path(payload.get("path") or "out/quiz_canvas.yaml")
      except ValueError as exc:
        self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        return

      save_path.parent.mkdir(parents=True, exist_ok=True)
      save_path.write_text(yaml_text, encoding="utf-8")

      try:
        exam_dicts = list(yaml.safe_load_all(yaml_text))
      except yaml.YAMLError as exc:
        self._send_json({"error": f"Invalid YAML: {exc}"}, status=HTTPStatus.BAD_REQUEST)
        return

      try:
        from QuizGenerator.quiz import Quiz
        from QuizGenerator import generate as quiz_generate
      except Exception as exc:
        self._send_json({"error": f"Quiz generation unavailable: {exc}"}, status=HTTPStatus.BAD_REQUEST)
        return

      try:
        quizzes = Quiz.from_exam_dicts(exam_dicts, source_path=str(save_path))
      except Exception as exc:
        self._send_json({"error": f"Failed to parse quiz YAML: {exc}"}, status=HTTPStatus.BAD_REQUEST)
        return

      if not quizzes:
        self._send_json({"error": "No quizzes found in YAML."}, status=HTTPStatus.BAD_REQUEST)
        return

      try:
        canvas_interface = self._get_canvas_interface(use_prod)
        canvas_course = canvas_interface.get_course(course_id=course_id)
        assignment_group = canvas_course.create_assignment_group(
          name=payload.get("assignment_group") or "dev",
          delete_existing=delete_assignment_group
        )
      except Exception as exc:
        self._send_json({"error": f"Canvas setup failed: {exc}"}, status=HTTPStatus.BAD_REQUEST)
        return

      uploaded = []
      multiple = len(quizzes) > 1
      for idx, quiz in enumerate(quizzes, start=1):
        title = None
        if title_override:
          title = f"{title_override} ({idx})" if multiple else title_override
        else:
          title = quiz.name
        practice_flag = is_practice if is_practice is not None else quiz.practice
        try:
          canvas_quiz = quiz_generate.upload_quiz_to_canvas(
            canvas_course,
            quiz,
            num_variations,
            title=title,
            is_practice=practice_flag,
            assignment_group=assignment_group,
            quiet=quiet
          )
          uploaded.append({
            "title": title,
            "url": getattr(canvas_quiz, "html_url", None),
            "id": getattr(canvas_quiz, "id", None),
          })
        except Exception as exc:
          self._send_json({"error": f"Canvas upload failed: {exc}"}, status=HTTPStatus.BAD_REQUEST)
          return

      self._send_json({
        "ok": True,
        "count": len(uploaded),
        "uploaded": uploaded,
      })
      return


def main() -> None:
  parser = argparse.ArgumentParser(description="Run the YAML question builder UI.")
  parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
  parser.add_argument("--port", default=8787, type=int, help="Port to bind (default: 8787)")
  args = parser.parse_args()

  _load_env()
  os.chdir(BASE_DIR)
  with TCPServer((args.host, args.port), QuestionBuilderHandler) as httpd:
    print(f"Question Builder UI running at http://{args.host}:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
  main()
