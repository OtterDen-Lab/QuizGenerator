#!/usr/bin/env python
import argparse
import copy
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
import traceback
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from lms_interface.canvas_interface import CanvasInterface
from QuizGenerator.contentast import Answer
from QuizGenerator.performance import PerformanceTracker
from QuizGenerator.question import Question, QuestionGroup, QuestionRegistry
from QuizGenerator.quiz import Quiz

log = logging.getLogger(__name__)


class QuizGenError(Exception):
  """User-facing error for CLI operations."""



def parse_args():
  parser = argparse.ArgumentParser(
    epilog=(
      "Generation controls:\n"
      "  --max_backoff_attempts N   Limit retries for 'interesting' questions\n"
      "  --float_tolerance X        Default tolerance for float answers\n"
    )
  )

  parser.add_argument(
    "--env",
    default=os.path.join(Path.home(), '.env'),
    help="Path to .env file specifying canvas details"
  )
  
  parser.add_argument("--debug", action="store_true", help="Set logging level to debug")

  parser.add_argument(
    "--yaml",
    dest="quiz_yaml",
    default=None,
    help="Path to quiz YAML configuration"
  )
  parser.add_argument(
    "--quiz_yaml",
    dest="quiz_yaml",
    default=None,
    help=argparse.SUPPRESS  # Backwards-compatible alias for --yaml
  )
  parser.add_argument("--seed", type=int, default=None,
                     help="Random seed for quiz generation (default: None for random)")
  parser.add_argument("--quiet", action="store_true",
                     help="Disable progress bars for variation prep and uploads")

  # Canvas flags
  parser.add_argument("--num_canvas", default=0, type=int, help="How many variations of each question to try to upload to canvas.")
  parser.add_argument("--prod", action="store_true")
  parser.add_argument("--course_id", type=int)
  parser.add_argument("--delete-assignment-group", action="store_true",
                     help="Delete existing assignment group before uploading new quizzes")
  parser.add_argument(
    "--generate_practice",
    nargs="+",
    metavar="TAG",
    help=(
      "Generate one ungraded practice quiz per matching registered question type. "
      "Accepts one or more tags (comma or space separated), e.g. --generate_practice cst334 memory"
    )
  )
  parser.add_argument(
    "--practice_match",
    choices=["any", "all"],
    default="any",
    help="With --generate_practice, match any tag (default) or require all tags."
  )
  parser.add_argument(
    "--practice_variations",
    type=int,
    default=5,
    help="With --generate_practice, number of Canvas variations per question (default: 5)."
  )
  parser.add_argument(
    "--practice_points",
    type=float,
    default=1.0,
    help="With --generate_practice, point value per practice question (default: 1.0)."
  )
  parser.add_argument(
    "--practice_assignment_group",
    default="practice",
    help="With --generate_practice, assignment group name for created quizzes (default: practice)."
  )
  
  # PDF Flags
  parser.add_argument("--num_pdfs", default=0, type=int, help="How many PDF quizzes to create")
  parser.add_argument("--latex", action="store_false", dest="typst", help="Use LaTeX instead of Typst for PDF generation")
  parser.set_defaults(typst=True)
  parser.add_argument("--typst_measurement", action="store_true",
                     help="Use Typst measurement for layout optimization (experimental)")
  parser.add_argument("--consistent_pages", action="store_true",
                     help="Reserve question heights to keep pagination consistent across versions (auto-enabled when --num_pdfs > 1)")
  parser.add_argument("--layout_samples", type=int, default=10,
                     help="Number of deterministic samples per question to estimate height")
  parser.add_argument("--layout_safety_factor", type=float, default=1.1,
                     help="Multiplier applied to max sampled height for safety")
  parser.add_argument("--optimize_space", action="store_true",
                     help="Optimize question order to reduce PDF page count (affects Canvas order too)")
  parser.add_argument("--no_embed_images_typst", action="store_false", dest="embed_images_typst",
                     help="Disable embedding images in Typst output")
  parser.set_defaults(embed_images_typst=True)

  # Testing flags
  parser.add_argument("--test_all", type=int, default=0, metavar="N",
                     help="Generate N variations of ALL registered questions to test they work correctly")
  parser.add_argument("--test_questions", nargs='+', metavar="NAME",
                     help="Only test specific question types by name (use with --test_all)")
  parser.add_argument("--strict", action="store_true",
                     help="With --test_all, skip PDF/Canvas generation if any questions fail")
  parser.add_argument("--skip_missing_extras", action="store_true",
                     help="With --test_all, skip questions that fail due to missing optional dependencies")
  parser.add_argument("--allow_generator", action="store_true",
                     help="Enable FromGenerator questions (executes Python from YAML)")
  parser.add_argument("--check-deps", action="store_true",
                     help="Check external dependencies (Typst/LaTeX/Pandoc) and exit")
  parser.add_argument("--max_backoff_attempts", type=int, default=None,
                     help="Max attempts for question generation backoff (default: 200)")
  parser.add_argument("--float_tolerance", type=float, default=None,
                     help="Default numeric tolerance for float answers (default: 0.01)")

  subparsers = parser.add_subparsers(dest='command')
  test_parser = subparsers.add_parser("TEST")


  args = parser.parse_args()

  if args.num_canvas > 0 and args.course_id is None:
    parser.error("Missing --course_id for Canvas upload. Example: --course_id 12345")

  if args.generate_practice and args.course_id is None:
    parser.error("Missing --course_id for --generate_practice. Example: --course_id 12345")

  if args.generate_practice and args.practice_variations < 1:
    parser.error("--practice_variations must be >= 1")

  if args.generate_practice and args.practice_points < 0:
    parser.error("--practice_points must be non-negative")

  if args.test_all <= 0 and not args.quiz_yaml and not args.generate_practice:
    parser.error("Missing --yaml. Example: quizgen --yaml example_files/example_exam.yaml --num_pdfs 1")

  return args


def _check_dependencies(*, require_typst: bool, require_latex: bool) -> tuple[bool, list[str]]:
  missing = []

  if require_typst and shutil.which("typst") is None:
    missing.append("Typst not found. Install from https://typst.app/ or ensure `typst` is in PATH.")

  if require_latex and shutil.which("latexmk") is None:
    missing.append("latexmk not found. Install a LaTeX distribution that provides latexmk.")

  # Pandoc is optional but improves markdown rendering.
  if shutil.which("pandoc") is None:
    log.warning("Pandoc not found. Markdown rendering may be lower quality.")

  return (len(missing) == 0), missing


def test():
  log.info("Running test...")

  print("\n" + "="*60)
  print("TEST COMPLETE")
  print("="*60)


def test_all_questions(
    num_variations: int,
    generate_pdf: bool = False,
    use_typst: bool = True,
    canvas_course=None,
    strict: bool = False,
    question_filter: list = None,
    skip_missing_extras: bool = False,
    embed_images_typst: bool = False
):
  """
  Test all registered questions by generating N variations of each.

  This helps verify that all question types work correctly and can generate
  valid output without errors.

  Args:
    num_variations: Number of variations to generate for each question type
    generate_pdf: If True, generate a PDF with all successful questions
    use_typst: If True, use Typst for PDF generation; otherwise use LaTeX
    canvas_course: If provided, push a test quiz to this Canvas course
    strict: If True, skip PDF/Canvas generation if any questions fail
    question_filter: If provided, only test questions whose names contain one of these strings (case-insensitive)
  """
  # Allow FromGenerator during test_all runs so coverage includes generator-based questions.
  os.environ["QUIZGEN_ALLOW_GENERATOR"] = "1"
  # Ensure all premade questions are loaded
  QuestionRegistry.load_premade_questions()

  registered_questions = QuestionRegistry._registry

  # Filter questions if a filter list is provided
  if question_filter:
    filter_lower = [f.lower() for f in question_filter]
    registered_questions = {
      name: cls for name, cls in registered_questions.items()
      if any(f in name.lower() for f in filter_lower)
    }
    if not registered_questions:
      print(f"No questions matched filter: {question_filter}")
      print(f"Available questions: {sorted(QuestionRegistry._registry.keys())}")
      return False
    print(f"Filtered to {len(registered_questions)} questions matching: {question_filter}")

  total_questions = len(registered_questions)

  # Test defaults for questions that require external input
  # These are "template" questions that can't work without content
  TEST_DEFAULTS = {
    'fromtext': {'text': 'Test question placeholder text.'},
    'fromgenerator': {'generator': 'return "Generated test content"'},
  }

  print(f"\nTesting {total_questions} registered question types with {num_variations} variations each...")
  print("=" * 70)

  failed_questions = []
  skipped_questions = []
  successful_questions = []
  # Collect question instances for PDF/Canvas generation
  test_question_instances = []

  for i, (question_name, question_class) in enumerate(sorted(registered_questions.items()), 1):
    print(f"\n[{i}/{total_questions}] Testing: {question_name}")
    print(f"  Class: {question_class.__name__}")

    question_failures = []

    for variation in range(num_variations):
      seed = variation * 1000  # Use different seeds for each variation
      try:
        # Get any test defaults for this question type
        extra_kwargs = TEST_DEFAULTS.get(question_name, {})

        # Create question instance with minimal required params
        question = question_class(
          name=f"{question_name} (v{variation+1})",
          points_value=1.0,
          **extra_kwargs
        )

        # Generate the question (this calls refresh and builds the AST)
        instance = question.instantiate(rng_seed=seed, max_backoff_attempts=200)
        question_ast = question._build_question_ast(instance)

        # Try rendering to both formats to catch format-specific issues
        try:
          question_ast.render("html")
        except Exception as e:
          tb = traceback.format_exc()
          question_failures.append(f"  Variation {variation+1}: HTML render failed - {e}\n{tb}")
          continue

        try:
          question_ast.render("typst")
        except Exception as e:
          tb = traceback.format_exc()
          question_failures.append(f"  Variation {variation+1}: Typst render failed - {e}\n{tb}")
          continue

        # If we got here, the question works - save the instance
        test_question_instances.append(question)

      except ImportError as e:
        if skip_missing_extras:
          skipped_questions.append(question_name)
          log.warning(f"Skipping {question_name} due to missing optional dependency: {e}")
          question_failures = []
          break
        tb = traceback.format_exc()
        question_failures.append(f"  Variation {variation+1}: Generation failed - {e}\n{tb}")
      except Exception as e:
        tb = traceback.format_exc()
        question_failures.append(f"  Variation {variation+1}: Generation failed - {e}\n{tb}")

    if question_failures:
      print(f"  FAILED ({len(question_failures)}/{num_variations} variations)")
      for failure in question_failures:
        print(failure)
      failed_questions.append((question_name, question_failures))
    elif question_name in skipped_questions:
      print("  SKIPPED (missing optional dependency)")
    else:
      print(f"  OK ({num_variations}/{num_variations} variations)")
      successful_questions.append(question_name)

  # Summary
  print("\n" + "=" * 70)
  print("TEST SUMMARY")
  print("=" * 70)
  print(f"Total question types: {total_questions}")
  print(f"Successful: {len(successful_questions)}")
  print(f"Failed: {len(failed_questions)}")
  if skipped_questions:
    print(f"Skipped (missing extras): {len(set(skipped_questions))}")

  if failed_questions:
    print("\nFailed questions:")
    for name, failures in failed_questions:
      print(f"  - {name}: {len(failures)} failures")
  if skipped_questions:
    print("\nSkipped questions (missing extras):")
    for name in sorted(set(skipped_questions)):
      print(f"  - {name}")

  print("=" * 70)

  # Generate PDF and/or push to Canvas if requested
  if strict and failed_questions:
    print("\n[STRICT MODE] Skipping PDF/Canvas generation due to failures")
  elif (generate_pdf or canvas_course) and test_question_instances:
    print(f"\nCreating test quiz with {len(test_question_instances)} questions...")

    # Create a Quiz object with all successful questions
    test_quiz = Quiz(
      name="Test All Questions",
      questions=test_question_instances,
      practice=True
    )

    if generate_pdf:
      print("Generating PDF...")
      pdf_seed = 12345  # Fixed seed for reproducibility
      if use_typst:
        typst_text = test_quiz.get_quiz(rng_seed=pdf_seed).render(
          "typst",
          embed_images_typst=embed_images_typst
        )
        if not generate_typst(typst_text, remove_previous=True, name_prefix="test_all_questions"):
          log.error("Test PDF generation failed (Typst).")
          return False
      else:
        latex_text = test_quiz.get_quiz(rng_seed=pdf_seed).render_latex()
        if not generate_latex(latex_text, remove_previous=True, name_prefix="test_all_questions"):
          log.error("Test PDF generation failed (LaTeX).")
          return False
      print("PDF generated in out/ directory")

    if canvas_course:
      print("Pushing to Canvas...")
      quiz_title = f"Test All Questions ({int(datetime.now().timestamp())} : {datetime.now().strftime('%b %d %I:%M%p')})"
      upload_quiz_to_canvas(
        canvas_course,
        test_quiz,
        1,
        title=quiz_title,
        is_practice=True
      )
      print(f"Quiz '{quiz_title}' pushed to Canvas")

  return len(failed_questions) == 0


def _practice_question_defaults(registered_name: str) -> dict:
  defaults = {
    "fromtext": {"text": "Practice placeholder question."},
    "fromgenerator": {"generator": "return 'Practice placeholder question.'"},
    "fromyaml": {"yaml_spec": {"body": ["Practice placeholder question."], "explanation": []}},
  }
  return defaults.get(registered_name.lower(), {})


def _tags_match(candidate_tags: set[str], requested_tags: set[str], *, match_all: bool) -> bool:
  if not requested_tags:
    return True
  if match_all:
    return requested_tags.issubset(candidate_tags)
  return bool(candidate_tags & requested_tags)


def _build_practice_question(registered_name: str, question_cls, *, points_value: float):
  pretty_name = question_cls.__name__.replace("Question", "") or question_cls.__name__
  try:
    question = QuestionRegistry.create(
      registered_name,
      name=pretty_name,
      points_value=points_value,
      **_practice_question_defaults(registered_name)
    )
    return question, None
  except Exception as exc:
    return None, str(exc)


def generate_practice_quizzes(
    *,
    tag_filters: list[str],
    course_id: int,
    use_prod: bool = False,
    env_path: str | None = None,
    num_variations: int = 5,
    points_value: float = 1.0,
    delete_assignment_group: bool = False,
    assignment_group_name: str = "practice",
    match_all: bool = False,
    quiet: bool = False,
    max_backoff_attempts=None
):
  requested_tags = Question.normalize_tags(tag_filters)
  if not requested_tags:
    raise QuizGenError("No valid tags supplied for --generate_practice.")

  QuestionRegistry.load_premade_questions()
  selected: list[tuple[str, object, set[str]]] = []
  skipped: list[tuple[str, str]] = []
  available_tags: set[str] = set()

  for registered_name, question_cls in sorted(QuestionRegistry._registry.items()):
    question, error = _build_practice_question(
      registered_name,
      question_cls,
      points_value=points_value
    )
    if question is None:
      skipped.append((registered_name, error or "unknown error"))
      continue

    question_tags = set(getattr(question, "tags", set()))
    question_tags.update(Question.infer_course_tags(registered_name, question_cls.__module__))
    available_tags.update(question_tags)
    if _tags_match(question_tags, requested_tags, match_all=match_all):
      selected.append((registered_name, question, question_tags))

  if not selected:
    available = ", ".join(sorted(available_tags)) if available_tags else "<none>"
    match_mode = "all" if match_all else "any"
    raise QuizGenError(
      f"No practice questions matched tags {sorted(requested_tags)} with match mode '{match_mode}'. "
      f"Available tags: {available}"
    )

  canvas_interface = CanvasInterface(prod=use_prod, env_path=env_path)
  canvas_course = canvas_interface.get_course(course_id=course_id)
  assignment_group = canvas_course.create_assignment_group(
    name=assignment_group_name,
    delete_existing=delete_assignment_group
  )

  selected.sort(key=lambda item: item[1].name.lower())
  for registered_name, question, question_tags in selected:
    title = f"(Ungraded Practice) {question.name}"
    tag_text = ", ".join(sorted(question_tags))
    quiz = Quiz(
      name=title,
      questions=[question],
      practice=True,
      description=f"Auto-generated practice quiz for '{registered_name}'. Tags: {tag_text}"
    )
    upload_quiz_to_canvas(
      canvas_course,
      quiz,
      num_variations,
      title=title,
      is_practice=True,
      assignment_group=assignment_group,
      max_backoff_attempts=max_backoff_attempts,
      quiet=quiet
    )

  if skipped:
    log.info(
      f"Skipped {len(skipped)} question type(s) that could not be instantiated for practice generation."
    )
    for name, reason in skipped[:10]:
      log.info(f"  - {name}: {reason}")
    if len(skipped) > 10:
      log.info(f"  ... and {len(skipped) - 10} more.")

  log.info(
    f"Generated {len(selected)} practice quizzes matching tags {sorted(requested_tags)} "
    f"({'all' if match_all else 'any'} mode)."
  )


def generate_latex(latex_text, remove_previous=False, name_prefix=None) -> bool:
  """
  Generate PDF from LaTeX source code.

  Args:
    latex_text: The LaTeX source code to compile
    remove_previous: Whether to remove the 'out' directory before generating
    name_prefix: Optional prefix for the temporary filename (e.g., quiz name)
  """
  if remove_previous:
    if os.path.exists('out'): shutil.rmtree('out')

  prefix = f"{sanitize_filename(name_prefix)}-" if name_prefix else "tmp"
  tmp_tex = tempfile.NamedTemporaryFile('w', prefix=prefix)

  tmp_tex.write(latex_text)

  tmp_tex.flush()
  os.makedirs(os.path.join("out", "debug"), exist_ok=True)
  debug_name = f"debug-{datetime.now().strftime('%Y%m%d-%H%M%S')}.tex"
  shutil.copy(f"{tmp_tex.name}", os.path.join("out", "debug", debug_name))
  try:
    result = subprocess.run(
      f"latexmk -pdf -shell-escape -output-directory={os.path.join(os.getcwd(), 'out')} {tmp_tex.name}",
      shell=True,
      capture_output=True,
      timeout=30,
      check=False
    )
  except subprocess.TimeoutExpired:
    logging.error("Latex Compile timed out")
    tmp_tex.close()
    return False

  cleanup_result = subprocess.run(
    f"latexmk -c {tmp_tex.name} -output-directory={os.path.join(os.getcwd(), 'out')}",
    shell=True,
    capture_output=True,
    timeout=30,
    check=False
  )
  tmp_tex.close()
  if result.returncode != 0:
    stderr_text = result.stderr.decode("utf-8", errors="ignore")
    log.error(f"Latex compilation failed: {stderr_text}")
    return False
  if cleanup_result.returncode != 0:
    stderr_text = cleanup_result.stderr.decode("utf-8", errors="ignore")
    log.warning(f"Latex cleanup failed: {stderr_text}")
  return True


def sanitize_filename(name):
  """
  Sanitize a quiz name for use as a filename prefix.

  Converts spaces to underscores, removes special characters,
  and limits length to avoid overly long filenames.

  Example: "CST 334 Exam 4 (Fall 25)" -> "CST_334_Exam_4_Fall_25"
  """
  # Replace spaces with underscores
  sanitized = name.replace(' ', '_')

  # Remove characters that aren't alphanumeric, underscore, or hyphen
  sanitized = re.sub(r'[^\w\-]', '', sanitized)

  # Limit length to avoid overly long filenames (keep first 50 chars)
  if len(sanitized) > 50:
    sanitized = sanitized[:50]

  return sanitized


def _generate_short_id(length: int = 12) -> str:
  return uuid.uuid4().hex[:length]


def _iter_question_entries(exam_dict: dict):
  questions = exam_dict.get("questions")
  if isinstance(questions, list):
    for entry in questions:
      if isinstance(entry, dict):
        yield entry
    return
  if not isinstance(questions, dict):
    return
  for question_definitions in questions.values():
    if not isinstance(question_definitions, dict):
      continue
    for q_name, q_data in question_definitions.items():
      if q_name == "_config":
        continue
      if not isinstance(q_data, dict):
        continue
      group_config = q_data.get("_config", {}) or {}
      if group_config.get("group", False):
        for group_name, group_data in q_data.items():
          if group_name == "_config":
            continue
          if isinstance(group_data, dict):
            yield group_data
      else:
        yield q_data


def _annotate_exam_dicts_for_replay(exam_dicts: list[dict]) -> None:
  for exam_dict in exam_dicts:
    if not isinstance(exam_dict, dict):
      continue
    if not exam_dict.get("yaml_id"):
      exam_dict["yaml_id"] = _generate_short_id(8)

    seen_ids: set[str] = set()
    for entry in _iter_question_entries(exam_dict):
      existing = entry.get("question_id")
      if existing:
        entry["question_id"] = str(existing)
        seen_ids.add(entry["question_id"])

    for entry in _iter_question_entries(exam_dict):
      if entry.get("question_id"):
        continue
      new_id = _generate_short_id(12)
      while new_id in seen_ids:
        new_id = _generate_short_id(12)
      entry["question_id"] = new_id
      seen_ids.add(new_id)


def _replay_yaml_path(path_to_quiz_yaml: str) -> str:
  base = os.path.splitext(os.path.basename(path_to_quiz_yaml))[0]
  sanitized = sanitize_filename(base) or "quiz"
  return os.path.join("out", f"{sanitized}_replay.yaml")


def _collect_recent_pdfs(start_time: float, *, out_dir: str = "out") -> list[str]:
  if not os.path.isdir(out_dir):
    return []
  pdfs: list[str] = []
  for name in os.listdir(out_dir):
    if not name.lower().endswith(".pdf"):
      continue
    path = os.path.join(out_dir, name)
    try:
      if os.path.getmtime(path) >= start_time - 1:
        pdfs.append(path)
    except OSError:
      continue
  return sorted(pdfs)


def _bundle_outputs(
  replay_path: str | None,
  *,
  bundle_label: str,
  pdf_paths: list[str] | None = None
) -> str | None:
  out_dir = "out"
  if not os.path.isdir(out_dir):
    return None
  bundle_name = f"{sanitize_filename(bundle_label)}_bundle.zip"
  bundle_path = os.path.join(out_dir, bundle_name)
  pdfs = pdf_paths or []
  if not pdfs and not replay_path:
    return None
  with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
    for pdf_path in pdfs:
      if os.path.exists(pdf_path):
        bundle.write(pdf_path, arcname=os.path.basename(pdf_path))
    if replay_path and os.path.exists(replay_path):
      bundle.write(replay_path, arcname=os.path.basename(replay_path))
  return bundle_path


def _canvas_payload_fingerprint(payload):
  fingerprint = _normalize_canvas_html(payload.get("question_text", ""))
  answers = payload.get("answers", [])
  try:
    fingerprint += "".join(
      "|".join(f"{k}:{a[k]}" for k in sorted(a.keys()))
      for a in answers
    )
  except TypeError:
    log.warning("Failed to fingerprint Canvas answers; falling back to question text only.")
  return fingerprint


def _normalize_canvas_html(html: str) -> str:
  """
  Normalize HTML for Canvas dedupe.

  Strips volatile URLs (e.g., uploaded image file IDs) while preserving
  stable attributes like data-quizgen-hash for image identity.
  """
  if not html:
    return ""
  # Only strip src on <img> tags that include data-quizgen-hash.
  def _normalize_img_tag(match: re.Match) -> str:
    tag = match.group(0)
    if "data-quizgen-hash" not in tag:
      return tag
    return re.sub(r'\s+src=(".*?"|\'.*?\')', r' src="__URL__"', tag)

  normalized = re.sub(r"<img\b[^>]*>", _normalize_img_tag, html)
  # Collapse whitespace for stability across rendering differences.
  normalized = re.sub(r"\s+", " ", normalized).strip()
  return normalized


def _build_canvas_payloads(
    question,
    course,
    canvas_quiz,
    num_variations,
    seed_base,
    max_attempts=1000,
    max_backoff_attempts=None,
    show_progress_bar: bool = False,
    progress_label: str | None = None,
    generation_callback=None
):
  max_variations = num_variations
  possible_variations = getattr(question, "possible_variations", None)
  try:
    if possible_variations is not None:
      max_variations = min(max_variations, int(possible_variations))
  except Exception:
    pass

  payloads = []
  seen = set()
  progress = None
  if show_progress_bar:
    try:
      from tqdm import tqdm
    except Exception as exc:
      log.warning(f"Progress bar requested but tqdm is unavailable: {exc}")
    else:
      desc = progress_label or f"Preparing {question.name}"
      progress = tqdm(total=max_variations, desc=desc, unit="var", leave=True)
  for attempt in range(max_attempts):
    rng_seed = seed_base + (attempt * 1000)
    payload = question.get__canvas(
      course,
      canvas_quiz,
      rng_seed=rng_seed,
      max_backoff_attempts=max_backoff_attempts
    )
    fingerprint = _canvas_payload_fingerprint(payload)
    if fingerprint in seen:
      continue
    seen.add(fingerprint)
    payloads.append(payload)
    if generation_callback is not None:
      generation_callback({"event": "success"})
    if progress is not None:
      progress.update(1)
    if len(payloads) >= max_variations:
      break
  if generation_callback is not None:
    generation_callback({
      "event": "complete",
      "generated": len(payloads),
      "expected": max_variations
    })
  if progress is not None:
    progress.close()

  if len(payloads) < max_variations:
    log.warning(
      f"Completed question '{question.name}': {len(payloads)}/{max_variations} variations "
      f"after {max_attempts} attempts (dedup collisions or limited variation space)."
    )
  return payloads


def upload_quiz_to_canvas(
    canvas_course,
    quiz,
    num_variations,
    *,
    title=None,
    is_practice=False,
    assignment_group=None,
    optimize_layout=False,
    max_backoff_attempts=None,
    quiet: bool = False
):
  if assignment_group is None:
    assignment_group = canvas_course.create_assignment_group()

  canvas_quiz = canvas_course.add_quiz(
    assignment_group,
    title,
    is_practice=is_practice,
    description=quiz.description
  )

  ordered_questions = quiz.get_ordered_questions(optimize_layout=optimize_layout)
  total_questions = len(ordered_questions)
  log.info(f"Starting to push quiz '{title or canvas_quiz.title}' with {total_questions} questions to Canvas")
  log.info(f"Target: {num_variations} variations per question")
  show_progress_bar = not quiet
  overall_bar = None
  if show_progress_bar:
    try:
      from tqdm import tqdm
    except Exception as exc:
      log.warning(f"Progress bar requested but tqdm is unavailable: {exc}")
    else:
      overall_total_expected = 0
      for question_i, question in enumerate(ordered_questions):
        seed_base = question_i * 100_000
        if isinstance(question, QuestionGroup):
          if question.pick_once:
            selected = question._select_questions(seed_base)
          else:
            selected = list(question.questions)
          expected = num_variations * len(selected)
        else:
          expected = num_variations
        overall_total_expected += expected * 2

      overall_bar = tqdm(
        total=overall_total_expected,
        desc="Overall (gen+upload)",
        unit="var",
        leave=True
      )

  def make_overall_callback(expected_total: int):
    if overall_bar is None:
      return None
    completed_local = 0

    def _callback(event):
      nonlocal completed_local
      event_type = event.get("event")
      if event_type in {"success", "failed"}:
        overall_bar.update(1)
        completed_local += 1
      if event_type == "complete":
        expected = event.get("expected", expected_total)
        if expected > completed_local:
          overall_bar.update(expected - completed_local)
          completed_local = expected

    return _callback

  for question_i, question in enumerate(ordered_questions):
    label = f"[{question_i + 1}/{total_questions}] {question.name}"
    log.info(f"Preparing question {label}")
    seed_base = question_i * 100_000

    if isinstance(question, QuestionGroup):
      if question.pick_once:
        selected = question._select_questions(seed_base)
      else:
        selected = list(question.questions)

      expected = num_variations * len(selected)
      generation_callback = make_overall_callback(expected)
      upload_callback = make_overall_callback(expected)

      group_payloads = []
      for idx, sub_question in enumerate(selected):
        sub_seed_base = seed_base + (idx * 10_000)
        group_payloads.extend(_build_canvas_payloads(
            sub_question,
            canvas_course.course,
            canvas_quiz,
            num_variations,
            sub_seed_base,
            max_backoff_attempts=max_backoff_attempts,
            show_progress_bar=show_progress_bar,
            progress_label=f"Preparing {label}",
            generation_callback=generation_callback
        ))

      log.info(f"Uploading group {label} with {len(group_payloads)} variations")
      canvas_course.create_question(
        canvas_quiz,
        group_payloads,
        group_name=question.name,
        question_points=question.points_value,
        pick_count=question.num_to_pick,
        show_progress_bar=show_progress_bar,
        progress_label=f"Uploading {label}",
        progress_callback=upload_callback
      )
      continue

    expected = num_variations
    generation_callback = make_overall_callback(expected)
    upload_callback = make_overall_callback(expected)

    payloads = _build_canvas_payloads(
      question,
      canvas_course.course,
      canvas_quiz,
      num_variations,
      seed_base,
      max_backoff_attempts=max_backoff_attempts,
      show_progress_bar=show_progress_bar,
      progress_label=f"Preparing {label}",
      generation_callback=generation_callback
    )
    log.info(f"Uploading question {label} with {len(payloads)} variations")
    canvas_course.create_question(
      canvas_quiz,
      payloads,
      group_name=question.name,
      question_points=question.points_value,
      pick_count=1,
      show_progress_bar=show_progress_bar,
      progress_label=f"Uploading {label}",
      progress_callback=upload_callback
    )

  if overall_bar is not None:
    overall_bar.close()
  log.info(f"Canvas quiz URL: {canvas_quiz.html_url}")
  return canvas_quiz


def generate_typst(typst_text, remove_previous=False, name_prefix=None) -> bool:
  """
  Generate PDF from Typst source code.

  Similar to generate_latex, but uses typst compiler instead of latexmk.

  Args:
    typst_text: The Typst source code to compile
    remove_previous: Whether to remove the 'out' directory before generating
    name_prefix: Optional prefix for the temporary filename (e.g., quiz name)
  """
  if remove_previous:
    if os.path.exists('out'):
      shutil.rmtree('out')

  # Ensure output directory exists
  os.makedirs('out', exist_ok=True)

  # Create temporary Typst file with optional name prefix
  prefix = f"{sanitize_filename(name_prefix)}-" if name_prefix else "tmp"
  tmp_typ = tempfile.NamedTemporaryFile('w', suffix='.typ', delete=False, prefix=prefix)

  try:
    tmp_typ.write(typst_text)
    tmp_typ.flush()
    tmp_typ.close()

    # Save debug copy
    os.makedirs(os.path.join("out", "debug"), exist_ok=True)
    debug_name = f"debug-{datetime.now().strftime('%Y%m%d-%H%M%S')}.typ"
    shutil.copy(tmp_typ.name, os.path.join("out", "debug", debug_name))

    # Compile with typst
    output_pdf = os.path.join(os.getcwd(), 'out', os.path.basename(tmp_typ.name).replace('.typ', '.pdf'))
    
    # Use --root to set the filesystem root so absolute paths work correctly
    p = subprocess.Popen(
      ['typst', 'compile', '--root', '/', tmp_typ.name, output_pdf],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE
    )

    try:
      stdout, stderr = p.communicate(timeout=30)
      if p.returncode != 0:
        stderr_text = stderr.decode("utf-8", errors="ignore")
        log.error(f"Typst compilation failed: {stderr_text}")
        return False
    except subprocess.TimeoutExpired:
      log.error("Typst compile timed out")
      p.kill()
      p.communicate()
      return False

  finally:
    # Clean up temp file
    if os.path.exists(tmp_typ.name):
      os.unlink(tmp_typ.name)
  return True


def generate_quiz(
    path_to_quiz_yaml,
    num_pdfs=0,
    num_canvas=0,
    use_prod=False,
    course_id=None,
    delete_assignment_group=False,
    use_typst=False,
    use_typst_measurement=False,
    base_seed=None,
    env_path=None,
    consistent_pages=False,
    layout_samples=10,
    layout_safety_factor=1.1,
    embed_images_typst=True,
    optimize_layout=False,
    max_backoff_attempts=None,
    quiet: bool = False
):

  start_time = time.time()
  with open(path_to_quiz_yaml) as fid:
    raw_exam_dicts = list(yaml.safe_load_all(fid))

  replay_exam_dicts = None
  if num_pdfs > 0:
    replay_exam_dicts = copy.deepcopy(raw_exam_dicts)
    _annotate_exam_dicts_for_replay(replay_exam_dicts)
    exam_dicts_for_parsing = copy.deepcopy(replay_exam_dicts)
  else:
    exam_dicts_for_parsing = raw_exam_dicts

  quizzes = Quiz.from_exam_dicts(exam_dicts_for_parsing, source_path=path_to_quiz_yaml)

  # Handle Canvas uploads with shared assignment group
  if num_canvas > 0:
    canvas_interface = CanvasInterface(prod=use_prod, env_path=env_path)
    canvas_course = canvas_interface.get_course(course_id=course_id)

    # Create assignment group once, with delete flag if specified
    assignment_group = canvas_course.create_assignment_group(
      name="dev",
      delete_existing=delete_assignment_group
    )

    log.info(f"Using assignment group '{assignment_group.name}' for all quizzes")

  for quiz in quizzes:

    for i in range(num_pdfs):
      log.debug(f"Generating PDF {i+1}/{num_pdfs}")
      # If base_seed is provided, use it with an offset for each PDF
      # Otherwise generate a random seed for this PDF
      if base_seed is not None:
        pdf_seed = base_seed + (i * 1000)  # Large gap to avoid overlap with rng_seed_offset
      else:
        pdf_seed = random.randint(0, 1_000_000)

      log.info(f"Generating PDF {i+1} with seed: {pdf_seed}")

      if use_typst:
        # Generate using Typst
        quiz_kwargs = {
          "rng_seed": pdf_seed,
          "use_typst_measurement": use_typst_measurement,
          "consistent_pages": consistent_pages,
        }
        if max_backoff_attempts is not None:
          quiz_kwargs["max_backoff_attempts"] = max_backoff_attempts
        if consistent_pages:
          quiz_kwargs["layout_samples"] = layout_samples
          quiz_kwargs["layout_safety_factor"] = layout_safety_factor
        typst_text = quiz.get_quiz(**quiz_kwargs, optimize_layout=optimize_layout).render(
          "typst",
          embed_images_typst=embed_images_typst
        )
        if not generate_typst(typst_text, remove_previous=(i==0), name_prefix=quiz.name):
          raise QuizGenError("PDF generation failed (Typst).")
      else:
        # Generate using LaTeX (default)
        quiz_kwargs = {
          "rng_seed": pdf_seed,
          "use_typst_measurement": use_typst_measurement,
          "consistent_pages": consistent_pages,
        }
        if max_backoff_attempts is not None:
          quiz_kwargs["max_backoff_attempts"] = max_backoff_attempts
        if consistent_pages:
          quiz_kwargs["layout_samples"] = layout_samples
          quiz_kwargs["layout_safety_factor"] = layout_safety_factor
        latex_text = quiz.get_quiz(**quiz_kwargs, optimize_layout=optimize_layout).render_latex()
        if not generate_latex(latex_text, remove_previous=(i==0), name_prefix=quiz.name):
          raise QuizGenError("PDF generation failed (LaTeX).")

    if num_canvas > 0:
      upload_quiz_to_canvas(
        canvas_course,
        quiz,
        num_canvas,
        title=quiz.name,
        is_practice=quiz.practice,
        assignment_group=assignment_group,
        optimize_layout=optimize_layout,
        max_backoff_attempts=max_backoff_attempts,
        quiet=quiet
      )
    
    quiz.describe()

  if replay_exam_dicts is not None:
    os.makedirs('out', exist_ok=True)
    replay_path = _replay_yaml_path(path_to_quiz_yaml)
    with open(replay_path, "w", encoding="utf-8") as handle:
      yaml.safe_dump_all(replay_exam_dicts, handle, sort_keys=False)
    log.info(f"Wrote replay YAML to {replay_path}")
    pdf_paths = _collect_recent_pdfs(start_time)
    bundle_path = _bundle_outputs(
      replay_path,
      bundle_label=os.path.splitext(os.path.basename(path_to_quiz_yaml))[0],
      pdf_paths=pdf_paths
    )
    if bundle_path:
      log.info(f"Wrote output bundle to {bundle_path}")

def main():

  args = parse_args()
  
  # Load environment variables
  load_dotenv(args.env)
  
  if args.debug:
    # Set root logger to DEBUG
    logging.getLogger().setLevel(logging.DEBUG)

    # Set all handlers to DEBUG level
    for handler in logging.getLogger().handlers:
      handler.setLevel(logging.DEBUG)

    # Set named loggers to DEBUG
    for logger_name in ['QuizGenerator', 'lms_interface', '__main__']:
      logger = logging.getLogger(logger_name)
      logger.setLevel(logging.DEBUG)
      for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)

  if args.command == "TEST":
    test()
    return

  # Dependency checks
  require_typst = bool(getattr(args, "typst", True))
  require_latex = not require_typst
  if args.num_pdfs > 0 or args.test_all > 0 or args.check_deps:
    ok, missing = _check_dependencies(
      require_typst=require_typst,
      require_latex=require_latex
    )
    if not ok:
      message = "\n".join(missing)
      raise QuizGenError(message)
    if args.check_deps:
      print("Dependency check passed.")
      return

  if args.allow_generator:
    os.environ["QUIZGEN_ALLOW_GENERATOR"] = "1"

  if args.float_tolerance is not None:
    if args.float_tolerance < 0:
      raise QuizGenError("float_tolerance must be non-negative.")
    Answer.DEFAULT_FLOAT_TOLERANCE = args.float_tolerance

  if args.max_backoff_attempts is not None:
    if args.max_backoff_attempts < 1:
      raise QuizGenError("max_backoff_attempts must be >= 1.")

  if args.num_pdfs and args.num_pdfs > 1:
    args.consistent_pages = True

  if args.test_all > 0:
    # Set up Canvas course if course_id provided
    canvas_course = None
    if args.course_id:
      canvas_interface = CanvasInterface(prod=args.prod, env_path=args.env)
      canvas_course = canvas_interface.get_course(course_id=args.course_id)

    success = test_all_questions(
      args.test_all,
      generate_pdf=True,
      use_typst=getattr(args, 'typst', True),
      canvas_course=canvas_course,
      strict=args.strict,
      question_filter=args.test_questions,
      skip_missing_extras=args.skip_missing_extras,
      embed_images_typst=getattr(args, "embed_images_typst", False)
    )
    if not success:
      raise QuizGenError("One or more questions failed during --test_all.")
    return

  if args.generate_practice:
    generate_practice_quizzes(
      tag_filters=args.generate_practice,
      course_id=args.course_id,
      use_prod=args.prod,
      env_path=args.env,
      num_variations=args.num_canvas if args.num_canvas > 0 else args.practice_variations,
      points_value=args.practice_points,
      delete_assignment_group=getattr(args, 'delete_assignment_group', False),
      assignment_group_name=args.practice_assignment_group,
      match_all=(args.practice_match == "all"),
      quiet=getattr(args, "quiet", False),
      max_backoff_attempts=getattr(args, "max_backoff_attempts", None)
    )
    return

  # Clear any previous metrics
  PerformanceTracker.clear_metrics()

  generate_quiz(
    args.quiz_yaml,
    num_pdfs=args.num_pdfs,
    num_canvas=args.num_canvas,
    use_prod=args.prod,
    course_id=args.course_id,
    delete_assignment_group=getattr(args, 'delete_assignment_group', False),
    use_typst=getattr(args, 'typst', True),
    use_typst_measurement=getattr(args, 'typst_measurement', False),
    base_seed=getattr(args, 'seed', None),
    env_path=args.env,
    consistent_pages=getattr(args, 'consistent_pages', False),
    layout_samples=getattr(args, 'layout_samples', 10),
    layout_safety_factor=getattr(args, 'layout_safety_factor', 1.1),
    embed_images_typst=getattr(args, 'embed_images_typst', True),
    optimize_layout=getattr(args, 'optimize_space', False),
    max_backoff_attempts=getattr(args, "max_backoff_attempts", None),
    quiet=getattr(args, "quiet", False)
  )


if __name__ == "__main__":
  try:
    main()
  except QuizGenError as exc:
    log.error(str(exc))
    exit(1)
