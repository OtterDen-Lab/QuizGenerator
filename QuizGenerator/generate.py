#!env python
import argparse
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

from lms_interface.canvas_interface import CanvasInterface
from QuizGenerator.performance import PerformanceTracker
from QuizGenerator.question import QuestionGroup, QuestionRegistry
from QuizGenerator.quiz import Quiz

log = logging.getLogger(__name__)


class QuizGenError(Exception):
  """User-facing error for CLI operations."""



def parse_args():
  parser = argparse.ArgumentParser()

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

  # Canvas flags
  parser.add_argument("--num_canvas", default=0, type=int, help="How many variations of each question to try to upload to canvas.")
  parser.add_argument("--prod", action="store_true")
  parser.add_argument("--course_id", type=int)
  parser.add_argument("--delete-assignment-group", action="store_true",
                     help="Delete existing assignment group before uploading new quizzes")
  
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

  subparsers = parser.add_subparsers(dest='command')
  test_parser = subparsers.add_parser("TEST")


  args = parser.parse_args()

  if args.num_canvas > 0 and args.course_id is None:
    parser.error("Missing --course_id for Canvas upload. Example: --course_id 12345")

  if args.test_all <= 0 and not args.quiz_yaml:
    parser.error("Missing --yaml. Example: quizgen --yaml example_files/example_exam.yaml --num_pdfs 1")

  return args


def _check_dependencies(*, require_typst: bool, require_latex: bool) -> Tuple[bool, list[str]]:
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


def _canvas_payload_fingerprint(payload):
  fingerprint = payload.get("question_text", "")
  answers = payload.get("answers", [])
  try:
    fingerprint += "".join(
      "|".join(f"{k}:{a[k]}" for k in sorted(a.keys()))
      for a in answers
    )
  except TypeError:
    log.warning("Failed to fingerprint Canvas answers; falling back to question text only.")
  return fingerprint


def _build_canvas_payloads(
    question,
    course,
    canvas_quiz,
    num_variations,
    seed_base,
    max_attempts=1000
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
  for attempt in range(max_attempts):
    rng_seed = seed_base + (attempt * 1000)
    payload = question.get__canvas(course, canvas_quiz, rng_seed=rng_seed)
    fingerprint = _canvas_payload_fingerprint(payload)
    if fingerprint in seen:
      continue
    seen.add(fingerprint)
    payloads.append(payload)
    if len(payloads) >= max_variations:
      break

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
    optimize_layout=False
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

  for question_i, question in enumerate(ordered_questions):
    log.info(f"Uploading question {question_i + 1}/{total_questions}: '{question.name}'")
    seed_base = question_i * 100_000

    if isinstance(question, QuestionGroup):
      if question.pick_once:
        selected = question._select_questions(seed_base)
      else:
        selected = list(question.questions)

      group_payloads = []
      for idx, sub_question in enumerate(selected):
        sub_seed_base = seed_base + (idx * 10_000)
        group_payloads.extend(_build_canvas_payloads(
          sub_question,
          canvas_course.course,
          canvas_quiz,
          num_variations,
          sub_seed_base
        ))

      canvas_course.create_question(
        canvas_quiz,
        group_payloads,
        group_name=question.name,
        question_points=question.points_value,
        pick_count=question.num_to_pick
      )
      continue

    payloads = _build_canvas_payloads(
      question,
      canvas_course.course,
      canvas_quiz,
      num_variations,
      seed_base
    )
    canvas_course.create_question(
      canvas_quiz,
      payloads,
      group_name=question.name,
      question_points=question.points_value,
      pick_count=1
    )

  log.info(f"Canvas quiz URL: {canvas_quiz.html_url}")


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
    optimize_layout=False
):

  quizzes = Quiz.from_yaml(path_to_quiz_yaml)

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
        optimize_layout=optimize_layout
      )
    
    quiz.describe()

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
    optimize_layout=getattr(args, 'optimize_space', False)
  )


if __name__ == "__main__":
  try:
    main()
  except QuizGenError as exc:
    log.error(str(exc))
    exit(1)
