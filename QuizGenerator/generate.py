#!env python
import argparse
from datetime import datetime
import os
import random
import shutil
import subprocess
import tempfile
import traceback
import re
from pathlib import Path
from dotenv import load_dotenv
from QuizGenerator.canvas.canvas_interface import CanvasInterface

from QuizGenerator.quiz import Quiz
from QuizGenerator.question import QuestionRegistry

import logging
log = logging.getLogger(__name__)

from QuizGenerator.performance import PerformanceTracker


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--env",
    default=os.path.join(Path.home(), '.env'),
    help="Path to .env file specifying canvas details"
  )
  
  parser.add_argument("--debug", action="store_true", help="Set logging level to debug")

  parser.add_argument("--quiz_yaml", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_files/exam_generation.yaml"))
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
  parser.add_argument("--latex", action="store_false", dest="typst", help="Use Typst instead of LaTeX for PDF generation")

  # Testing flags
  parser.add_argument("--test_all", type=int, default=0, metavar="N",
                     help="Generate N variations of ALL registered questions to test they work correctly")
  parser.add_argument("--test_questions", nargs='+', metavar="NAME",
                     help="Only test specific question types by name (use with --test_all)")
  parser.add_argument("--strict", action="store_true",
                     help="With --test_all, skip PDF/Canvas generation if any questions fail")

  subparsers = parser.add_subparsers(dest='command')
  test_parser = subparsers.add_parser("TEST")


  args = parser.parse_args()

  if args.num_canvas > 0 and args.course_id is None:
    log.error("Must provide course_id when pushing to canvas")
    exit(8)

  return args


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
    question_filter: list = None
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
        question_ast = question.get_question(rng_seed=seed)

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

      except Exception as e:
        tb = traceback.format_exc()
        question_failures.append(f"  Variation {variation+1}: Generation failed - {e}\n{tb}")

    if question_failures:
      print(f"  FAILED ({len(question_failures)}/{num_variations} variations)")
      for failure in question_failures:
        print(failure)
      failed_questions.append((question_name, question_failures))
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

  if failed_questions:
    print("\nFailed questions:")
    for name, failures in failed_questions:
      print(f"  - {name}: {len(failures)} failures")

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
        typst_text = test_quiz.get_quiz(rng_seed=pdf_seed).render("typst")
        generate_typst(typst_text, remove_previous=True, name_prefix="test_all_questions")
      else:
        latex_text = test_quiz.get_quiz(rng_seed=pdf_seed).render_latex()
        generate_latex(latex_text, remove_previous=True, name_prefix="test_all_questions")
      print("PDF generated in out/ directory")

    if canvas_course:
      print("Pushing to Canvas...")
      quiz_title = f"Test All Questions ({int(datetime.now().timestamp())} : {datetime.now().strftime('%b %d %I:%M%p')})"
      canvas_course.push_quiz_to_canvas(
        test_quiz,
        num_variations=1,
        title=quiz_title,
        is_practice=True
      )
      print(f"Quiz '{quiz_title}' pushed to Canvas")

  return len(failed_questions) == 0


def generate_latex(latex_text, remove_previous=False, name_prefix=None):
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
  shutil.copy(f"{tmp_tex.name}", "debug.tex")
  p = subprocess.Popen(
    f"latexmk -pdf -output-directory={os.path.join(os.getcwd(), 'out')} {tmp_tex.name}",
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE)
  try:
    p.wait(30)
  except subprocess.TimeoutExpired:
    logging.error("Latex Compile timed out")
    p.kill()
    tmp_tex.close()
    return
  proc = subprocess.Popen(
    f"latexmk -c {tmp_tex.name} -output-directory={os.path.join(os.getcwd(), 'out')}",
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
  )
  proc.wait(timeout=30)
  tmp_tex.close()


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


def generate_typst(typst_text, remove_previous=False, name_prefix=None):
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
    shutil.copy(tmp_typ.name, "debug.typ")

    # Compile with typst
    output_pdf = os.path.join(os.getcwd(), 'out', os.path.basename(tmp_typ.name).replace('.typ', '.pdf'))
    
    # Use --root to set the filesystem root so absolute paths work correctly
    p = subprocess.Popen(
      ['typst', 'compile', '--root', '/', tmp_typ.name, output_pdf],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE
    )

    try:
      p.wait(30)
      if p.returncode != 0:
        stderr = p.stderr.read().decode('utf-8')
        log.error(f"Typst compilation failed: {stderr}")
    except subprocess.TimeoutExpired:
      log.error("Typst compile timed out")
      p.kill()

  finally:
    # Clean up temp file
    if os.path.exists(tmp_typ.name):
      os.unlink(tmp_typ.name)


def generate_quiz(
    path_to_quiz_yaml,
    num_pdfs=0,
    num_canvas=0,
    use_prod=False,
    course_id=None,
    delete_assignment_group=False,
    use_typst=False,
    use_typst_measurement=False,
    base_seed=None
):

  quizzes = Quiz.from_yaml(path_to_quiz_yaml)

  # Handle Canvas uploads with shared assignment group
  if num_canvas > 0:
    canvas_interface = CanvasInterface(prod=use_prod)
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
        typst_text = quiz.get_quiz(rng_seed=pdf_seed, use_typst_measurement=use_typst_measurement).render("typst")
        generate_typst(typst_text, remove_previous=(i==0), name_prefix=quiz.name)
      else:
        # Generate using LaTeX (default)
        latex_text = quiz.get_quiz(rng_seed=pdf_seed, use_typst_measurement=use_typst_measurement).render_latex()
        generate_latex(latex_text, remove_previous=(i==0), name_prefix=quiz.name)

    if num_canvas > 0:
      canvas_course.push_quiz_to_canvas(
        quiz,
        num_canvas,
        title=quiz.name,
        is_practice=quiz.practice,
        assignment_group=assignment_group
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

  if args.test_all > 0:
    # Set up Canvas course if course_id provided
    canvas_course = None
    if args.course_id:
      canvas_interface = CanvasInterface(prod=args.prod)
      canvas_course = canvas_interface.get_course(course_id=args.course_id)

    success = test_all_questions(
      args.test_all,
      generate_pdf=True,
      use_typst=getattr(args, 'typst', True),
      canvas_course=canvas_course,
      strict=args.strict,
      question_filter=args.test_questions
    )
    exit(0 if success else 1)

  # Clear any previous metrics
  PerformanceTracker.clear_metrics()

  generate_quiz(
    args.quiz_yaml,
    num_pdfs=args.num_pdfs,
    num_canvas=args.num_canvas,
    use_prod=args.prod,
    course_id=args.course_id,
    delete_assignment_group=getattr(args, 'delete_assignment_group', False),
    use_typst=getattr(args, 'typst', False),
    use_typst_measurement=getattr(args, 'typst_measurement', False),
    base_seed=getattr(args, 'seed', None)
  )


if __name__ == "__main__":
  main()
