#!env python
import argparse
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from QuizGenerator.canvas.canvas_interface import CanvasInterface

from QuizGenerator.quiz import Quiz

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
  parser.add_argument("--typst", action="store_true",
                     help="Use Typst instead of LaTeX for PDF generation")

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
  
  
def generate_latex(latex_text, remove_previous=False):

  if remove_previous:
    if os.path.exists('out'): shutil.rmtree('out')

  tmp_tex = tempfile.NamedTemporaryFile('w')

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


def generate_typst(typst_text, remove_previous=False):
  """
  Generate PDF from Typst source code.

  Similar to generate_latex, but uses typst compiler instead of latexmk.
  """
  if remove_previous:
    if os.path.exists('out'):
      shutil.rmtree('out')

  # Ensure output directory exists
  os.makedirs('out', exist_ok=True)

  # Create temporary Typst file
  tmp_typ = tempfile.NamedTemporaryFile('w', suffix='.typ', delete=False)

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
        generate_typst(typst_text, remove_previous=(i==0))
      else:
        # Generate using LaTeX (default)
        latex_text = quiz.get_quiz(rng_seed=pdf_seed, use_typst_measurement=use_typst_measurement).render_latex()
        generate_latex(latex_text, remove_previous=(i==0))

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
