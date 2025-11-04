#!env python
import argparse
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from lms_interface.canvas_interface import CanvasInterface, CanvasCourse

from QuizGenerator.quiz import Quiz

# Load environment variables from ~/.env
load_dotenv(Path.home() / '.env')

import logging
log = logging.getLogger(__name__)

from QuizGenerator.performance import PerformanceTracker


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--prod", action="store_true")
  parser.add_argument("--course_id", type=int)

  parser.add_argument("--quiz_yaml", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_files/exam_generation.yaml"))
  parser.add_argument("--num_canvas", default=0, type=int)
  parser.add_argument("--num_pdfs", default=0, type=int)
  parser.add_argument("--seed", type=int, default=None,
                     help="Random seed for quiz generation (default: None for random)")
  parser.add_argument("--typst", action="store_true",
                     help="Use Typst instead of LaTeX for PDF generation")
  parser.add_argument("--typst-measurement", action="store_true",
                     help="Use Typst to measure question heights for optimal bin-packing (requires Typst)")
  parser.add_argument("--delete-assignment-group", action="store_true",
                     help="Delete existing assignment group before uploading new quizzes")

  subparsers = parser.add_subparsers(dest='command')
  test_parser = subparsers.add_parser("TEST")


  args = parser.parse_args()

  if args.num_canvas > 0 and args.course_id is None:
    log.error("Must provide course_id when pushing to canvas")
    exit(8)

  return args


def test():
  log.info("Running test...")

  # Load the CST463 quiz configuration to test vector questions
  quiz_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_files/cst463.yaml")

  # Generate a quiz
  quizzes = Quiz.from_yaml(quiz_yaml)

  # Find a vector question to test
  from QuizGenerator.question import QuestionRegistry

  print("="*60)
  print("CANVAS ANSWER DUPLICATION TEST")
  print("="*60)

  # Test multiple question types to find which ones use VECTOR variable kind
  question_types = ["VectorAddition", "VectorDotProduct", "VectorMagnitude", "DerivativeBasic", "DerivativeChain"]

  for question_type in question_types:
    print(f"\n" + "="*20 + f" TESTING {question_type} " + "="*20)
    question = QuestionRegistry.create(question_type, name=f"Test {question_type}", points_value=1)

    print(f"Question answers: {len(question.answers)} answer objects")

    # Show all individual answers and their Canvas representations
    for key, answer in question.answers.items():
      canvas_answers = answer.get_for_canvas()
      print(f"\nAnswer key '{key}':")
      print(f"  Variable kind: {answer.variable_kind}")
      print(f"  Value: {answer.value}")
      print(f"  Canvas entries: {len(canvas_answers)}")
      for i, canvas_answer in enumerate(canvas_answers):
        print(f"    {i+1}: '{canvas_answer['answer_text']}'")

      # Show duplicates if they exist
      texts = [ca['answer_text'] for ca in canvas_answers]
      unique_texts = set(texts)
      duplicates = len(texts) - len(unique_texts)
      if duplicates > 0:
        print(f"    *** {duplicates} DUPLICATE ENTRIES FOUND ***")
        for text in sorted(unique_texts):
          count = texts.count(text)
          if count > 1:
            print(f"      '{text}' appears {count} times")

  # Check for phantom blank_ids that aren't displayed
  print(f"\n" + "="*20 + " PHANTOM BLANK_ID CHECK " + "="*20)

  for question_type in ["DerivativeBasic", "DerivativeChain"]:
    print(f"\n--- {question_type} ---")
    question = QuestionRegistry.create(question_type, name=f"Test {question_type}", points_value=1)

    # Get the question body to see what blank_ids are actually displayed
    body = question.get_body()
    body_html = body.render("html")

    print(f"  HTML body preview:")
    print(f"    {body_html[:200]}...")

    # Extract blank_ids from the HTML using regex
    import re
    displayed_blank_ids = set(re.findall(r'name="([^"]*)"', body_html))

    # Get all blank_ids from the answers
    question_type_enum, canvas_answers = question.get_answers()
    all_blank_ids = set(answer['blank_id'] for answer in canvas_answers)

    print(f"  Total answers in self.answers: {len(question.answers)}")
    print(f"  Total Canvas answer entries: {len(canvas_answers)}")
    print(f"  Unique blank_ids in Canvas answers: {len(all_blank_ids)}")
    print(f"  Blank_ids displayed in HTML: {len(displayed_blank_ids)}")

    # Find phantom blank_ids
    phantom_blank_ids = all_blank_ids - displayed_blank_ids
    if phantom_blank_ids:
      print(f"  *** PHANTOM BLANK_IDS FOUND: {phantom_blank_ids} ***")
      for phantom_id in phantom_blank_ids:
        phantom_answers = [a for a in canvas_answers if a['blank_id'] == phantom_id]
        print(f"    '{phantom_id}': {len(phantom_answers)} entries not displayed")
    else:
      print(f"  No phantom blank_ids found")

    # Show what blank_ids are actually displayed
    print(f"  Displayed blank_ids: {sorted(displayed_blank_ids)}")
    print(f"  All answer blank_ids: {sorted(all_blank_ids)}")

  # Now create a synthetic test to demonstrate the VECTOR bug
  print(f"\n" + "="*20 + " SYNTHETIC VECTOR TEST " + "="*20)
  from QuizGenerator.misc import Answer

  # Create a synthetic answer with VECTOR variable kind to demonstrate the bug
  vector_answer = Answer(
    key="test_vector",
    value=[1, 2, 3],  # 3D vector
    variable_kind=Answer.VariableKind.VECTOR
  )

  canvas_answers = vector_answer.get_for_canvas()
  print(f"Synthetic VECTOR answer:")
  print(f"  Value: {vector_answer.value}")
  print(f"  Canvas entries: {len(canvas_answers)}")
  # Only show first few to save space
  for i, canvas_answer in enumerate(canvas_answers[:5]):
    print(f"    {i+1}: '{canvas_answer['answer_text']}'")
  if len(canvas_answers) > 5:
    print(f"    ... and {len(canvas_answers) - 5} more entries")

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

  # Generate performance report if Canvas questions were generated
  if num_canvas > 0:
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS REPORT")
    print("="*80)
    PerformanceTracker.report_summary(min_duration=0.01)  # Show operations taking >10ms

    # Show detailed breakdown for slowest operations
    print("\n" + "="*60)
    print("DETAILED TIMING BREAKDOWN")
    print("="*60)

    slow_operations = ["canvas_prepare_question", "canvas_api_upload", "ast_render_body", "question_body"]
    for op in slow_operations:
      metrics = PerformanceTracker.get_metrics_by_operation(op)
      if metrics:
        print(f"\n{op.upper()}:")
        # Show stats by question type
        by_type = {}
        for m in metrics:
          qtype = m.question_type or "unknown"
          if qtype not in by_type:
            by_type[qtype] = []
          by_type[qtype].append(m.duration)

        for qtype, durations in by_type.items():
          avg = sum(durations) / len(durations)
          print(f"  {qtype}: {len(durations)} calls, avg {avg:.3f}s (range: {min(durations):.3f}s - {max(durations):.3f}s)")

def main():

  args = parse_args()

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