#!env python
import argparse
import os
import shutil
import subprocess
import tempfile
from lms_interface.canvas_interface import CanvasInterface, CanvasCourse

from QuizGenerator.quiz import Quiz


import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from QuizGenerator.performance import PerformanceTracker


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--prod", action="store_true")
  parser.add_argument("--course_id", type=int)

  parser.add_argument("--quiz_yaml", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_files/exam_generation.yaml"))
  parser.add_argument("--num_canvas", default=0, type=int)
  parser.add_argument("--num_pdfs", default=0, type=int)
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

  # Load the default quiz configuration
  quiz_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_files/scratch.yaml")

  # Generate a quiz
  quizzes = Quiz.from_yaml(quiz_yaml)
  quiz = quizzes[0]  # Take the first quiz
  quiz_document = quiz.get_quiz(rng_seed=1)

  # Get the first question from the quiz
  if not quiz_document.elements:
    print("No questions generated!")
    return

  question = quiz_document.elements[0]

  print("="*60)
  print("QUESTION TEST OUTPUT")
  print("="*60)

  print("\n" + "="*20 + " HTML (Canvas) Output " + "="*20)
  html_output = question.body.render("html")
  print(html_output)

  print("\n" + "="*20 + " LaTeX (PDF) Output " + "="*20)
  latex_output = question.body.render("latex")
  print(latex_output)

  print("\n" + "="*20 + " Markdown Output " + "="*20)
  markdown_output = question.body.render("markdown")
  print(markdown_output)

  print("\n" + "="*20 + " Explanation Output " + "="*20)
  explanation_output = question.explanation.render("html")
  print(explanation_output)

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

def generate_quiz(
    path_to_quiz_yaml,
    num_pdfs=0,
    num_canvas=0,
    use_prod=False,
    course_id=None,
    delete_assignment_group=False
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
      # Use a different seed for each PDF to ensure different workloads across PDFs
      # but consistent workloads within the same PDF
      pdf_seed = i * 1000  # Large gap to avoid overlap with rng_seed_offset
      latex_text = quiz.get_quiz(rng_seed=pdf_seed).render_latex()
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
    delete_assignment_group=getattr(args, 'delete_assignment_group', False)
  )


if __name__ == "__main__":
  main()