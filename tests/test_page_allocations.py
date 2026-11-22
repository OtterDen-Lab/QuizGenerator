"""Run with python -m unittest discover -s tests -p test_page_allocations.py."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from QuizGenerator.generation.contentast import Document
from QuizGenerator.generation.quiz import Quiz


def make_quiz(**settings):
  return Quiz.from_exam_dicts([{
    "name": "Allocation test",
    "questions": [{"name": "Example", "points": 2, "class": "FromText",
                   "kwargs": {"text": "Explain your answer.", **settings}}],
  }])[0]


class PageAllocationTests(unittest.TestCase):
  def test_settings_reach_ast(self):
    quiz = make_quiz(questions_per_page=3, extra_pages=1)
    rendered = quiz.get_quiz(rng_seed=42).render("typst")
    self.assertIn("questions_per_page: 3", rendered)
    self.assertIn("extra_pages: 1", rendered)

  def test_invalid_settings(self):
    for settings in ({"questions_per_page": 0}, {"questions_per_page": True},
                     {"questions_per_page": 2.5}, {"extra_pages": -1},
                     {"questions_per_page": 2, "spacing": "SHORT"}):
      with self.subTest(settings=settings), self.assertRaises(ValueError):
        make_quiz(**settings)

  def test_input_order_even_when_optimization_requested(self):
    quiz = make_quiz()
    first = quiz.questions[0]
    second = make_quiz().questions[0]
    second.points_value = 9
    quiz.questions.append(second)
    self.assertEqual(quiz.get_ordered_questions(optimize_layout=True), [first, second])

  @unittest.skipUnless(shutil.which("typst"), "Typst is required")
  def test_typst_boundaries_and_overflow(self):
    with tempfile.TemporaryDirectory() as directory:
      source = Path(directory) / "quiz.typ"
      # Record the actual page and vertical position at each question boundary.
      questions = '\n'.join(
        '#question(1, questions_per_page: 3, question_name: "test")[Hello]'
        for _ in range(6)
      )
      source.write_text(Document.TYPST_HEADER + questions)
      result = subprocess.run(["typst", "query", str(source), "<question-allocation>"],
                              capture_output=True, text=True, check=True)
      positions = [item["value"]["position"] for item in json.loads(result.stdout)]
      self.assertEqual([item["page"] for item in positions], [1, 1, 1, 2, 2, 2])
      self.assertEqual([item["y"] for item in positions[:3]],
                       [item["y"] for item in positions[3:]])
      source.write_text(Document.TYPST_HEADER +
                        '#question(1, questions_per_page: 2)[Half]'
                        '#question(1, questions_per_page: 1, extra_pages: 2)[Full]'
                        '#question(1, questions_per_page: 2)[Next]')
      result = subprocess.run(["typst", "query", str(source), "<question-allocation>"],
                              capture_output=True, text=True, check=True)
      positions = [item["value"]["position"] for item in json.loads(result.stdout)]
      self.assertEqual([item["page"] for item in positions], [1, 2, 5])
      # Wrapped prose must be measured at printable width, not intrinsic width.
      prose = "Explain how this operation changes the filesystem state. " * 8
      source.write_text(Document.TYPST_HEADER +
                        '#question(1, questions_per_page: 3)[' + prose + ']')
      subprocess.run(["typst", "query", str(source), "<question-allocation>"],
                     capture_output=True, text=True, check=True)
      source.write_text(Document.TYPST_HEADER +
                        '#question(1, questions_per_page: 100, question_name: "Too tall")[Hello]')
      result = subprocess.run(["typst", "query", str(source), "<question-allocation>"],
                              capture_output=True, text=True)
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("Too tall", result.stderr)
      self.assertIn("allocation provides", result.stderr)

  @unittest.skipUnless(shutil.which("typst"), "Typst is required")
  def test_first_page_header_leaves_room_for_two_half_page_questions(self):
    from QuizGenerator.generation import contentast as ca
    document = Document(title="Header test")
    for number in range(4):
      document.add_element(ca.Question(
        body=ca.Section([ca.Paragraph(["Short response"])]),
        explanation=ca.Section(), questions_per_page=2,
        name=str(number), show_pdf_aids=False,
      ))
    with tempfile.TemporaryDirectory() as directory:
      source = Path(directory) / "quiz.typ"
      source.write_text(document.render("typst"))
      result = subprocess.run(
        ["typst", "eval", "query(<question-allocation>)", "--in", str(source)],
        capture_output=True, text=True, check=True,
      )
      allocations = [item["value"] for item in json.loads(result.stdout)]
      self.assertEqual([item["position"]["page"] for item in allocations], [1, 1, 2, 2])
      self.assertEqual([item["number"] for item in allocations], [1, 2, 3, 4])
