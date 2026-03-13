from QuizGenerator.generation import contentast as ca
from QuizGenerator.generation.quiz import Quiz


def test_default_order_preserves_yaml_order_within_point_tier():
  exam_dict = {
    "name": "Ordering Test",
    "questions": {
      8: {
        "Misc First": {
          "class": "FromText",
          "topic": "misc",
          "kwargs": {"text": "First in YAML."},
        },
        "Memory Second": {
          "class": "FromText",
          "topic": "memory",
          "kwargs": {"text": "Second in YAML."},
        },
      },
      4: {
        "Lower Value": {
          "class": "FromText",
          "topic": "processes",
          "kwargs": {"text": "Comes after all 8-point questions."},
        },
      },
    },
  }

  quiz = Quiz.from_exam_dicts([exam_dict])[0]

  assert [question.name for question in quiz.get_ordered_questions()] == [
    "Misc First",
    "Memory Second",
    "Lower Value",
  ]


def test_page_spacing_starts_later_questions_on_fresh_page_and_fills_page():
  question = ca.Question(
    body=ca.Section([ca.Paragraph(["Write the function below."])]),
    explanation=ca.Section(),
    value=8,
    spacing=99,
    question_number=2,
  )

  rendered_question = question.render("typst")
  assert rendered_question.startswith("#pagebreak(weak: true)\n")
  assert "\n#question(" in rendered_question

  document = ca.Document()
  document.add_element(question)
  rendered_document = document.render("typst")
  assert "if spacing >= 99cm {" in rendered_document
  assert "v(1fr)" in rendered_document
