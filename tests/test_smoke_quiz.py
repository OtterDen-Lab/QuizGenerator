from QuizGenerator.quiz import Quiz


def test_smoke_quiz_typst_render(tmp_path):
    yaml_text = """
name: "Smoke Test Quiz"
questions:
  1:
    "Simple text question":
      class: FromText
      kwargs:
        text: "What is 2 + 2?"
"""
    yaml_path = tmp_path / "smoke.yaml"
    yaml_path.write_text(yaml_text)

    quizzes = Quiz.from_yaml(str(yaml_path))
    assert quizzes, "Expected at least one quiz from smoke YAML"

    quiz = quizzes[0]
    typst_text = quiz.get_quiz(rng_seed=123).render("typst")
    assert "question(" in typst_text
