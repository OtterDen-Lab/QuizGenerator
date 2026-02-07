from QuizGenerator.quiz import Quiz


def test_yaml_list_preserves_order():
    yaml_text = """
name: "Ordered Quiz"
questions:
  - name: "First"
    points: 2
    class: FromText
    kwargs:
      text: "Q1"
  - name: "Second"
    points: 10
    class: FromText
    kwargs:
      text: "Q2"
"""
    quiz = Quiz.from_yaml(_write_tmp_yaml(yaml_text))[0]
    ordered = quiz.get_ordered_questions()
    assert [q.name for q in ordered] == ["First", "Second"]


def _write_tmp_yaml(text):
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "ordered.yaml"
    path.write_text(text)
    return str(path)
