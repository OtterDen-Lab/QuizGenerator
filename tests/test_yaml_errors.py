import pytest

from QuizGenerator.quiz import Quiz


def test_yaml_missing_questions_key(tmp_path):
    yaml_text = """
name: "Bad Quiz"
"""
    path = tmp_path / "bad.yaml"
    path.write_text(yaml_text)
    with pytest.raises(KeyError):
        Quiz.from_yaml(str(path))


def test_yaml_list_entry_missing_name(tmp_path):
    yaml_text = """
name: "Bad Quiz"
questions:
  - points: 5
    class: FromText
    kwargs:
      text: "hi"
"""
    path = tmp_path / "bad.yaml"
    path.write_text(yaml_text)
    with pytest.raises(ValueError, match="include a 'name'"):
        Quiz.from_yaml(str(path))


def test_yaml_list_entry_missing_points(tmp_path):
    yaml_text = """
name: "Bad Quiz"
questions:
  - name: "No points"
    class: FromText
    kwargs:
      text: "hi"
"""
    path = tmp_path / "bad.yaml"
    path.write_text(yaml_text)
    with pytest.raises(ValueError, match="include 'points'"):
        Quiz.from_yaml(str(path))


def test_yaml_unknown_question_class(tmp_path):
    yaml_text = """
name: "Bad Quiz"
questions:
  5:
    "Oops":
      class: NotAQuestion
"""
    path = tmp_path / "bad.yaml"
    path.write_text(yaml_text)
    with pytest.raises(ValueError, match="Unknown question type"):
        Quiz.from_yaml(str(path))


def test_yaml_fromgenerator_disabled_error(tmp_path, monkeypatch):
    monkeypatch.delenv("QUIZGEN_ALLOW_GENERATOR", raising=False)
    yaml_text = """
name: "Gen Quiz"
questions:
  5:
    "Gen":
      class: FromGenerator
      generator: |
        return "hi"
"""
    path = tmp_path / "gen.yaml"
    path.write_text(yaml_text)
    with pytest.raises(ValueError, match="FromGenerator is disabled by default"):
        Quiz.from_yaml(str(path))


def test_yaml_seed_group_must_be_string(tmp_path):
    yaml_text = """
name: "Bad seed group"
questions:
  5:
    "Q1":
      class: FromText
      seed_group: 123
      kwargs:
        text: "hi"
"""
    path = tmp_path / "bad-seed-group.yaml"
    path.write_text(yaml_text)
    with pytest.raises(ValueError, match="seed_group"):
        Quiz.from_yaml(str(path))
