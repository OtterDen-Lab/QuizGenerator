from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import QuizGenerator.generate as generate_module


class _DummyRenderedQuiz:
    def __init__(self):
        self.render_calls: list[tuple[str, dict]] = []
        self.render_latex_calls = 0

    def render(self, fmt, **kwargs):
        self.render_calls.append((fmt, kwargs))
        return f"{fmt}-source"

    def render_latex(self):
        self.render_latex_calls += 1
        return "latex-source"


class _DummyQuiz:
    def __init__(self, *, name="Dummy Quiz", practice=False):
        self.name = name
        self.practice = practice
        self.description = "dummy description"
        self.describe_calls = 0
        self.get_quiz_calls: list[dict] = []
        self.last_rendered_quiz: _DummyRenderedQuiz | None = None

    def get_quiz(self, **kwargs):
        self.get_quiz_calls.append(kwargs)
        self.last_rendered_quiz = _DummyRenderedQuiz()
        return self.last_rendered_quiz

    def describe(self):
        self.describe_calls += 1


def _write_minimal_yaml(path):
    path.write_text('name: "Test Quiz"\nquestions: {}\n', encoding="utf-8")


def test_generate_quiz_typst_passes_expected_options(monkeypatch, tmp_path):
    yaml_path = tmp_path / "quiz.yaml"
    _write_minimal_yaml(yaml_path)

    dummy_quiz = _DummyQuiz(name="Typst Quiz")
    monkeypatch.setattr(
        generate_module,
        "Quiz",
        SimpleNamespace(from_exam_dicts=lambda exam_dicts, source_path=None: [dummy_quiz]),
    )
    typst_mock = Mock(return_value=True)
    bundle_mock = Mock(return_value="out/quiz_bundle.zip")
    collect_mock = Mock(return_value=["out/test.pdf"])
    monkeypatch.setattr(generate_module, "generate_typst", typst_mock)
    monkeypatch.setattr(generate_module, "_bundle_outputs", bundle_mock)
    monkeypatch.setattr(generate_module, "_collect_recent_pdfs", collect_mock)
    monkeypatch.chdir(tmp_path)

    generate_module.generate_quiz(
        str(yaml_path),
        num_pdfs=1,
        use_typst=True,
        use_typst_measurement=True,
        base_seed=123,
        consistent_pages=True,
        layout_samples=7,
        layout_safety_factor=1.25,
        embed_images_typst=False,
        show_pdf_aids=False,
        optimize_layout=True,
        max_backoff_attempts=55,
    )

    assert dummy_quiz.describe_calls == 1
    assert len(dummy_quiz.get_quiz_calls) == 1
    get_quiz_kwargs = dummy_quiz.get_quiz_calls[0]
    assert get_quiz_kwargs["rng_seed"] == 123
    assert get_quiz_kwargs["use_typst_measurement"] is True
    assert get_quiz_kwargs["consistent_pages"] is True
    assert get_quiz_kwargs["layout_samples"] == 7
    assert get_quiz_kwargs["layout_safety_factor"] == 1.25
    assert get_quiz_kwargs["max_backoff_attempts"] == 55
    assert get_quiz_kwargs["optimize_layout"] is True

    assert dummy_quiz.last_rendered_quiz is not None
    assert dummy_quiz.last_rendered_quiz.render_calls == [
        ("typst", {"embed_images_typst": False, "show_pdf_aids": False})
    ]
    typst_mock.assert_called_once_with(
        "typst-source",
        remove_previous=True,
        name_prefix="Typst Quiz",
    )
    bundle_mock.assert_called_once()
    replay_path = tmp_path / "out" / "quiz_replay.yaml"
    assert replay_path.exists()


def test_generate_quiz_latex_failure_raises_quizgen_error(monkeypatch, tmp_path):
    yaml_path = tmp_path / "quiz.yaml"
    _write_minimal_yaml(yaml_path)

    dummy_quiz = _DummyQuiz(name="LaTeX Quiz")
    monkeypatch.setattr(
        generate_module,
        "Quiz",
        SimpleNamespace(from_exam_dicts=lambda exam_dicts, source_path=None: [dummy_quiz]),
    )
    latex_mock = Mock(return_value=False)
    monkeypatch.setattr(generate_module, "generate_latex", latex_mock)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(generate_module.QuizGenError, match="LaTeX"):
        generate_module.generate_quiz(
            str(yaml_path),
            num_pdfs=1,
            use_typst=False,
            base_seed=77,
        )

    assert dummy_quiz.last_rendered_quiz is not None
    assert dummy_quiz.last_rendered_quiz.render_latex_calls == 1
    latex_mock.assert_called_once_with(
        "latex-source",
        remove_previous=True,
        name_prefix="LaTeX Quiz",
    )


def test_generate_quiz_canvas_upload_wires_arguments(monkeypatch, tmp_path):
    yaml_path = tmp_path / "quiz.yaml"
    _write_minimal_yaml(yaml_path)

    dummy_quiz = _DummyQuiz(name="Canvas Quiz", practice=True)
    monkeypatch.setattr(
        generate_module,
        "Quiz",
        SimpleNamespace(from_exam_dicts=lambda exam_dicts, source_path=None: [dummy_quiz]),
    )

    assignment_group = SimpleNamespace(name="dev")
    canvas_course = Mock()
    canvas_course.create_assignment_group.return_value = assignment_group

    interface_instances = []

    class _FakeCanvasInterface:
        def __init__(self, *, prod=False, env_path=None):
            self.prod = prod
            self.env_path = env_path
            self.requested_course_id = None
            interface_instances.append(self)

        def get_course(self, course_id=None):
            self.requested_course_id = course_id
            return canvas_course

    upload_mock = Mock()
    monkeypatch.setattr(generate_module, "CanvasInterface", _FakeCanvasInterface)
    monkeypatch.setattr(generate_module, "upload_quiz_to_canvas", upload_mock)
    monkeypatch.chdir(tmp_path)

    generate_module.generate_quiz(
        str(yaml_path),
        num_canvas=3,
        use_prod=True,
        course_id=2468,
        delete_assignment_group=True,
        env_path="/tmp/test.env",
        optimize_layout=True,
        max_backoff_attempts=9,
        quiet=True,
    )

    assert len(interface_instances) == 1
    assert interface_instances[0].prod is True
    assert interface_instances[0].env_path == "/tmp/test.env"
    assert interface_instances[0].requested_course_id == 2468
    canvas_course.create_assignment_group.assert_called_once_with(
        name="dev",
        delete_existing=True,
    )
    upload_mock.assert_called_once_with(
        canvas_course,
        dummy_quiz,
        3,
        title="Canvas Quiz",
        is_practice=True,
        assignment_group=assignment_group,
        optimize_layout=True,
        max_backoff_attempts=9,
        quiet=True,
    )
    assert dummy_quiz.describe_calls == 1
