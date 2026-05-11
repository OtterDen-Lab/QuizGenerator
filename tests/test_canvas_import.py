from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from QuizGenerator.canvas_import import export_canvas_quiz_to_yaml, resolve_canvas_quiz
from QuizGenerator.generation.quiz import Quiz


class _FakeFile:
    def __init__(self, content: bytes):
        self.content = content

    def download(self, location):
        Path(location).write_bytes(self.content)


class _FakeRequester:
    def __init__(self, content: bytes, headers=None):
        self.content = content
        self.headers = headers or {}

    def request(self, method, _url=None, _kwargs=None):  # noqa: ARG002
        return SimpleNamespace(content=self.content, headers=self.headers)


class _FakeCourse:
    def __init__(self, *, files=None, quizzes=None):
        self.files = files or {}
        self.quizzes = quizzes or []
        self._requester = _FakeRequester(b"")

    def get_file(self, file_id):
        return self.files[int(file_id)]

    def get_quizzes(self):
        return list(self.quizzes)

    def get_quiz(self, quiz_id):
        for quiz in self.quizzes:
            if int(getattr(quiz, "id")) == int(quiz_id):
                return quiz
        raise KeyError(quiz_id)


class _FakeCanvasCourse:
    def __init__(self, course):
        self.course = course


def test_resolve_canvas_quiz_by_assignment_id():
    quizzes = [
        SimpleNamespace(id=1, assignment_id=10, title="Quiz A"),
        SimpleNamespace(id=2, assignment_id=11, title="Quiz B"),
    ]
    canvas_course = _FakeCanvasCourse(_FakeCourse(quizzes=quizzes))

    quiz = resolve_canvas_quiz(canvas_course, assignment_id=11)

    assert quiz.title == "Quiz B"


def test_export_canvas_quiz_flatten_groups_and_download_images(tmp_path):
    image_bytes = b"fake-image-bytes"
    course = _FakeCourse(
        files={99: _FakeFile(image_bytes)},
    )
    canvas_course = _FakeCanvasCourse(course)

    group_question = SimpleNamespace(
        id=101,
        quiz_group_id=7,
        question_name="Group Variant 1",
        question_type="multiple_choice_question",
        question_text='<p>Pick one</p><img src="/courses/1/files/99/preview" alt="diagram">',
        points_possible=5,
        answers=[
            {"answer_text": "A", "answer_weight": 100},
            {"answer_text": "B", "answer_weight": 0},
        ],
        correct_comments_html="<p>Good.</p>",
        incorrect_comments_html=None,
        neutral_comments_html="<p>Check the definition.</p>",
    )
    skipped_group_variant = SimpleNamespace(
        id=102,
        quiz_group_id=7,
        question_name="Group Variant 2",
        question_type="multiple_choice_question",
        question_text="<p>Different variant</p>",
        points_possible=5,
        answers=[
            {"answer_text": "C", "answer_weight": 100},
            {"answer_text": "D", "answer_weight": 0},
        ],
    )
    standalone_question = SimpleNamespace(
        id=103,
        quiz_group_id=None,
        question_name="Standalone",
        question_type="essay_question",
        question_text="<p>Explain it.</p>",
        points_possible=2,
        answers=[],
        correct_comments_html=None,
        incorrect_comments_html=None,
        neutral_comments_html=None,
    )
    quiz = SimpleNamespace(
        id=555,
        title="Imported Quiz",
        quiz_type="practice_quiz",
        description="<p>Quiz description</p>",
        get_questions=lambda: [group_question, skipped_group_variant, standalone_question],
        get_quiz_group=lambda group_id: SimpleNamespace(
            id=group_id,
            name="Group Name",
            question_points=7,
        ),
    )
    course.quizzes = [quiz]

    output_path = tmp_path / "canvas_import.yaml"
    result_path = export_canvas_quiz_to_yaml(
        canvas_course,
        quiz_id=555,
        output_path=output_path,
        flatten_groups=True,
    )

    assert result_path == output_path.resolve()
    assert output_path.exists()

    exported = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert exported["name"] == "Imported Quiz"
    assert exported["practice"] is True
    assert len(exported["questions"]) == 2
    assert exported["questions"][0]["name"] == "Group Name"
    assert exported["questions"][0]["points"] == 7.0

    body_nodes = exported["questions"][0]["kwargs"]["yaml_spec"]["body"]
    assert any("picture" in node for node in body_nodes)

    image_path = Path(body_nodes[1]["picture"]["path"])
    assert image_path.exists()
    assert image_path.read_bytes() == image_bytes

    quizzes = Quiz.from_yaml(str(output_path))
    instantiated = quizzes[0].questions[0].instantiate(rng_seed=123)
    latex = instantiated.body.render("latex")
    assert "\\includegraphics" in latex
    assert "Pick one" in instantiated.body.render("html")


def test_export_canvas_quiz_preserves_tables_and_inline_blanks(tmp_path):
    course = _FakeCourse()
    canvas_course = _FakeCanvasCourse(course)

    question = SimpleNamespace(
        id=201,
        quiz_group_id=None,
        question_name="Table and blanks",
        question_type="fill_in_multiple_blanks_question",
        question_text="""
<p>Complete the table and fill in the missing value.</p>
<table>
  <tr>
    <th>Item</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Alpha</td>
    <td>[alpha_blank]</td>
  </tr>
</table>
<p>Final answer: [final_blank]</p>
""",
        points_possible=3,
        answers=[
            {"blank_id": "alpha_blank", "answer_text": "42", "answer_weight": 100},
            {"blank_id": "final_blank", "answer_text": "done", "answer_weight": 100},
        ],
        correct_comments_html=None,
        incorrect_comments_html=None,
        neutral_comments_html=None,
    )
    quiz = SimpleNamespace(
        id=777,
        title="Table Quiz",
        quiz_type="assignment_quiz",
        description=None,
        get_questions=lambda: [question],
        get_quiz_group=lambda group_id: SimpleNamespace(
            id=group_id,
            name=None,
            question_points=None,
        ),
    )
    course.quizzes = [quiz]

    output_path = tmp_path / "canvas_table_import.yaml"
    export_canvas_quiz_to_yaml(
        canvas_course,
        quiz_id=777,
        output_path=output_path,
        flatten_groups=True,
    )

    exported = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    body = exported["questions"][0]["kwargs"]["yaml_spec"]["body"]
    assert any(isinstance(node, dict) and "table" in node for node in body)

    quizzes = Quiz.from_yaml(str(output_path))
    instantiated = quizzes[0].questions[0].instantiate(rng_seed=123)
    latex = instantiated.body.render("latex")

    assert "\\begin{tabular}" in latex
    assert "[alpha_blank]" not in latex
    assert "[final_blank]" not in latex
    assert "\\answerblank" in latex


def test_export_canvas_quiz_splits_inline_math_in_table_cells(tmp_path):
    course = _FakeCourse()
    canvas_course = _FakeCanvasCourse(course)

    question = SimpleNamespace(
        id=202,
        quiz_group_id=None,
        question_name="Math table",
        question_type="essay_question",
        question_text="""
<table>
  <tr>
    <th>Label</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>w11</td>
    <td>\\(w_{11}\\)</td>
  </tr>
</table>
""",
        points_possible=1,
        answers=[],
        correct_comments_html=None,
        incorrect_comments_html=None,
        neutral_comments_html=None,
    )
    quiz = SimpleNamespace(
        id=779,
        title="Math Table Quiz",
        quiz_type="assignment_quiz",
        description=None,
        get_questions=lambda: [question],
        get_quiz_group=lambda group_id: SimpleNamespace(
            id=group_id,
            name=None,
            question_points=None,
        ),
    )
    course.quizzes = [quiz]

    output_path = tmp_path / "canvas_math_table_import.yaml"
    export_canvas_quiz_to_yaml(
        canvas_course,
        quiz_id=779,
        output_path=output_path,
        flatten_groups=True,
    )

    exported = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    body = exported["questions"][0]["kwargs"]["yaml_spec"]["body"]
    table_nodes = [node for node in body if isinstance(node, dict) and "table" in node]
    assert table_nodes
    table = table_nodes[0]["table"]
    assert any(
        isinstance(cell, dict) and "equation" in cell
        for row in table.get("rows", [])
        for cell in row
    )

    quizzes = Quiz.from_yaml(str(output_path))
    instantiated = quizzes[0].questions[0].instantiate(rng_seed=123)
    typst = instantiated.body.render("typst")
    assert "$ w_(11) $" in typst or "$w_(11)$" in typst


def test_export_canvas_quiz_uses_image_content_type_extension(tmp_path):
    course = _FakeCourse()
    course._requester = _FakeRequester(b"fake-png-bytes", headers={"Content-Type": "image/png"})
    canvas_course = _FakeCanvasCourse(course)

    question = SimpleNamespace(
        id=301,
        quiz_group_id=None,
        question_name="Preview image",
        question_type="essay_question",
        question_text='<p>See this diagram:</p><img src="https://canvas.example/files/preview" alt="diagram">',
        points_possible=1,
        answers=[],
        correct_comments_html=None,
        incorrect_comments_html=None,
        neutral_comments_html=None,
    )
    quiz = SimpleNamespace(
        id=888,
        title="Image Quiz",
        quiz_type="assignment_quiz",
        description=None,
        get_questions=lambda: [question],
        get_quiz_group=lambda group_id: SimpleNamespace(
            id=group_id,
            name=None,
            question_points=None,
        ),
    )
    course.quizzes = [quiz]

    output_path = tmp_path / "canvas_image_import.yaml"
    export_canvas_quiz_to_yaml(
        canvas_course,
        quiz_id=888,
        output_path=output_path,
        flatten_groups=True,
    )

    exported = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    body = exported["questions"][0]["kwargs"]["yaml_spec"]["body"]
    picture_nodes = [node for node in body if isinstance(node, dict) and "picture" in node]
    assert picture_nodes, body
    assert Path(picture_nodes[0]["picture"]["path"]).suffix == ".png"


def test_export_canvas_quiz_wraps_long_yaml_lines(tmp_path):
    course = _FakeCourse()
    canvas_course = _FakeCanvasCourse(course)

    long_text = (
        "This is a deliberately long line that should be wrapped by the YAML emitter so "
        "the exported canvas import file stays readable even when Canvas provides a lot "
        "of prose in a single paragraph."
    )
    question = SimpleNamespace(
        id=401,
        quiz_group_id=None,
        question_name="Wrapped lines",
        question_type="essay_question",
        question_text=f"<p>{long_text}</p>",
        points_possible=1,
        answers=[],
        correct_comments_html=None,
        incorrect_comments_html=None,
        neutral_comments_html=None,
    )
    quiz = SimpleNamespace(
        id=990,
        title="Wrapped YAML Quiz",
        quiz_type="assignment_quiz",
        description=long_text,
        get_questions=lambda: [question],
        get_quiz_group=lambda group_id: SimpleNamespace(
            id=group_id,
            name=None,
            question_points=None,
        ),
    )
    course.quizzes = [quiz]

    output_path = tmp_path / "canvas_wrapped.yaml"
    export_canvas_quiz_to_yaml(
        canvas_course,
        quiz_id=990,
        output_path=output_path,
        flatten_groups=True,
    )

    text = output_path.read_text(encoding="utf-8")
    assert max(len(line) for line in text.splitlines()) < 100


def test_answer_blanks_in_tables_render_relative_width():
    from QuizGenerator.generation.contentast import Answer, Table

    table = Table([[Answer(value="42", blank_length=20)]])
    typst = table.render("typst")

    assert "#fillline(width: 100%)" in typst


def test_equation_typst_strips_displaystyle():
    from QuizGenerator.generation.contentast import Equation

    eq = Equation(r"$$\displaystyle 0.5 \left(x_{0} - 1\right)^{2}$$")
    typst = eq.render("typst")

    assert "displaystyle" not in typst
    assert r"\left" not in typst
    assert r"\right" not in typst
    assert "$$" not in typst


def test_equation_typst_converts_wrapped_matrix():
    from QuizGenerator.generation.contentast import Equation

    eq = Equation(r"\displaystyle \nabla f = \left[\begin{matrix}1.0 x_{0} - 1.0\\ 3.0 x_{1} - 3.0\end{matrix}\right]")
    typst = eq.render("typst")

    assert "[mat(" not in typst
    assert "mat(delim: \"[\"" in typst
    assert "x_(0)" in typst
    assert "x_(1)" in typst


def test_canvas_import_promotes_raw_latex_formula_lines(tmp_path):
    course = _FakeCourse()
    canvas_course = _FakeCanvasCourse(course)

    question = SimpleNamespace(
        id=501,
        quiz_group_id=None,
        question_name="Raw latex",
        question_type="essay_question",
        question_text="""
<p>Consider the optimization problem of minimizing the function:</p>
<p>$$\\displaystyle 0.5 \\left(x_{0} - 1\\right)^{2} + 1.5 \\left(x_{1} - 1\\right)^{2}$$</p>
""",
        points_possible=1,
        answers=[],
        correct_comments_html=None,
        incorrect_comments_html=None,
        neutral_comments_html=None,
    )
    quiz = SimpleNamespace(
        id=991,
        title="Raw Latex Quiz",
        quiz_type="assignment_quiz",
        description=None,
        get_questions=lambda: [question],
        get_quiz_group=lambda group_id: SimpleNamespace(
            id=group_id,
            name=None,
            question_points=None,
        ),
    )
    course.quizzes = [quiz]

    output_path = tmp_path / "canvas_raw_latex.yaml"
    export_canvas_quiz_to_yaml(
        canvas_course,
        quiz_id=991,
        output_path=output_path,
        flatten_groups=True,
    )

    exported = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    body = exported["questions"][0]["kwargs"]["yaml_spec"]["body"]
    equation_nodes = [node for node in body if isinstance(node, dict) and "equation" in node]
    assert equation_nodes
    assert equation_nodes[0]["equation"]["latex"].startswith("0.5")
    assert "$$" not in equation_nodes[0]["equation"]["latex"]
