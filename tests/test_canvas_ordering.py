from unittest.mock import Mock

from QuizGenerator.generate import upload_quiz_to_canvas
from QuizGenerator.premade_questions.basic import FromText
from QuizGenerator.quiz import Quiz


def _make_canvas_course():
    canvas_course = Mock()
    canvas_course.course = Mock()
    canvas_course.course.id = 1
    canvas_course.course.create_folder = Mock()
    canvas_course.course.upload = Mock(return_value=(True, {"id": 1}))
    canvas_course.add_quiz = Mock(return_value=Mock(html_url="http://example"))
    canvas_course.create_question = Mock()
    return canvas_course


def test_canvas_upload_preserves_yaml_order_when_enabled():
    q1 = FromText(name="A", points_value=1.0, text="A")
    q2 = FromText(name="B", points_value=10.0, text="B")
    quiz = Quiz("Test", [q1, q2], practice=False, preserve_yaml_order=True)

    canvas_course = _make_canvas_course()
    upload_quiz_to_canvas(
        canvas_course,
        quiz,
        num_variations=1,
        title="Test",
        assignment_group=Mock()
    )

    names = [call.kwargs["group_name"] for call in canvas_course.create_question.call_args_list]
    assert names == ["A", "B"]


def test_canvas_upload_defaults_to_point_order():
    q1 = FromText(name="A", points_value=1.0, text="A")
    q2 = FromText(name="B", points_value=10.0, text="B")
    quiz = Quiz("Test", [q1, q2], practice=False, preserve_yaml_order=False)

    canvas_course = _make_canvas_course()
    upload_quiz_to_canvas(
        canvas_course,
        quiz,
        num_variations=1,
        title="Test",
        assignment_group=Mock()
    )

    names = [call.kwargs["group_name"] for call in canvas_course.create_question.call_args_list]
    assert names == ["B", "A"]
