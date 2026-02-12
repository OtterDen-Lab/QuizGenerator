"""
Tests for Canvas LMS integration.

These tests use mocking to avoid hitting real Canvas APIs.
They verify the logic of the canvas_interface module without network calls.
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestCanvasInterfaceCredentials:
    """Tests for CanvasInterface credential handling."""

    def test_missing_credentials_raises_valueerror(self):
        """CanvasInterface should raise ValueError when credentials are missing."""
        from lms_interface.canvas_interface import CanvasInterface

        # Clear any existing credentials and prevent dotenv from loading from file
        with patch.dict(os.environ, {}, clear=True):
            with patch("lms_interface.canvas_interface.dotenv.load_dotenv"):
                with pytest.raises(ValueError, match="Canvas credentials are missing"):
                    CanvasInterface()

    def test_prod_mode_uses_prod_credentials(self):
        """In prod mode, should use _prod suffixed env vars."""
        from lms_interface.canvas_interface import CanvasInterface

        with patch.dict(os.environ, {
            "CANVAS_API_URL_prod": "https://prod.canvas.com",
            "CANVAS_API_KEY_prod": "prod_key_123"
        }, clear=True):
            with patch("lms_interface.canvas_interface.canvasapi.Canvas"):
                interface = CanvasInterface(prod=True)
                assert interface.canvas_url == "https://prod.canvas.com"
                assert interface.canvas_key == "prod_key_123"
                assert interface.prod is True

    def test_dev_mode_uses_standard_credentials(self):
        """In dev mode, should use standard env vars."""
        from lms_interface.canvas_interface import CanvasInterface

        with patch.dict(os.environ, {
            "CANVAS_API_URL": "https://dev.canvas.com",
            "CANVAS_API_KEY": "dev_key_456"
        }, clear=True):
            with patch("lms_interface.canvas_interface.canvasapi.Canvas"):
                interface = CanvasInterface(prod=False)
                assert interface.canvas_url == "https://dev.canvas.com"
                assert interface.canvas_key == "dev_key_456"
                assert interface.prod is False


class TestCanvasExceptionHandling:
    """Tests for Canvas exception handling functions."""

    def test_canvas_exception_status_extraction(self):
        """Should extract status code from Canvas exceptions."""
        from lms_interface.canvas_interface import _canvas_exception_status

        # Exception with status_code attribute
        exc1 = Mock()
        exc1.status_code = 429
        assert _canvas_exception_status(exc1) == 429

        # Exception with response.status_code
        exc2 = Mock(spec=[])  # No status_code attribute
        exc2.response = Mock()
        exc2.response.status_code = 500
        assert _canvas_exception_status(exc2) == 500

        # Exception with neither
        exc3 = Exception("Generic error")
        assert _canvas_exception_status(exc3) is None

    def test_retryable_exception_429(self):
        """429 (rate limit) should be retryable."""
        from lms_interface.canvas_interface import _is_retryable_canvas_exception

        exc = Mock()
        exc.status_code = 429
        assert _is_retryable_canvas_exception(exc) is True

    def test_retryable_exception_5xx(self):
        """5xx errors should be retryable."""
        from lms_interface.canvas_interface import _is_retryable_canvas_exception

        for status in [500, 502, 503, 504, 599]:
            exc = Mock()
            exc.status_code = status
            assert _is_retryable_canvas_exception(exc) is True, f"Status {status} should be retryable"

    def test_non_retryable_4xx(self):
        """4xx errors (except 429) should not be retryable."""
        from lms_interface.canvas_interface import _is_retryable_canvas_exception

        for status in [400, 401, 403, 404, 422]:
            exc = Mock()
            exc.status_code = status
            assert _is_retryable_canvas_exception(exc) is False, f"Status {status} should not be retryable"

    def test_unknown_exception_is_retryable(self):
        """Exceptions without status are retryable (conservative approach)."""
        from lms_interface.canvas_interface import _is_retryable_canvas_exception

        exc = Exception("Network error")
        assert _is_retryable_canvas_exception(exc) is True


class TestCanvasCourse:
    """Tests for CanvasCourse class."""

    @pytest.fixture
    def mock_canvas_course(self):
        """Create a mock Canvas course."""
        from lms_interface.canvas_interface import CanvasCourse, CanvasInterface

        mock_interface = Mock(spec=CanvasInterface)
        mock_canvasapi_course = MagicMock()
        mock_canvasapi_course.name = "Test Course"
        mock_canvasapi_course.id = 12345

        return CanvasCourse(
            canvas_interface=mock_interface,
            canvasapi_course=mock_canvasapi_course
        )

    def test_create_assignment_group_new(self, mock_canvas_course):
        """Should create a new assignment group when none exists."""
        mock_canvas_course.course.get_assignment_groups.return_value = []
        mock_new_group = Mock()
        mock_new_group.name = "test_group"
        mock_new_group.id = 999
        mock_canvas_course.course.create_assignment_group.return_value = mock_new_group

        result = mock_canvas_course.create_assignment_group(name="test_group")

        assert result == mock_new_group
        mock_canvas_course.course.create_assignment_group.assert_called_once()

    def test_create_assignment_group_existing(self, mock_canvas_course):
        """Should return existing assignment group when one exists."""
        existing_group = Mock()
        existing_group.name = "existing_group"
        existing_group.id = 888
        mock_canvas_course.course.get_assignment_groups.return_value = [existing_group]

        result = mock_canvas_course.create_assignment_group(name="existing_group")

        assert result == existing_group
        mock_canvas_course.course.create_assignment_group.assert_not_called()

    def test_create_assignment_group_delete_existing(self, mock_canvas_course):
        """Should delete and recreate when delete_existing=True."""
        existing_group = Mock()
        existing_group.name = "dev"
        existing_group.id = 888
        mock_canvas_course.course.get_assignment_groups.return_value = [existing_group]

        new_group = Mock()
        new_group.name = "dev"
        new_group.id = 999
        mock_canvas_course.course.create_assignment_group.return_value = new_group

        result = mock_canvas_course.create_assignment_group(name="dev", delete_existing=True)

        existing_group.delete.assert_called_once()
        assert result == new_group

    def test_add_quiz(self, mock_canvas_course):
        """Should create a quiz with correct parameters."""
        mock_group = Mock()
        mock_group.id = 123

        mock_quiz = Mock()
        mock_canvas_course.course.create_quiz.return_value = mock_quiz

        result = mock_canvas_course.add_quiz(
            mock_group,
            title="Test Quiz",
            is_practice=False,
            description="Test description"
        )

        assert result == mock_quiz

        # Check that create_quiz was called with correct quiz_type
        call_kwargs = mock_canvas_course.course.create_quiz.call_args
        quiz_params = call_kwargs[1]["quiz"]
        assert quiz_params["title"] == "Test Quiz"
        assert quiz_params["quiz_type"] == "assignment"
        assert quiz_params["assignment_group_id"] == 123

    def test_add_quiz_practice_mode(self, mock_canvas_course):
        """Practice quizzes should have quiz_type='practice_quiz'."""
        mock_group = Mock()
        mock_group.id = 123

        mock_quiz = Mock()
        mock_canvas_course.course.create_quiz.return_value = mock_quiz

        mock_canvas_course.add_quiz(mock_group, title="Practice", is_practice=True)

        call_kwargs = mock_canvas_course.course.create_quiz.call_args
        quiz_params = call_kwargs[1]["quiz"]
        assert quiz_params["quiz_type"] == "practice_quiz"

    def test_get_students(self, mock_canvas_course):
        """Should return list of Student objects."""
        from lms_interface.classes import Student

        mock_user1 = Mock()
        mock_user1.name = "Alice"
        mock_user1.id = 1

        mock_user2 = Mock()
        mock_user2.name = "Bob"
        mock_user2.id = 2

        mock_canvas_course.course.get_users.return_value = [mock_user1, mock_user2]

        students = mock_canvas_course.get_students(include_names=True)

        assert len(students) == 2
        assert all(isinstance(s, Student) for s in students)
        assert students[0].name == "Alice"
        assert students[1].name == "Bob"


class TestQuestionUpload:
    """Tests for question upload logic."""

    @pytest.fixture
    def mock_canvas_course(self):
        """Create a mock Canvas course for question upload tests."""
        from lms_interface.canvas_interface import CanvasCourse, CanvasInterface

        mock_interface = Mock(spec=CanvasInterface)
        mock_canvasapi_course = MagicMock()

        return CanvasCourse(
            canvas_interface=mock_interface,
            canvasapi_course=mock_canvasapi_course
        )

    def test_create_question_single_payload(self, mock_canvas_course):
        """Single payload should be uploaded without creating a group."""
        mock_quiz = Mock()
        mock_quiz.create_question = Mock()

        payload = {
            "question_name": "Q1",
            "question_text": "What is 2+2?",
            "question_type": "numerical_question",
            "points_possible": 5
        }

        result = mock_canvas_course.create_question(mock_quiz, payload, max_workers=1)

        # Single payload should not create a group
        assert result is None
        mock_quiz.create_question.assert_called_once()

    def test_create_question_multiple_payloads_creates_group(self, mock_canvas_course):
        """Multiple payloads should create a question group."""
        mock_quiz = Mock()
        mock_quiz.create_question = Mock()

        mock_group = Mock()
        mock_group.id = 999
        mock_quiz.create_question_group = Mock(return_value=mock_group)

        payloads = [
            {"question_name": "Q1", "question_text": "Question 1", "points_possible": 5},
            {"question_name": "Q2", "question_text": "Question 2", "points_possible": 5},
        ]

        result = mock_canvas_course.create_question(
            mock_quiz,
            payloads,
            question_points=5,
            max_workers=1
        )

        assert result == mock_group
        mock_quiz.create_question_group.assert_called_once()
        # Both questions should be uploaded
        assert mock_quiz.create_question.call_count == 2

    def test_create_question_empty_payloads_returns_none(self, mock_canvas_course):
        """Empty payload list should return None."""
        mock_quiz = Mock()

        result = mock_canvas_course.create_question(mock_quiz, [])

        assert result is None


class TestSubmissionClasses:
    """Tests for submission data classes."""

    def test_submission_status_from_string(self):
        """Test Submission.Status.from_string conversion."""
        from lms_interface.classes import Submission

        assert Submission.Status.from_string("graded", 100) == Submission.Status.GRADED
        assert Submission.Status.from_string("unsubmitted", None) == Submission.Status.MISSING
        assert Submission.Status.from_string("submitted", None) == Submission.Status.UNGRADED
        assert Submission.Status.from_string("pending_review", None) == Submission.Status.UNGRADED

    def test_text_submission_word_count(self):
        """Test TextSubmission word counting."""
        from lms_interface.classes import TextSubmission

        submission = TextSubmission(submission_text="Hello world this is a test")

        assert submission.get_word_count() == 6
        assert submission.get_character_count(include_spaces=True) == 26
        assert submission.get_character_count(include_spaces=False) == 21

    def test_text_submission_empty(self):
        """Test TextSubmission with empty text."""
        from lms_interface.classes import TextSubmission

        submission = TextSubmission(submission_text="")

        assert submission.get_word_count() == 0
        assert submission.get_character_count() == 0
        assert submission.get_paragraph_count() == 0

    def test_text_submission_paragraph_count(self):
        """Test TextSubmission paragraph counting."""
        from lms_interface.classes import TextSubmission

        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        submission = TextSubmission(submission_text=text)

        assert submission.get_paragraph_count() == 3


class TestFeedback:
    """Tests for Feedback dataclass."""

    def test_feedback_ordering(self):
        """Feedback should be orderable by percentage_score."""
        from lms_interface.classes import Feedback

        f1 = Feedback(percentage_score=50.0)
        f2 = Feedback(percentage_score=80.0)
        f3 = Feedback(percentage_score=30.0)

        assert f3 < f1 < f2
        assert f2 > f1 > f3

    def test_feedback_none_score_handling(self):
        """Feedback with None score should be treated as greater (not graded yet)."""
        from lms_interface.classes import Feedback

        graded = Feedback(percentage_score=50.0)
        ungraded = Feedback(percentage_score=None)

        # None should not be less than any value
        assert not (ungraded < graded)

    def test_feedback_str(self):
        """Test Feedback string representation."""
        from lms_interface.classes import Feedback

        feedback = Feedback(percentage_score=85.5, comments="Good work!")
        s = str(feedback)

        assert "85.5" in s
        assert "Good work" in s


class TestStudent:
    """Tests for Student class."""

    def test_student_creation(self):
        """Test Student dataclass creation."""
        from lms_interface.classes import Student

        mock_inner = Mock()
        student = Student(name="Alice", user_id=123, _inner=mock_inner)

        assert student.name == "Alice"
        assert student.user_id == 123

    def test_student_wrapper_access(self):
        """Student should pass through unknown attributes to inner object."""
        from lms_interface.classes import Student

        mock_inner = Mock()
        mock_inner.email = "alice@example.com"
        student = Student(name="Alice", user_id=123, _inner=mock_inner)

        # Should delegate to mock_inner
        assert student.email == "alice@example.com"
