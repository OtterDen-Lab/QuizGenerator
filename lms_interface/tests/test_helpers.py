from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from lms_interface.helpers import (
  _parse_canvas_datetime,
  cleanup_missing_by_due_date,
)


@dataclass
class FakeStudent:
  user_id: int


class FakeSubmission:
  def __init__(
      self,
      *,
      workflow_state: str = "unsubmitted",
      submitted_at=None,
      late_policy_status: str | None = "none",
      cached_due_date: str | None = None,
      due_at: str | None = None,
      submission_type: str | None = None,
      attachments: list | None = None,
      body: str | None = None,
      url: str | None = None,
      media_comment_id=None,
      missing: bool = False,
      excused: bool = False,
  ):
    self.workflow_state = workflow_state
    self.submitted_at = submitted_at
    self.late_policy_status = late_policy_status
    self.cached_due_date = cached_due_date
    self.due_at = due_at
    self.submission_type = submission_type
    self.attachments = attachments
    self.body = body
    self.url = url
    self.media_comment_id = media_comment_id
    self.missing = missing
    self.excused = excused
    self.edit_calls = []

  def edit(self, *, submission):
    self.edit_calls.append(submission)
    self.late_policy_status = submission.get("late_policy_status")


class FakeAssignment:
  def __init__(
      self,
      *,
      assignment_id: int,
      due_at: str | None,
      submissions_by_user: dict[int, FakeSubmission],
      published: bool = True
  ):
    self.id = assignment_id
    self.name = f"Assignment {assignment_id}"
    self.due_at = due_at
    self.published = published
    self._submissions_by_user = submissions_by_user

  def get_submission(self, user_id: int) -> FakeSubmission:
    return self._submissions_by_user[user_id]


class FakeCourse:
  def __init__(self, assignments: list[FakeAssignment], students: list[FakeStudent]):
    self._assignments = assignments
    self._students = students

  def get_students(self, *, include_names: bool = False):
    return self._students

  def get_assignments(self, **kwargs):
    return self._assignments


def test_parse_canvas_datetime_handles_zulu_suffix():
  parsed = _parse_canvas_datetime("2026-02-16T12:34:56Z")
  assert parsed is not None
  assert parsed.tzinfo is not None
  assert parsed.utcoffset().total_seconds() == 0


def test_cleanup_missing_marks_missing_after_due_date():
  submission = FakeSubmission(late_policy_status="none")
  assignment = FakeAssignment(
    assignment_id=1,
    due_at="2026-02-01T00:00:00+00:00",
    submissions_by_user={101: submission},
  )
  course = FakeCourse([assignment], [FakeStudent(101)])

  stats = cleanup_missing_by_due_date(
    course,
    now=datetime(2026, 2, 16, tzinfo=timezone.utc),
  )

  assert submission.edit_calls == [{"late_policy_status": "missing"}]
  assert stats["updated_to_missing"] == 1
  assert stats["updated_to_none"] == 0


def test_cleanup_missing_clears_missing_before_due_date():
  submission = FakeSubmission(late_policy_status="missing")
  assignment = FakeAssignment(
    assignment_id=2,
    due_at="2026-03-01T00:00:00+00:00",
    submissions_by_user={102: submission},
  )
  course = FakeCourse([assignment], [FakeStudent(102)])

  stats = cleanup_missing_by_due_date(
    course,
    now=datetime(2026, 2, 16, tzinfo=timezone.utc),
  )

  assert submission.edit_calls == [{"late_policy_status": "none"}]
  assert stats["updated_to_none"] == 1
  assert stats["updated_to_missing"] == 0


def test_cleanup_missing_skips_submitted_work():
  submission = FakeSubmission(
    workflow_state="graded",
    submitted_at="2026-02-10T00:00:00+00:00",
    late_policy_status="missing",
    submission_type="online_upload",
    attachments=[{"id": 1}],
  )
  assignment = FakeAssignment(
    assignment_id=3,
    due_at="2026-03-01T00:00:00+00:00",
    submissions_by_user={103: submission},
  )
  course = FakeCourse([assignment], [FakeStudent(103)])

  stats = cleanup_missing_by_due_date(
    course,
    now=datetime(2026, 2, 16, tzinfo=timezone.utc),
  )

  assert submission.edit_calls == []
  assert stats["skipped_submitted"] == 1
  assert stats["submitted_with_content"] == 1


def test_cleanup_missing_treats_contentless_submitted_as_placeholder():
  submission = FakeSubmission(
    workflow_state="submitted",
    submitted_at="2026-02-10T00:00:00+00:00",
    late_policy_status="none",
    submission_type=None,
  )
  assignment = FakeAssignment(
    assignment_id=31,
    due_at="2026-02-01T00:00:00+00:00",
    submissions_by_user={1031: submission},
  )
  course = FakeCourse([assignment], [FakeStudent(1031)])

  stats = cleanup_missing_by_due_date(
    course,
    now=datetime(2026, 2, 16, tzinfo=timezone.utc),
  )

  assert submission.edit_calls == [{"late_policy_status": "missing"}]
  assert stats["placeholder_without_content"] == 1


def test_cleanup_missing_uses_submission_specific_due_date():
  submission = FakeSubmission(
    late_policy_status="none",
    cached_due_date="2026-02-01T00:00:00+00:00",
  )
  assignment = FakeAssignment(
    assignment_id=4,
    due_at="2026-03-01T00:00:00+00:00",
    submissions_by_user={104: submission},
  )
  course = FakeCourse([assignment], [FakeStudent(104)])

  stats = cleanup_missing_by_due_date(
    course,
    now=datetime(2026, 2, 16, tzinfo=timezone.utc),
  )

  assert submission.edit_calls == [{"late_policy_status": "missing"}]
  assert stats["updated_to_missing"] == 1


def test_cleanup_missing_dry_run_does_not_write_changes():
  submission = FakeSubmission(late_policy_status="none")
  assignment = FakeAssignment(
    assignment_id=5,
    due_at="2026-02-01T00:00:00+00:00",
    submissions_by_user={105: submission},
  )
  course = FakeCourse([assignment], [FakeStudent(105)])

  stats = cleanup_missing_by_due_date(
    course,
    dry_run=True,
    now=datetime(2026, 2, 16, tzinfo=timezone.utc),
  )

  assert submission.edit_calls == []
  assert stats["updated_to_missing"] == 1


def test_cleanup_missing_forces_none_when_missing_flag_is_stale():
  submission = FakeSubmission(
    late_policy_status="none",
    missing=True,
  )
  assignment = FakeAssignment(
    assignment_id=6,
    due_at="2026-03-01T00:00:00+00:00",
    submissions_by_user={106: submission},
  )
  course = FakeCourse([assignment], [FakeStudent(106)])

  stats = cleanup_missing_by_due_date(
    course,
    now=datetime(2026, 2, 16, tzinfo=timezone.utc),
  )

  assert submission.edit_calls == [{"late_policy_status": "none"}]
  assert stats["forced_none_due_stale_missing_flag"] == 1
  assert stats["updated_to_none"] == 1


def test_cleanup_missing_can_clear_placeholder_grade_in_dry_run():
  submission = FakeSubmission(
    late_policy_status="none",
    missing=False,
  )
  submission.grade = "incomplete"
  submission.posted_grade = "incomplete"
  submission.entered_grade = "incomplete"
  submission.score = 0
  assignment = FakeAssignment(
    assignment_id=7,
    due_at="2026-03-01T00:00:00+00:00",
    submissions_by_user={107: submission},
  )
  course = FakeCourse([assignment], [FakeStudent(107)])

  stats = cleanup_missing_by_due_date(
    course,
    clear_placeholder_grade=True,
    dry_run=True,
    now=datetime(2026, 2, 16, tzinfo=timezone.utc),
  )

  # Dry run should not write, but should classify and count.
  assert submission.edit_calls == []
  assert stats["placeholder_grade_needs_clear"] == 1
  assert stats["placeholder_grade_clear_attempted"] == 1
  assert stats["placeholder_grade_clear_succeeded"] == 1


def test_cleanup_missing_skips_excused_submissions():
  submission = FakeSubmission(
    excused=True,
    late_policy_status="missing",
  )
  assignment = FakeAssignment(
    assignment_id=8,
    due_at="2026-02-01T00:00:00+00:00",
    submissions_by_user={108: submission},
  )
  course = FakeCourse([assignment], [FakeStudent(108)])

  stats = cleanup_missing_by_due_date(
    course,
    clear_placeholder_grade=True,
    now=datetime(2026, 2, 16, tzinfo=timezone.utc),
  )

  assert submission.edit_calls == []
  assert stats["skipped_excused"] == 1


def test_cleanup_missing_limit_processes_only_first_n_students():
  submissions = {
    201: FakeSubmission(late_policy_status="none"),
    202: FakeSubmission(late_policy_status="none"),
    203: FakeSubmission(late_policy_status="none"),
  }
  assignment = FakeAssignment(
    assignment_id=9,
    due_at="2026-02-01T00:00:00+00:00",
    submissions_by_user=submissions,
  )
  course = FakeCourse([assignment], [FakeStudent(201), FakeStudent(202), FakeStudent(203)])

  stats = cleanup_missing_by_due_date(
    course,
    limit=2,
    now=datetime(2026, 2, 16, tzinfo=timezone.utc),
  )

  assert submissions[201].edit_calls == [{"late_policy_status": "missing"}]
  assert submissions[202].edit_calls == [{"late_policy_status": "missing"}]
  assert submissions[203].edit_calls == []
  assert stats["students_available"] == 3
  assert stats["student_limit"] == 2
  assert stats["students_considered"] == 2
