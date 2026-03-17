import math
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-test-cache")

import QuizGenerator.generation.contentast as ca
from QuizGenerator.generation.premade_questions.cst334.process import SchedulingQuestion


def test_fifo_event_driven_simulation_matches_expected_values():
  jobs = [
    SchedulingQuestion.Job(job_id=0, arrival_time=0, duration=8),
    SchedulingQuestion.Job(job_id=1, arrival_time=1, duration=4),
    SchedulingQuestion.Job(job_id=2, arrival_time=2, duration=2),
  ]

  SchedulingQuestion.run_simulation(
    jobs_to_run=jobs,
    selector=(lambda j, curr_time: (j.arrival_time, j.job_id)),
    preemptable=False,
    scheduler_algorithm=SchedulingQuestion.Kind.FIFO
  )

  expected = {
    0: {"response": 0.0, "tat": 8.0},
    1: {"response": 7.0, "tat": 11.0},
    2: {"response": 10.0, "tat": 12.0},
  }
  for job in jobs:
    assert math.isclose(job.response_time, expected[job.job_id]["response"])
    assert math.isclose(job.turnaround_time, expected[job.job_id]["tat"])


def test_stcf_event_driven_simulation_matches_expected_values():
  jobs = [
    SchedulingQuestion.Job(job_id=0, arrival_time=0, duration=8),
    SchedulingQuestion.Job(job_id=1, arrival_time=1, duration=4),
    SchedulingQuestion.Job(job_id=2, arrival_time=2, duration=2),
  ]

  SchedulingQuestion.run_simulation(
    jobs_to_run=jobs,
    selector=(lambda j, curr_time: (j.time_remaining(curr_time), j.job_id)),
    preemptable=True,
    scheduler_algorithm=SchedulingQuestion.Kind.ShortestTimeRemaining
  )

  expected = {
    0: {"response": 0.0, "tat": 14.0},
    1: {"response": 0.0, "tat": 6.0},
    2: {"response": 0.0, "tat": 2.0},
  }
  for job in jobs:
    assert math.isclose(job.response_time, expected[job.job_id]["response"])
    assert math.isclose(job.turnaround_time, expected[job.job_id]["tat"])


def test_round_robin_event_driven_simulation_matches_exact_values():
  jobs = [
    SchedulingQuestion.Job(job_id=0, arrival_time=14, duration=8),
    SchedulingQuestion.Job(job_id=1, arrival_time=13, duration=3),
    SchedulingQuestion.Job(job_id=2, arrival_time=4, duration=10),
    SchedulingQuestion.Job(job_id=3, arrival_time=9, duration=4),
  ]

  SchedulingQuestion.run_simulation(
    jobs_to_run=jobs,
    selector=(lambda j, curr_time: (j.last_run, j.job_id)),
    preemptable=True,
    scheduler_algorithm=SchedulingQuestion.Kind.RoundRobin
  )

  expected_tats = {
    0: 15.0,
    1: 32 / 3,
    2: 59 / 3,
    3: 35 / 3,
  }
  expected_average_tat = 57 / 4

  for job in jobs:
    assert job.response_time == 0.0
    assert math.isclose(job.turnaround_time, expected_tats[job.job_id], rel_tol=0, abs_tol=1e-12)

  average_tat = sum(job.turnaround_time for job in jobs) / len(jobs)
  assert math.isclose(average_tat, expected_average_tat, rel_tol=0, abs_tol=1e-12)


def test_round_robin_canvas_answers_round_up_repeating_sixths():
  rr_denominators = SchedulingQuestion._round_robin_tat_fraction_denominators({"num_jobs": 4})

  accepted = ca.Answer.accepted_strings(
    32 / 3,
    allowed_fraction_denominators=rr_denominators,
  )

  assert "10.6667" in accepted
  assert "10.6666" not in accepted
