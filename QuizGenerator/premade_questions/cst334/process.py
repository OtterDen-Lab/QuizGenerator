#!env python
from __future__ import annotations

import abc
import collections
import dataclasses
import enum
import io
import logging
import os
import random
import tempfile
import uuid
from typing import List

import matplotlib.pyplot as plt

import QuizGenerator.contentast as ca
from QuizGenerator.mixins import BodyTemplatesMixin, TableQuestionMixin
from QuizGenerator.question import Question, QuestionRegistry, RegenerableChoiceMixin

log = logging.getLogger(__name__)


class ProcessQuestion(Question, abc.ABC):
  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.PROCESS)
    super().__init__(*args, **kwargs)


@QuestionRegistry.register()
class SchedulingQuestion(ProcessQuestion, RegenerableChoiceMixin, TableQuestionMixin, BodyTemplatesMixin):
  class Kind(enum.Enum):
    FIFO = enum.auto()
    ShortestDuration = enum.auto()
    ShortestTimeRemaining = enum.auto()
    RoundRobin = enum.auto()

    def __str__(self):
      display_names = {
        self.FIFO: "First In First Out",
        self.ShortestDuration: "Shortest Job First",
        self.ShortestTimeRemaining: "Shortest Time To Completion",
        self.RoundRobin: "Round Robin"
      }
      return display_names.get(self, self.name)
  
  @staticmethod
  def get_kind_from_string(kind_str: str) -> SchedulingQuestion.Kind:
    try:
      return SchedulingQuestion.Kind[kind_str]
    except KeyError:
      return SchedulingQuestion.Kind.FIFO

  MAX_JOBS = 4
  MAX_ARRIVAL_TIME = 20
  MIN_JOB_DURATION = 2
  MAX_JOB_DURATION = 10
  
  ANSWER_EPSILON = 1.0
  
  scheduler_algorithm = None
  SELECTOR = None
  PREEMPTABLE = False
  TIME_QUANTUM = None
  
  ROUNDING_DIGITS = 2
  IMAGE_DPI = 140
  IMAGE_FIGSIZE = (9.5, 5.5)
  
  @dataclasses.dataclass
  class Job:
    job_id: int
    arrival_time: float
    duration: float
    elapsed_time: float = 0
    response_time: float = None
    turnaround_time: float = None
    unpause_time: float | None = None
    last_run: float = 0               # When were we last scheduled
    
    state_change_times: List[float] = dataclasses.field(default_factory=lambda: [])
    
    SCHEDULER_EPSILON = 1e-09
    
    def run(self, curr_time, is_rr=False) -> None:
      if self.response_time is None:
        # Then this is the first time running
        self.mark_start(curr_time)
      self.unpause_time = curr_time
      if not is_rr:
        self.state_change_times.append(curr_time)
    
    def stop(self, curr_time, is_rr=False) -> None:
      self.elapsed_time += (curr_time - self.unpause_time)
      if self.is_complete(curr_time):
        self.mark_end(curr_time)
      self.unpause_time = None
      self.last_run = curr_time
      if not is_rr:
        self.state_change_times.append(curr_time)
    
    def mark_start(self, curr_time) -> None:
      self.start_time = curr_time
      self.response_time = curr_time - self.arrival_time + self.SCHEDULER_EPSILON
    
    def mark_end(self, curr_time) -> None:
      self.end_time = curr_time
      self.turnaround_time = curr_time - self.arrival_time + self.SCHEDULER_EPSILON
    
    def time_remaining(self, curr_time) -> float:
      time_remaining = self.duration
      time_remaining -= self.elapsed_time
      if self.unpause_time is not None:
        time_remaining -= (curr_time - self.unpause_time)
      return time_remaining
    
    def is_complete(self, curr_time) -> bool:
      return self.duration <= self.elapsed_time + self.SCHEDULER_EPSILON # self.time_remaining(curr_time) <= 0
    
    def has_started(self) -> bool:
      return self.response_time is None
  
  @classmethod
  def get_workload(cls, rng, num_jobs, *args, **kwargs) -> List[SchedulingQuestion.Job]:
    """Makes a guaranteed interesting workload by following rules
    1. First job to arrive is the longest
    2. At least 2 other jobs arrive in its runtime
    3. Those jobs arrive in reverse length order, with the smaller arriving 2nd

    This will clearly show when jobs arrive how they are handled, since FIFO will be different than SJF, and STCF will cause interruptions
    """

    workload = []
    
    # First create a job that is relatively long-running and arrives first.
    # Set arrival time to something fairly low
    job0_arrival = rng.randint(0, int(0.25 * cls.MAX_ARRIVAL_TIME))
    # Set duration to something fairly long
    job0_duration = rng.randint(int(cls.MAX_JOB_DURATION * 0.75), cls.MAX_JOB_DURATION)
    
    # Next, let's create a job that will test whether we are preemptive or not.
    #  The core characteristics of this job are that it:
    #  1) would also finish _before_ the end of job0 if selected to run immediately.  This tests STCF
    # The bounds for arrival and completion will be:
    #  arrival:
    #   lower: (job0_arrival + 1) so we have a definite first job
    #   upper: (job0_arrival + job0_duration - self.MIN_JOB_DURATION) so we have enough time for a job to run
    #  duration:
    #   lower: self.MIN_JOB_DURATION
    #   upper:
    job1_arrival = rng.randint(
      job0_arrival + 1, # Make sure we start _after_ job0
      job0_arrival + job0_duration - cls.MIN_JOB_DURATION - 2 # Make sure we always have enough time for job1 & job2
    )
    job1_duration = rng.randint(
      cls.MIN_JOB_DURATION + 1, # default minimum and leave room for job2
      job0_arrival + job0_duration - job1_arrival - 1 # Make sure our job ends _at least_ before job0 would end
    )
    
    # Finally, we want to differentiate between STCF and SJF
    #  So, if we don't preempt job0 we want to make it be a tough choice between the next 2 jobs when it completes.
    #  This means we want a job that arrives _before_ job0 finishes, after job1 enters, and is shorter than job1
    job2_arrival = rng.randint(
      job1_arrival + 1, # Make sure we arrive after job1 so we subvert FIFO
      job0_arrival + job0_duration - 1 # ...but before job0 would exit the system
    )
    job2_duration = rng.randint(
      cls.MIN_JOB_DURATION, # Make sure it's at least the minimum.
      job1_duration - 1, # Make sure it's shorter than job1
    )
    
    # Package them up so we can add more jobs as necessary
    job_tuples = [
      (job0_arrival, job0_duration),
      (job1_arrival, job1_duration),
      (job2_arrival, job2_duration),
    ]
    
    # Add more jobs as necessary, if more than 3 are requested
    if num_jobs > 3:
      job_tuples.extend([
        (rng.randint(0, cls.MAX_ARRIVAL_TIME), rng.randint(cls.MIN_JOB_DURATION, cls.MAX_JOB_DURATION))
        for _ in range(num_jobs - 3)
      ])
    
    # Shuffle jobs so they are in a random order
    rng.shuffle(job_tuples)
    
    # Make workload from job tuples
    workload = []
    for i, (arr, dur) in enumerate(job_tuples):
      workload.append(
        SchedulingQuestion.Job(
          job_id=i,
          arrival_time=arr,
          duration=dur
        )
      )
    
    return workload
  
  @classmethod
  def run_simulation(
    cls,
    jobs_to_run: List[SchedulingQuestion.Job],
    selector,
    preemptable,
    time_quantum=None,
    scheduler_algorithm=None
  ):
    curr_time = 0
    selected_job: SchedulingQuestion.Job | None = None

    timeline = collections.defaultdict(list)
    timeline[curr_time].append("Simulation Start")
    for job in jobs_to_run:
      timeline[job.arrival_time].append(f"Job{job.job_id} arrived")

    while len(jobs_to_run) > 0:
      possible_time_slices = []

      # Get the jobs currently in the system
      available_jobs = list(filter(
        (lambda j: j.arrival_time <= curr_time),
        jobs_to_run
      ))

      # Get the jobs that will enter the system in the future
      future_jobs : List[SchedulingQuestion.Job] = list(filter(
        (lambda j: j.arrival_time > curr_time),
        jobs_to_run
      ))

      # Check whether there are jobs in the system already
      if len(available_jobs) > 0:
        # Use the selector to identify what job we are going to run
        selected_job : SchedulingQuestion.Job = min(
          available_jobs,
          key=(lambda j: selector(j, curr_time))
        )
        if selected_job.has_started():
          timeline[curr_time].append(
            f"Starting Job{selected_job.job_id} "
            f"(resp = {curr_time - selected_job.arrival_time:0.{cls.ROUNDING_DIGITS}f}s)"
          )
        # We start the job that we selected
        selected_job.run(curr_time, (scheduler_algorithm == cls.Kind.RoundRobin))

        # We could run to the end of the job
        possible_time_slices.append(selected_job.time_remaining(curr_time))

      # Check if we are preemptable or if we haven't found any time slices yet
      if preemptable or len(possible_time_slices) == 0:
        # Then when a job enters we could stop the current task
        if len(future_jobs) != 0:
          next_arrival : SchedulingQuestion.Job = min(
            future_jobs,
            key=(lambda j: j.arrival_time)
          )
          possible_time_slices.append((next_arrival.arrival_time - curr_time))

      if time_quantum is not None:
        possible_time_slices.append(time_quantum)

      ## Now we pick the minimum
      try:
        next_time_slice = min(possible_time_slices)
      except ValueError:
        log.error("No jobs available to schedule")
        break
      if scheduler_algorithm != SchedulingQuestion.Kind.RoundRobin:
        if selected_job is not None:
          timeline[curr_time].append(
            f"Running Job{selected_job.job_id} "
            f"for {next_time_slice:0.{cls.ROUNDING_DIGITS}f}s"
          )
        else:
          timeline[curr_time].append(f"(No job running)")
      curr_time += next_time_slice

      # We stop the job we selected, and potentially mark it as complete
      if selected_job is not None:
        selected_job.stop(curr_time, (scheduler_algorithm == cls.Kind.RoundRobin))
        if selected_job.is_complete(curr_time):
          timeline[curr_time].append(
            f"Completed Job{selected_job.job_id} "
            f"(TAT = {selected_job.turnaround_time:0.{cls.ROUNDING_DIGITS}f}s)"
          )
      selected_job = None

      # Filter out completed jobs
      jobs_to_run : List[SchedulingQuestion.Job] = list(filter(
        (lambda j: not j.is_complete(curr_time)),
        jobs_to_run
      ))
      if len(jobs_to_run) == 0:
        break

    return timeline
  
  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)
    num_jobs = kwargs.get("num_jobs", 3)

    scheduler_kind = kwargs.get("scheduler_kind")
    if scheduler_kind is None:
      scheduler_algorithm = rng.choice(list(cls.Kind))
      config_params = {"scheduler_kind": scheduler_algorithm.name}
    else:
      if isinstance(scheduler_kind, cls.Kind):
        scheduler_algorithm = scheduler_kind
      else:
        scheduler_algorithm = cls.get_kind_from_string(str(scheduler_kind))
      config_params = {"scheduler_kind": scheduler_algorithm.name}

    jobs = cls.get_workload(rng, num_jobs)

    match scheduler_algorithm:
      case SchedulingQuestion.Kind.ShortestDuration:
        timeline = cls.run_simulation(
          jobs_to_run=jobs,
          selector=(lambda j, curr_time: (j.duration, j.job_id)),
          preemptable=False,
          time_quantum=None,
          scheduler_algorithm=scheduler_algorithm
        )
      case SchedulingQuestion.Kind.ShortestTimeRemaining:
        timeline = cls.run_simulation(
          jobs_to_run=jobs,
          selector=(lambda j, curr_time: (j.time_remaining(curr_time), j.job_id)),
          preemptable=True,
          time_quantum=None,
          scheduler_algorithm=scheduler_algorithm
        )
      case SchedulingQuestion.Kind.RoundRobin:
        timeline = cls.run_simulation(
          jobs_to_run=jobs,
          selector=(lambda j, curr_time: (j.last_run, j.job_id)),
          preemptable=True,
          time_quantum=1e-05,
          scheduler_algorithm=scheduler_algorithm
        )
      case _:
        timeline = cls.run_simulation(
          jobs_to_run=jobs,
          selector=(lambda j, curr_time: (j.arrival_time, j.job_id)),
          preemptable=False,
          time_quantum=None,
          scheduler_algorithm=scheduler_algorithm
        )

    job_stats = {
      i : {
        "arrival_time" : job.arrival_time,
        "duration" : job.duration,
        "Response" : job.response_time,
        "TAT" : job.turnaround_time,
        "state_changes" : [job.arrival_time] + job.state_change_times + [job.arrival_time + job.turnaround_time],
      }
      for (i, job) in enumerate(jobs)
    }
    overall_stats = {
      "Response" : sum([job.response_time for job in jobs]) / len(jobs),
      "TAT" : sum([job.turnaround_time for job in jobs]) / len(jobs)
    }

    return {
      "num_jobs": num_jobs,
      "scheduler_algorithm": scheduler_algorithm,
      "job_stats": job_stats,
      "overall_stats": overall_stats,
      "timeline": timeline,
      "_config_params": config_params,
    }

  @classmethod
  def is_interesting_ctx(cls, context) -> bool:
    duration_sum = sum([context["job_stats"][job_id]['duration'] for job_id in context["job_stats"].keys()])
    tat_sum = sum([context["job_stats"][job_id]['TAT'] for job_id in context["job_stats"].keys()])
    return (tat_sum >= duration_sum * 1.1)
  
  @classmethod
  def _build_body(cls, context):
    """
    Build question body and collect answers.
    Returns:
        Tuple of (body_ast, answers_list)
    """
    answers: List[ca.Answer] = []

    # Create table data for scheduling results
    table_rows = []
    for job_id in sorted(context["job_stats"].keys()):
      response_answer = ca.AnswerTypes.Float(context["job_stats"][job_id]["Response"])
      tat_answer = ca.AnswerTypes.Float(context["job_stats"][job_id]["TAT"])
      answers.append(response_answer)
      answers.append(tat_answer)
      table_rows.append({
        "Job ID": f"Job{job_id}",
        "Arrival": context["job_stats"][job_id]["arrival_time"],
        "Duration": context["job_stats"][job_id]["duration"],
        "Response Time": response_answer,
        "TAT": tat_answer
      })

    # Create table using mixin
    scheduling_table = cls.create_answer_table(
      headers=["Job ID", "Arrival", "Duration", "Response Time", "TAT"],
      data_rows=table_rows,
      answer_columns=["Response Time", "TAT"]
    )

    # Collect average answers
    avg_response_answer = ca.AnswerTypes.Float(
      context["overall_stats"]["Response"],
      label="Overall average response time"
    )
    avg_tat_answer = ca.AnswerTypes.Float(
      context["overall_stats"]["TAT"],
      label="Overall average TAT"
    )
    answers.append(avg_response_answer)
    answers.append(avg_tat_answer)

    # Create average answer block
    average_block = ca.AnswerBlock([avg_response_answer, avg_tat_answer])

    # Use mixin to create complete body
    intro_text = (
      f"Given the below information, compute the required values if using **{context['scheduler_algorithm']}** scheduling. "
      f"Break any ties using the job number."
    )

    body = cls.create_fill_in_table_body(intro_text, None, scheduling_table)
    body.add_element(average_block)
    return body
  
  @classmethod
  def _build_explanation(cls, context):
    """
    Build question explanation.
    Returns:
        Tuple of (explanation_ast, answers_list)
    """
    explanation = ca.Section()

    explanation.add_element(
      ca.Paragraph([
        f"To calculate the overall Turnaround and Response times using {context['scheduler_algorithm']} "
        f"we want to first start by calculating the respective target and response times of all of our individual jobs."
      ])
    )

    explanation.add_elements([
      ca.Paragraph([
        "We do this by subtracting arrival time from either the completion time or the start time.  That is:"
        ]),
      ca.Equation(r"Job_{TAT} = Job_{completion} - Job_{arrival\_time}"),
      ca.Equation(r"Job_{response} = Job_{start} - Job_{arrival\_time}"),
    ])

    explanation.add_element(
      ca.Paragraph([
        f"For each of our {len(context['job_stats'].keys())} jobs, we can make these calculations.",
      ])
    )

    ## Add in TAT
    explanation.add_element(
      ca.Paragraph([
        "For turnaround time (TAT) this would be:"
      ] + [
        f"Job{job_id}_TAT "
        f"= {context['job_stats'][job_id]['arrival_time'] + context['job_stats'][job_id]['TAT']:0.{cls.ROUNDING_DIGITS}f} "
        f"- {context['job_stats'][job_id]['arrival_time']:0.{cls.ROUNDING_DIGITS}f} "
        f"= {context['job_stats'][job_id]['TAT']:0.{cls.ROUNDING_DIGITS}f}"
        for job_id in sorted(context['job_stats'].keys())
      ])
    )

    summation_line = ' + '.join([
      f"{context['job_stats'][job_id]['TAT']:0.{cls.ROUNDING_DIGITS}f}" for job_id in sorted(context['job_stats'].keys())
    ])
    explanation.add_element(
      ca.Paragraph([
        f"We then calculate the average of these to find the average TAT time",
        f"Avg(TAT) = ({summation_line}) / ({len(context['job_stats'].keys())}) "
        f"= {context['overall_stats']['TAT']:0.{cls.ROUNDING_DIGITS}f}",
      ])
    )


    ## Add in Response
    explanation.add_element(
      ca.Paragraph([
        "For response time this would be:"
      ] + [
      f"Job{job_id}_response "
      f"= {context['job_stats'][job_id]['arrival_time'] + context['job_stats'][job_id]['Response']:0.{cls.ROUNDING_DIGITS}f} "
      f"- {context['job_stats'][job_id]['arrival_time']:0.{cls.ROUNDING_DIGITS}f} "
      f"= {context['job_stats'][job_id]['Response']:0.{cls.ROUNDING_DIGITS}f}"
      for job_id in sorted(context['job_stats'].keys())
    ])
    )

    summation_line = ' + '.join([
      f"{context['job_stats'][job_id]['Response']:0.{cls.ROUNDING_DIGITS}f}" for job_id in sorted(context['job_stats'].keys())
    ])
    explanation.add_element(
      ca.Paragraph([
        f"We then calculate the average of these to find the average Response time",
        f"Avg(Response) "
        f"= ({summation_line}) / ({len(context['job_stats'].keys())}) "
        f"= {context['overall_stats']['Response']:0.{cls.ROUNDING_DIGITS}f}",
        "\n",
      ])
    )

    explanation.add_element(
      ca.Table(
        headers=["Time", "Events"],
        data=[
          [f"{t:02.{cls.ROUNDING_DIGITS}f}s"] + ['\n'.join(context['timeline'][t])]
          for t in sorted(context['timeline'].keys())
        ]
      )
    )

    explanation.add_element(
      ca.Picture(
        img_data=cls.make_image(context),
        caption="Process Scheduling Overview"
      )
    )

    return explanation
  
  @classmethod
  def make_image(cls, context):
    
    fig, ax = plt.subplots(1, 1, figsize=cls.IMAGE_FIGSIZE, dpi=cls.IMAGE_DPI)

    job_stats = context["job_stats"]
    scheduler_algorithm = context["scheduler_algorithm"]

    for x_loc in set([t for job_id in job_stats.keys() for t in job_stats[job_id]["state_changes"] ]):
      ax.axvline(x_loc, zorder=0)
      plt.text(x_loc + 0, len(job_stats.keys())-0.3, f'{x_loc:0.{cls.ROUNDING_DIGITS}f}s', rotation=90)

    if scheduler_algorithm != cls.Kind.RoundRobin:
      for y_loc, job_id in enumerate(sorted(job_stats.keys(), reverse=True)):
        for i, (start, stop) in enumerate(zip(job_stats[job_id]["state_changes"], job_stats[job_id]["state_changes"][1:])):
          ax.barh(
            y = [y_loc],
            left = [start],
            width = [stop - start],
            edgecolor='black',
            linewidth = 2,
            color = 'white' if (i % 2 == 1) else 'black'
          )
    else:
      job_deltas = collections.defaultdict(int)
      for job_id in job_stats.keys():
        job_deltas[job_stats[job_id]["state_changes"][0]] += 1
        job_deltas[job_stats[job_id]["state_changes"][1]] -= 1
      
      regimes_ranges = zip(sorted(job_deltas.keys()), sorted(job_deltas.keys())[1:])
      
      for (low, high) in regimes_ranges:
        jobs_in_range = [
          i for i, job_id in enumerate(list(job_stats.keys())[::-1])
          if
          (job_stats[job_id]["state_changes"][0] <= low)
          and
          (job_stats[job_id]["state_changes"][1] >= high)
        ]
        
        if len(jobs_in_range) == 0: continue
        
        ax.barh(
          y = jobs_in_range,
          left = [low for _ in jobs_in_range],
          width = [high - low for _ in jobs_in_range],
          color=f"{ 1 - ((len(jobs_in_range) - 1) / (len(job_stats.keys())))}",
        )
    
    # Plot the overall TAT
    ax.barh(
      y = [i for i in range(len(job_stats))][::-1],
      left = [job_stats[job_id]["arrival_time"] for job_id in sorted(job_stats.keys())],
      width = [job_stats[job_id]["TAT"] for job_id in sorted(job_stats.keys())],
      tick_label = [f"Job{job_id}" for job_id in sorted(job_stats.keys())],
      color=(0,0,0,0),
      edgecolor='black',
      linewidth=2,
    )
    
    ax.set_xlim(xmin=0)
    
    # Save to BytesIO object instead of a file
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png', dpi=cls.IMAGE_DPI, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    
    # Reset buffer position to the beginning
    buffer.seek(0)
    return buffer
    
  @classmethod
  def make_image_file(cls, context, image_dir=None):
    
    image_buffer = cls.make_image(context)
    
    # Original file-saving logic
    if image_dir is None:
      image_dir = os.path.join(tempfile.gettempdir(), "quiz_images")
    if not os.path.exists(image_dir):
      os.makedirs(image_dir)
    image_path = os.path.join(
      image_dir,
      f"{str(context['scheduler_algorithm']).replace(' ', '_')}-{uuid.uuid4()}.png"
    )

    with open(image_path, 'wb') as fid:
      fid.write(image_buffer.getvalue())
    return image_path
    

@QuestionRegistry.register()
class MLFQQuestion(ProcessQuestion, TableQuestionMixin, BodyTemplatesMixin):
  MIN_DURATION = 4
  MAX_DURATION = 12
  MIN_ARRIVAL = 0
  MAX_ARRIVAL = 10
  DEFAULT_NUM_JOBS = 3
  DEFAULT_NUM_QUEUES = 3
  ROUNDING_DIGITS = 2
  IMAGE_DPI = 140
  IMAGE_FIGSIZE = (9.5, 6.5)

  @dataclasses.dataclass
  class Job:
    job_id: int
    arrival_time: int
    duration: int
    remaining_time: int
    queue_level: int = 0
    time_in_queue: int = 0
    response_time: float | None = None
    turnaround_time: float | None = None
    remaining_quantum: int | None = None
    run_intervals: List[tuple] = dataclasses.field(default_factory=list)
    max_queue_level: int = 0

  @classmethod
  def get_workload(cls, rng, num_jobs: int, min_job_length: int, max_job_length: int) -> List[MLFQQuestion.Job]:
    arrivals = [0]
    if num_jobs > 1:
      arrivals.extend(
        rng.randint(cls.MIN_ARRIVAL, cls.MAX_ARRIVAL)
        for _ in range(num_jobs - 1)
      )
      if max(arrivals) == 0:
        arrivals[-1] = rng.randint(1, cls.MAX_ARRIVAL)

    durations = [
      rng.randint(min_job_length, max_job_length)
      for _ in range(num_jobs)
    ]

    jobs = []
    for i in range(num_jobs):
      jobs.append(
        MLFQQuestion.Job(
          job_id=i,
          arrival_time=arrivals[i],
          duration=durations[i],
          remaining_time=durations[i],
        )
      )
    return jobs

  @staticmethod
  def _normalize_queue_params(values: List[int] | None, num_queues: int) -> List[int]:
    if values is None:
      return []
    values = list(values)
    while len(values) < num_queues:
      values.append(values[-1])
    return values[:num_queues]

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)
    num_jobs = kwargs.get("num_jobs", cls.DEFAULT_NUM_JOBS)
    num_queues = kwargs.get("num_queues", cls.DEFAULT_NUM_QUEUES)
    min_job_length = kwargs.get("min_job_length", cls.MIN_DURATION)
    max_job_length = kwargs.get("max_job_length", cls.MAX_DURATION)
    boost_interval = kwargs.get("boost_interval", None)
    boost_interval_range = kwargs.get("boost_interval_range", None)

    if boost_interval is None and boost_interval_range:
      low, high = boost_interval_range
      boost_interval = rng.randint(low, high)

    jobs = cls.get_workload(rng, num_jobs, min_job_length, max_job_length)

    queue_quantums = [2**(num_queues - 1 - i) for i in range(num_queues)]
    queue_quantums = cls._normalize_queue_params(queue_quantums, num_queues)
    queue_quantums = [int(q) for q in queue_quantums]

    queue_allotments = [None] + [
      queue_quantums[i] * 2 for i in range(1, num_queues)
    ]
    queue_allotments = cls._normalize_queue_params(queue_allotments, num_queues)
    queue_allotments = [
      int(allotment) if allotment is not None else None
      for allotment in queue_allotments
    ]
    queue_allotments[0] = None

    timeline, boost_times, jobs = cls.run_simulation(
      jobs,
      queue_quantums,
      queue_allotments,
      boost_interval
    )

    job_stats = {
      job.job_id: {
        "arrival_time": job.arrival_time,
        "duration": job.duration,
        "Response": job.response_time,
        "TAT": job.turnaround_time,
        "run_intervals": list(job.run_intervals),
      }
      for job in jobs
    }

    return {
      "num_jobs": num_jobs,
      "num_queues": num_queues,
      "min_job_length": min_job_length,
      "max_job_length": max_job_length,
      "boost_interval": boost_interval,
      "queue_quantums": queue_quantums,
      "queue_allotments": queue_allotments,
      "timeline": timeline,
      "boost_times": boost_times,
      "job_stats": job_stats,
    }

  @classmethod
  def run_simulation(
    cls,
    jobs: List[MLFQQuestion.Job],
    queue_quantums: List[int],
    queue_allotments: List[int | None],
    boost_interval: int | None,
  ) -> None:
    timeline = collections.defaultdict(list)
    boost_times = []
    pending = sorted(jobs, key=lambda j: (j.arrival_time, j.job_id))
    queues = [collections.deque() for _ in range(len(queue_quantums))]
    completed = set()

    curr_time = pending[0].arrival_time if pending else 0
    timeline[curr_time].append("Simulation Start")
    next_boost_time = None
    if boost_interval is not None:
      next_boost_time = boost_interval

    def enqueue_arrivals(up_to_time: int) -> None:
      nonlocal pending
      while pending and pending[0].arrival_time <= up_to_time:
        job = pending.pop(0)
        job.queue_level = len(queues) - 1
        job.time_in_queue = 0
        job.remaining_quantum = None
        queues[-1].append(job)
        timeline[job.arrival_time].append(
          f"Job{job.job_id} arrived (dur = {job.duration})"
        )

    def apply_boost(curr_time: int, running_job: MLFQQuestion.Job | None = None) -> None:
      jobs_to_boost = []
      for q in queues:
        while q:
          jobs_to_boost.append(q.popleft())
      if running_job is not None and running_job.remaining_time > 0:
        jobs_to_boost.append(running_job)
      if not jobs_to_boost:
        boost_times.append(curr_time)
        return
      for job in sorted(jobs_to_boost, key=lambda j: j.job_id):
        job.queue_level = len(queues) - 1
        job.time_in_queue = 0
        job.remaining_quantum = None
        queues[-1].append(job)
      timeline[curr_time].append(
        f"Boosted all jobs to Q{len(queues) - 1}"
      )
      boost_times.append(curr_time)

    enqueue_arrivals(curr_time)

    while len(completed) < len(jobs):
      q_idx = next(
        (i for i in range(len(queues) - 1, -1, -1) if queues[i]),
        None
      )
      if q_idx is None:
        next_times = []
        if pending:
          next_times.append(pending[0].arrival_time)
        if next_boost_time is not None:
          next_times.append(next_boost_time)
        if next_times:
          next_time = min(next_times)
          if next_time > curr_time:
            timeline[curr_time].append("CPU idle")
          curr_time = next_time
          enqueue_arrivals(curr_time)
          while next_boost_time is not None and curr_time >= next_boost_time:
            apply_boost(curr_time)
            next_boost_time += boost_interval
          continue
        break

      job = queues[q_idx].popleft()
      quantum = queue_quantums[q_idx]
      if job.remaining_quantum is None or job.remaining_quantum <= 0:
        job.remaining_quantum = quantum

      slice_duration = min(job.remaining_quantum, job.remaining_time)
      preempted = False
      if q_idx < len(queues) - 1 and pending:
        next_arrival = pending[0].arrival_time
        if next_arrival < curr_time + slice_duration:
          slice_duration = next_arrival - curr_time
          preempted = True
      if next_boost_time is not None and next_boost_time < curr_time + slice_duration:
        slice_duration = next_boost_time - curr_time
        preempted = True

      if job.response_time is None:
        job.response_time = curr_time - job.arrival_time

      if slice_duration > 0:
        timeline[curr_time].append(
          f"Running Job{job.job_id} in Q{q_idx} for {slice_duration}"
        )
        job.run_intervals.append((curr_time, curr_time + slice_duration, q_idx))
        curr_time += slice_duration
        job.remaining_time -= slice_duration
        job.time_in_queue += slice_duration
        job.remaining_quantum -= slice_duration

      enqueue_arrivals(curr_time)
      boosted_current_job = False
      while next_boost_time is not None and curr_time >= next_boost_time:
        apply_boost(curr_time, running_job=job if not boosted_current_job else None)
        boosted_current_job = True
        next_boost_time += boost_interval

      if job.remaining_time <= 0:
        job.turnaround_time = curr_time - job.arrival_time
        job.remaining_quantum = None
        completed.add(job.job_id)
        timeline[curr_time].append(
          f"Completed Job{job.job_id} (TAT = {job.turnaround_time})"
        )
        continue
      if boosted_current_job:
        continue

      allotment = queue_allotments[q_idx]
      if (
        allotment is not None
        and job.time_in_queue >= allotment
        and q_idx > 0
      ):
        job.queue_level = q_idx - 1
        job.max_queue_level = max(job.max_queue_level, job.queue_level)
        job.time_in_queue = 0
        job.remaining_quantum = None
        queues[q_idx - 1].append(job)
        timeline[curr_time].append(
          f"Demoted Job{job.job_id} to Q{q_idx - 1}"
        )
        continue

      if preempted and job.remaining_quantum > 0:
        queues[q_idx].appendleft(job)
      else:
        if job.remaining_quantum <= 0:
          job.remaining_quantum = None
        queues[q_idx].append(job)

    return timeline, boost_times, jobs

  @classmethod
  def _build_body(cls, context):
    answers: List[ca.Answer] = []

    queue_rows = []
    for i in reversed(range(context["num_queues"])):
      allotment = context["queue_allotments"][i]
      queue_rows.append([
        f"Q{i}",
        context["queue_quantums"][i],
        "infinite" if allotment is None else allotment
      ])
    queue_table = ca.Table(
      headers=["Queue", "Quantum", "Allotment"],
      data=queue_rows
    )

    table_rows = []
    for job_id in sorted(context["job_stats"].keys()):
      tat_answer = ca.AnswerTypes.Float(context["job_stats"][job_id]["TAT"])
      answers.append(tat_answer)
      table_rows.append({
        "Job ID": f"Job{job_id}",
        "Arrival": context["job_stats"][job_id]["arrival_time"],
        "Duration": context["job_stats"][job_id]["duration"],
        "TAT": tat_answer,
      })

    scheduling_table = cls.create_answer_table(
      headers=["Job ID", "Arrival", "Duration", "TAT"],
      data_rows=table_rows,
      answer_columns=["TAT"]
    )

    intro_text = (
      "Assume an MLFQ scheduler with round-robin inside each queue. "
      f"New jobs enter the highest-priority queue (Q{context['num_queues'] - 1}) "
      "and a job is demoted after using its total allotment for that queue. "
      "If a higher-priority job arrives, it preempts any lower-priority job."
    )

    instructions = (
      f"Compute the turnaround time (TAT) for each job. "
      f"Round to at most {ca.Answer.DEFAULT_ROUNDING_DIGITS} digits after the decimal."
    )

    body = ca.Section()
    body.add_element(ca.Paragraph([intro_text]))
    body.add_element(queue_table)
    if context["boost_interval"] is not None:
      body.add_element(ca.Paragraph([
        f"Every {context['boost_interval']} time units, all jobs are boosted to "
        f"Q{context['num_queues'] - 1}. After a boost, scheduling restarts with the "
        "lowest job number in that queue."
      ]))
    body.add_element(ca.Paragraph([instructions]))
    body.add_element(scheduling_table)
    return body

  @classmethod
  def _build_explanation(cls, context):
    explanation = ca.Section()

    explanation.add_element(
      ca.Paragraph([
        "Turnaround time (TAT) is the completion time minus the arrival time.",
        "We calculate it for each job after simulating the schedule."
      ])
    )

    explanation.add_element(
      ca.Paragraph([
        "For each job:"
      ] + [
        f"Job{job_id}_TAT = "
        f"{context['job_stats'][job_id]['arrival_time'] + context['job_stats'][job_id]['TAT']:0.{cls.ROUNDING_DIGITS}f} "
        f"- {context['job_stats'][job_id]['arrival_time']:0.{cls.ROUNDING_DIGITS}f} "
        f"= {context['job_stats'][job_id]['TAT']:0.{cls.ROUNDING_DIGITS}f}"
        for job_id in sorted(context['job_stats'].keys())
      ])
    )

    explanation.add_element(
      ca.Table(
        headers=["Time", "Events"],
        data=[
          [f"{t:0.{cls.ROUNDING_DIGITS}f}s"] + ['\n'.join(events)]
          for t in sorted(context['timeline'].keys())
          if (events := [
            event for event in context['timeline'][t]
            if (
              "arrived" in event
              or "Demoted" in event
              or "Boosted" in event
              or "Completed" in event
              or "Simulation Start" in event
              or "CPU idle" in event
            )
          ])
        ]
      )
    )

    explanation.add_element(
      ca.Picture(
        img_data=cls.make_image(context),
        caption="MLFQ Scheduling Overview"
      )
    )

    return explanation

  @classmethod
  def make_image(cls, context):
    fig, ax = plt.subplots(1, 1, figsize=cls.IMAGE_FIGSIZE, dpi=cls.IMAGE_DPI)

    job_stats = context["job_stats"]
    num_jobs = len(job_stats)
    if num_jobs == 0:
      buffer = io.BytesIO()
      plt.tight_layout()
      plt.savefig(buffer, format='png', dpi=cls.IMAGE_DPI, bbox_inches='tight')
      plt.close(fig)
      buffer.seek(0)
      return buffer

    job_colors = {
      job_id: str(0.15 + 0.7 * (idx / max(1, num_jobs - 1)))
      for idx, job_id in enumerate(sorted(job_stats.keys()))
    }
    job_lane = {
      job_id: idx
      for idx, job_id in enumerate(sorted(job_stats.keys(), reverse=True))
    }
    lanes_per_queue = num_jobs

    for job_id in sorted(job_stats.keys()):
      for start, stop, queue_level in job_stats[job_id]["run_intervals"]:
        y_loc = queue_level * lanes_per_queue + job_lane[job_id]
        ax.barh(
          y=[y_loc],
          left=[start],
          width=[stop - start],
          edgecolor='black',
          linewidth=1.5,
          color=job_colors[job_id]
        )

    for queue_idx in range(context["num_queues"]):
      if queue_idx % 2 == 1:
        ax.axhspan(
          queue_idx * lanes_per_queue - 0.5,
          (queue_idx + 1) * lanes_per_queue - 0.5,
          facecolor='0.97',
          edgecolor='none',
          zorder=-1
        )

    arrival_times = sorted({
      job_stats[job_id]["arrival_time"]
      for job_id in job_stats.keys()
    })
    bottom_label_y = -0.1
    for arrival_time in arrival_times:
      ax.axvline(arrival_time, color='0.2', linestyle=':', linewidth=1.2, zorder=0)
      ax.text(
        arrival_time + 0.2,
        bottom_label_y,
        f"{arrival_time:0.{cls.ROUNDING_DIGITS}f}s",
        color='0.2',
        rotation=90,
        ha='left',
        va='bottom'
      )

    completion_times = sorted({
      job_stats[job_id]["arrival_time"] + job_stats[job_id]["TAT"]
      for job_id in job_stats.keys()
    })
    for completion_time in completion_times:
      ax.axvline(completion_time, color='red', linewidth=1.5, zorder=0)
      ax.text(
        completion_time - 0.6,
        context["num_queues"] * lanes_per_queue - 0.5,
        f"{completion_time:0.{cls.ROUNDING_DIGITS}f}s",
        color='red',
        rotation=90,
        ha='center',
        va='top'
      )

    for boost_time in sorted(set(context["boost_times"])):
      ax.axvline(boost_time, color='tab:blue', linestyle='--', linewidth=1.2, zorder=0)

    tick_positions = [
      q * lanes_per_queue + (lanes_per_queue - 1) / 2
      for q in range(context["num_queues"])
    ]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([f"Q{i}" for i in range(context["num_queues"])])
    ax.set_ylim(-0.5, context["num_queues"] * lanes_per_queue - 0.5)
    ax.set_xlim(xmin=0)
    ax.set_xlabel("Time")

    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png', dpi=cls.IMAGE_DPI, bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    return buffer
