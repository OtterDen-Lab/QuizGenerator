#!env python
from __future__ import annotations

import abc
import io
import dataclasses
import datetime
import enum
import importlib
import itertools
import os
import pathlib
import pkgutil
import random
import re
import uuid

import pypandoc
import yaml
from typing import List, Dict, Any, Tuple, Optional
import canvasapi.course, canvasapi.quiz

from QuizGenerator.misc import OutputFormat, Answer
from QuizGenerator.contentast import ContentAST
from QuizGenerator.performance import timer, PerformanceTracker

import logging
log = logging.getLogger(__name__)


class QuestionRegistry:
  _registry = {}
  _scanned = False
  
  @classmethod
  def register(cls, question_type=None):
    def decorator(subclass):
      # Use the provided name or fall back to the class name
      name = question_type.lower() if question_type else subclass.__name__.lower()
      cls._registry[name] = subclass
      return subclass
    return decorator
    
  @classmethod
  def create(cls, question_type, **kwargs) -> Question:
    """Instantiate a registered subclass."""
    # If we haven't already loaded our premades, do so now
    if not cls._scanned:
      cls.load_premade_questions()

    # Check to see if it's in the registry
    question_key = question_type.lower()
    if question_key not in cls._registry:
      # Try stripping common course prefixes and module paths for backward compatibility
      for prefix in ['cst334.', 'cst463.']:
        if question_key.startswith(prefix):
          stripped_name = question_key[len(prefix):]
          if stripped_name in cls._registry:
            question_key = stripped_name
            break
          # Also try extracting just the final class name after dots
          if '.' in stripped_name:
            final_name = stripped_name.split('.')[-1]
            if final_name in cls._registry:
              question_key = final_name
              break
      else:
        # As a final fallback, try just the last part after dots
        if '.' in question_key:
          final_name = question_key.split('.')[-1]
          if final_name in cls._registry:
            question_key = final_name
          else:
            raise ValueError(f"Unknown question type: {question_type}")
        else:
          raise ValueError(f"Unknown question type: {question_type}")

    new_question : Question = cls._registry[question_key](**kwargs)
    new_question.refresh()
    return new_question
    
    
  @classmethod
  def load_premade_questions(cls):
    package_name = "QuizGenerator.premade_questions"  # Fully qualified package name
    package_path = pathlib.Path(__file__).parent / "premade_questions"

    def load_modules_recursively(path, package_prefix):
      # Load modules from the current directory
      for _, module_name, _ in pkgutil.iter_modules([str(path)]):
        # Import the module
        module = importlib.import_module(f"{package_prefix}.{module_name}")

      # Recursively load modules from subdirectories
      for subdir in path.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('_'):
          subpackage_name = f"{package_prefix}.{subdir.name}"
          load_modules_recursively(subdir, subpackage_name)

    load_modules_recursively(package_path, package_name)
    cls._scanned = True


class Question(abc.ABC):
  """
  Base class for all quiz questions with cross-format rendering support.

  CRITICAL: When implementing Question subclasses, ALWAYS use ContentAST elements
  for all content in get_body() and get_explanation() methods.

  NEVER create manual LaTeX, HTML, or Markdown strings. The ContentAST system
  ensures consistent rendering across PDF/LaTeX and Canvas/HTML formats.

  Required Methods:
    - get_body(): Return ContentAST.Section with question content
    - get_explanation(): Return ContentAST.Section with solution steps

  ContentAST Usage Examples:
    def get_body(self):
        body = ContentAST.Section()
        body.add_element(ContentAST.Paragraph(["Calculate the matrix:"]))

        # Use ContentAST.Matrix for math, NOT manual LaTeX
        matrix_data = [[1, 2], [3, 4]]
        body.add_element(ContentAST.Matrix(data=matrix_data, bracket_type="b"))

        # Use ContentAST.Answer for input fields
        body.add_element(ContentAST.Answer(answer=self.answers["result"]))
        return body

  Common ContentAST Elements:
    - ContentAST.Paragraph: Text blocks
    - ContentAST.Equation: Mathematical expressions
    - ContentAST.Matrix: Matrices and vectors (use instead of manual LaTeX!)
    - ContentAST.Table: Data tables
    - ContentAST.OnlyHtml/OnlyLatex: Platform-specific content

  See existing questions in premade_questions/ for patterns and examples.
  """
  
  class Topic(enum.Enum):
    # CST334 (Operating Systems) Topics
    SYSTEM_MEMORY = enum.auto()      # Virtual memory, paging, segmentation, caching
    SYSTEM_PROCESSES = enum.auto()   # Process management, scheduling
    SYSTEM_CONCURRENCY = enum.auto() # Threads, synchronization, locks
    SYSTEM_IO = enum.auto()          # File systems, persistence, I/O operations
    SYSTEM_SECURITY = enum.auto()    # Access control, protection mechanisms

    # CST463 (Machine Learning/Data Science) Topics
    ML_OPTIMIZATION = enum.auto()    # Gradient descent, optimization algorithms
    ML_LINEAR_ALGEBRA = enum.auto()  # Matrix operations, vector mathematics
    ML_STATISTICS = enum.auto()      # Probability, distributions, statistical inference
    ML_ALGORITHMS = enum.auto()      # Classification, regression, clustering
    DATA_PREPROCESSING = enum.auto() # Data cleaning, transformation, feature engineering

    # General/Shared Topics
    MATH_GENERAL = enum.auto()       # Basic mathematics, calculus, algebra
    PROGRAMMING = enum.auto()        # General programming concepts
    LANGUAGES = enum.auto()          # Programming languages specifics
    MISC = enum.auto()              # Uncategorized questions

    # Legacy aliases for backward compatibility
    PROCESS = SYSTEM_PROCESSES
    MEMORY = SYSTEM_MEMORY
    CONCURRENCY = SYSTEM_CONCURRENCY
    IO = SYSTEM_IO
    SECURITY = SYSTEM_SECURITY
    MATH = MATH_GENERAL

    @classmethod
    def from_string(cls, string) -> Question.Topic:
      mappings = {
        member.name.lower() : member for member in cls
      }
      mappings.update({
        # Legacy mappings
        "processes": cls.SYSTEM_PROCESSES,
        "process": cls.SYSTEM_PROCESSES,
        "threads": cls.SYSTEM_CONCURRENCY,
        "concurrency": cls.SYSTEM_CONCURRENCY,
        "persistance": cls.SYSTEM_IO,
        "persistence": cls.SYSTEM_IO,
        "io": cls.SYSTEM_IO,
        "memory": cls.SYSTEM_MEMORY,
        "security": cls.SYSTEM_SECURITY,
        "math": cls.MATH_GENERAL,
        "mathematics": cls.MATH_GENERAL,

        # New mappings
        "optimization": cls.ML_OPTIMIZATION,
        "gradient_descent": cls.ML_OPTIMIZATION,
        "machine_learning": cls.ML_ALGORITHMS,
        "ml": cls.ML_ALGORITHMS,
        "linear_algebra": cls.ML_LINEAR_ALGEBRA,
        "matrix": cls.ML_LINEAR_ALGEBRA,
        "statistics": cls.ML_STATISTICS,
        "stats": cls.ML_STATISTICS,
        "data": cls.DATA_PREPROCESSING,
        "programming" : cls.PROGRAMMING,
        "misc": cls.MISC,
      })
      
      if string.lower() in mappings:
        return mappings.get(string.lower())
      return cls.MISC
  
  def __init__(self, name: str, points_value: float, topic: Question.Topic = Topic.MISC, *args, **kwargs):
    if name is None:
      name = self.__class__.__name__
    self.name = name
    self.points_value = points_value
    self.topic = topic
    self.spacing = kwargs.get("spacing", 0)
    self.answer_kind = Answer.AnswerKind.BLANK
    
    self.extra_attrs = kwargs # clear page, etc.
    
    self.answers = {}
    self.possible_variations = float('inf')
    
    self.rng_seed_offset = kwargs.get("rng_seed_offset", 0)
    
    # To be used throughout when generating random things
    self.rng = random.Random()
  
  @classmethod
  def from_yaml(cls, path_to_yaml):
    with open(path_to_yaml) as fid:
      question_dicts = yaml.safe_load_all(fid)
  
  def get_question(self, **kwargs) -> ContentAST.Question:
    """
    Gets the question in AST format
    :param kwargs:
    :return: (ContentAST.Question) Containing question.
    """
    # todo: would it make sense to refresh here?
    with timer("question_refresh", question_name=self.name, question_type=self.__class__.__name__):
      base_seed = kwargs.get("rng_seed", None)
      self.refresh(rng_seed=base_seed)
      backoff_counter = 1
      while not self.is_interesting():
        # Increment seed for each backoff attempt to maintain deterministic behavior
        backoff_seed = None if base_seed is None else base_seed + backoff_counter
        self.refresh(rng_seed=backoff_seed, hard_refresh=False)
        backoff_counter += 1

    with timer("question_body", question_name=self.name, question_type=self.__class__.__name__):
      body = self.get_body()

    with timer("question_explanation", question_name=self.name, question_type=self.__class__.__name__):
      explanation = self.get_explanation()

    return ContentAST.Question(
      body=body,
      explanation=explanation,
      value=self.points_value,
      spacing=self.spacing,
      topic=self.topic
    )
  
  @abc.abstractmethod
  def get_body(self, **kwargs) -> ContentAST.Section:
    """
    Gets the body of the question during generation
    :param kwargs:
    :return: (ContentAST.Section) Containing question body
    """
    pass
  
  def get_explanation(self, **kwargs) -> ContentAST.Section:
    """
    Gets the body of the question during generation
    :param kwargs:
    :return: (ContentAST.Section) Containing question explanation or None
    """
    return ContentAST.Section(
      [ContentAST.Text("[Please reach out to your professor for clarification]")]
    )
  
  def get_answers(self, *args, **kwargs) -> Tuple[Answer.AnswerKind, List[Dict[str,Any]]]:
    return (
      self.answer_kind,
      list(itertools.chain(*[a.get_for_canvas() for a in self.answers.values()]))
    )

  def refresh(self, rng_seed=None, *args, **kwargs):
    """If it is necessary to regenerate aspects between usages, this is the time to do it.
    This base implementation simply resets everything.
    :param rng_seed: random number generator seed to use when regenerating question
    :param *args:
    :param **kwargs:
    """
    self.answers = {}
    # todo: maybe have it randomly generate a seed every time, or use the time, and return this
    self.rng.seed(None if rng_seed is None else rng_seed + self.rng_seed_offset)
    # self.rng.seed(self.rng_seed_offset + (rng_seed or 0))
    
  def is_interesting(self) -> bool:
    return True
  
  def get__canvas(self, course: canvasapi.course.Course, quiz : canvasapi.quiz.Quiz, interest_threshold=1.0, *args, **kwargs):

    # Get the AST for the question
    with timer("question_get_ast", question_name=self.name, question_type=self.__class__.__name__):
      questionAST = self.get_question(**kwargs)

    # Get the answers and type of question
    question_type, answers = self.get_answers(*args, **kwargs)

    # Define a helper function for uploading images to canvas
    def image_upload(img_data) -> str:

      course.create_folder(f"{quiz.id}", parent_folder_path="Quiz Files")
      file_name = f"{uuid.uuid4()}.png"

      with io.FileIO(file_name, 'w+') as ffid:
        ffid.write(img_data.getbuffer())
        ffid.flush()
        ffid.seek(0)
        upload_success, f = course.upload(ffid, parent_folder_path=f"Quiz Files/{quiz.id}")
      os.remove(file_name)

      img_data.name = "img.png"
      # upload_success, f = course.upload(img_data, parent_folder_path=f"Quiz Files/{quiz.id}")
      log.debug("path: " + f"/courses/{course.id}/files/{f['id']}/preview")
      return f"/courses/{course.id}/files/{f['id']}/preview"

    # Render AST to HTML for Canvas
    with timer("ast_render_body", question_name=self.name, question_type=self.__class__.__name__):
      question_html = questionAST.render("html", upload_func=image_upload)

    with timer("ast_render_explanation", question_name=self.name, question_type=self.__class__.__name__):
      explanation_html = questionAST.explanation.render("html", upload_func=image_upload)

    # Build appropriate dictionary to send to canvas
    return {
      "question_name": f"{self.name} ({datetime.datetime.now().strftime('%m/%d/%y %H:%M:%S.%f')})",
      "question_text": question_html,
      "question_type": question_type.value,
      "points_possible": self.points_value,
      "answers": answers,
      "neutral_comments_html": explanation_html
    }

class QuestionGroup():
  
  def __init__(self, questions_in_group: List[Question], pick_once : bool):
    self.questions = questions_in_group
    self.pick_once = pick_once
  
    self._current_question : Optional[Question] = None
    
  def instantiate(self, *args, **kwargs):
    
    # todo: Make work with rng_seed (or at least verify)
    random.seed(kwargs.get("rng_seed", None))
    
    if not self.pick_once or self._current_question is None:
      self._current_question = random.choice(self.questions)
    
  def __getattr__(self, name):
    if self._current_question is None or name == "generate":
      self.instantiate()
    try:
      attr = getattr(self._current_question, name)
    except AttributeError:
      raise AttributeError(
        f"Neither QuestionGroup nor Question has attribute '{name}'"
      )
    
    if callable(attr):
      def wrapped_method(*args, **kwargs):
        return attr(*args, **kwargs)
      return wrapped_method
    
    return attr