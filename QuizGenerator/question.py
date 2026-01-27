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
import pprint
import random
import re
import uuid

import pypandoc
import yaml
from typing import List, Dict, Any, Tuple, Optional
import canvasapi.course, canvasapi.quiz

import QuizGenerator.contentast as ca
from QuizGenerator.performance import timer, PerformanceTracker

import logging
log = logging.getLogger(__name__)


@dataclasses.dataclass
class QuestionComponents:
    """Bundle of question parts generated during construction."""
    body: ca.Element
    answers: List[ca.Answer]
    explanation: ca.Element


# Spacing presets for questions
SPACING_PRESETS = {
    "NONE": 0,
    "SHORT": 4,
    "MEDIUM": 6,
    "LONG": 9,
    "PAGE": 99,  # Special value that will be handled during bin-packing
    "EXTRA_PAGE": 199,  # Special value that adds a full blank page after the question
}


def parse_spacing(spacing_value) -> float:
    """
    Parse spacing value from YAML config.

    Args:
        spacing_value: Either a preset name ("NONE", "SHORT", "LONG", "PAGE")
                      or a numeric value in cm

    Returns:
        Spacing in cm as a float

    Examples:
        parse_spacing("SHORT") -> 5.0
        parse_spacing("NONE") -> 1.0
        parse_spacing(3.5) -> 3.5
        parse_spacing("3.5") -> 3.5
    """
    if isinstance(spacing_value, str):
        # Check if it's a preset
        if spacing_value.upper() in SPACING_PRESETS:
            return float(SPACING_PRESETS[spacing_value.upper()])
        # Try to parse as a number
        try:
            return float(spacing_value)
        except ValueError:
            log.warning(f"Invalid spacing value '{spacing_value}', defaulting to 0")
            return 0.0
    elif isinstance(spacing_value, (int, float)):
        return float(spacing_value)
    else:
        log.warning(f"Invalid spacing type {type(spacing_value)}, defaulting to 0")
        return 0.0


class QuestionRegistry:
  _registry = {}
  _class_name_to_registered_name = {}  # Reverse mapping: ClassName -> registered_name
  _scanned = False

  @classmethod
  def register(cls, question_type=None):
    def decorator(subclass):
      # Use the provided name or fall back to the class name
      name = question_type.lower() if question_type else subclass.__name__.lower()
      cls._registry[name] = subclass

      # Build reverse mapping from class name to registered name
      # This allows looking up by class name when QR codes store the class name
      class_name = subclass.__name__.lower()
      cls._class_name_to_registered_name[class_name] = name

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
      # Try the reverse mapping from class name to registered name
      # This handles cases where QR codes store class name (e.g., "RNNForwardPass")
      # but the question is registered with a custom name (e.g., "cst463.rnn.forward-pass")
      if question_key in cls._class_name_to_registered_name:
        question_key = cls._class_name_to_registered_name[question_key]
        log.debug(f"Resolved class name '{question_type}' to registered name '{question_key}'")
      else:
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
            elif final_name in cls._class_name_to_registered_name:
              # Try the class name reverse mapping on the final part
              question_key = cls._class_name_to_registered_name[final_name]
              log.debug(f"Resolved class name '{final_name}' to registered name '{question_key}'")
            else:
              raise ValueError(f"Unknown question type: {question_type}")
          else:
            raise ValueError(f"Unknown question type: {question_type}")

    new_question : Question = cls._registry[question_key](**kwargs)
    # Note: Don't call refresh() here - it will be called by get_question()
    # Calling it here would consume RNG calls and break QR code regeneration
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

    # Load user-registered questions via entry points (Option 1: Robust PyPI approach)
    # Users can register custom questions in their package's pyproject.toml:
    # [project.entry-points."quizgenerator.questions"]
    # my_custom_question = "my_package.questions:CustomQuestion"
    try:
      # Python 3.10+ approach
      from importlib.metadata import entry_points
      eps = entry_points()
      # Handle both Python 3.10+ (dict-like) and 3.12+ (select method)
      if hasattr(eps, 'select'):
        question_eps = eps.select(group='quizgenerator.questions')
      else:
        question_eps = eps.get('quizgenerator.questions', [])

      for ep in question_eps:
        try:
          # Loading the entry point will trigger @QuestionRegistry.register() decorator
          ep.load()
          log.debug(f"Loaded custom question type from entry point: {ep.name}")
        except Exception as e:
          log.warning(f"Failed to load entry point '{ep.name}': {e}")
    except ImportError:
      # Python < 3.10 fallback using pkg_resources
      try:
        import pkg_resources
        for ep in pkg_resources.iter_entry_points('quizgenerator.questions'):
          try:
            ep.load()
            log.debug(f"Loaded custom question type from entry point: {ep.name}")
          except Exception as e:
            log.warning(f"Failed to load entry point '{ep.name}': {e}")
      except ImportError:
        # If pkg_resources isn't available either, just skip entry points
        log.debug("Entry points not supported (importlib.metadata and pkg_resources unavailable)")

    cls._scanned = True


class RegenerableChoiceMixin:
  """
  Mixin for questions that need to make random choices from enums/lists that are:
  1. Different across multiple refreshes (when the same Question instance is reused for multiple PDFs)
  2. Reproducible from QR code config_params

  The Problem:
  ------------
  When generating multiple PDFs, Quiz.from_yaml() creates Question instances ONCE.
  These instances are then refresh()ed multiple times with different RNG seeds.
  If a question randomly selects an algorithm/policy in __init__(), all PDFs get the same choice
  because __init__() only runs once with an unseeded RNG.

  The Solution:
  -------------
  1. In __init__(): Register choices with fixed values (if provided) or None (for random)
  2. In refresh(): Make random selections using the seeded RNG, store in config_params
  3. Result: Each refresh gets a different random choice, and it's captured for QR codes

  Usage Example:
  --------------
  class SchedulingQuestion(Question, RegenerableChoiceMixin):
      class Kind(enum.Enum):
          FIFO = enum.auto()
          SJF = enum.auto()

      def __init__(self, scheduler_kind=None, **kwargs):
          # Register the choice BEFORE calling super().__init__()
          self.register_choice('scheduler_kind', self.Kind, scheduler_kind, kwargs)
          super().__init__(**kwargs)

      def refresh(self, **kwargs):
          super().refresh(**kwargs)
          # Get the choice (randomly selected or from config_params)
          self.scheduler_algorithm = self.get_choice('scheduler_kind', self.Kind)
          # ... rest of refresh logic
  """

  def __init__(self, *args, **kwargs):
    # Initialize the choices registry if it doesn't exist
    if not hasattr(self, '_regenerable_choices'):
      self._regenerable_choices = {}
    super().__init__(*args, **kwargs)

  def register_choice(self, param_name: str, enum_class: type[enum.Enum], fixed_value: str | None, kwargs_dict: dict):
    """
    Register a choice parameter that needs to be regenerable.

    Args:
        param_name: The parameter name (e.g., 'scheduler_kind', 'policy')
        enum_class: The enum class to choose from (e.g., SchedulingQuestion.Kind)
        fixed_value: The fixed value if provided, or None for random selection
        kwargs_dict: The kwargs dictionary to update (for config_params capture)

    This should be called in __init__() BEFORE super().__init__().
    """
    # Store the enum class for later use
    if not hasattr(self, '_regenerable_choices'):
      self._regenerable_choices = {}

    self._regenerable_choices[param_name] = {
      'enum_class': enum_class,
      'fixed_value': fixed_value
    }

    # Add to kwargs so config_params captures it
    if fixed_value is not None:
      kwargs_dict[param_name] = fixed_value

  def get_choice(self, param_name: str, enum_class: type[enum.Enum]) -> enum.Enum:
    """
    Get the choice for a registered parameter.
    Should be called in refresh() AFTER super().refresh().

    Args:
        param_name: The parameter name registered earlier
        enum_class: The enum class to choose from

    Returns:
        The selected enum value (either fixed or randomly chosen)
    """
    choice_info = self._regenerable_choices.get(param_name)
    if choice_info is None:
      raise ValueError(f"Choice '{param_name}' not registered. Call register_choice() in __init__() first.")

    # Check for temporary fixed value (set during backoff loop in get_question())
    fixed_value = choice_info.get('_temp_fixed_value', choice_info['fixed_value'])

    # CRITICAL: Always consume an RNG call to keep RNG state synchronized between
    # original generation and QR code regeneration. During original generation,
    # we pick randomly. During regeneration, we already know the answer from
    # config_params, but we still need to consume the RNG call.
    enum_list = list(enum_class)
    random_choice = self.rng.choice(enum_list)

    if fixed_value is None:
      # No fixed value - use the random choice we just picked
      self.config_params[param_name] = random_choice.name
      return random_choice
    else:
      # Fixed value provided - ignore the random choice, use the fixed value
      # (but we still consumed the RNG call above to keep state synchronized)

      # If already an enum instance, return it directly
      if isinstance(fixed_value, enum_class):
        return fixed_value

      # If it's a string, look up the enum member by name
      if isinstance(fixed_value, str):
        try:
          # Try exact match first (handles "RoundRobin", "FIFO", etc.)
          return enum_class[fixed_value]
        except KeyError:
          # Try uppercase as fallback (handles "roundrobin" -> "ROUNDROBIN")
          try:
            return enum_class[fixed_value.upper()]
          except KeyError:
            log.warning(
              f"Invalid {param_name} '{fixed_value}'. Valid options are: {[k.name for k in enum_class]}. Defaulting to random"
            )
            self.config_params[param_name] = random_choice.name
            return random_choice

      # Unexpected type
      log.warning(
        f"Invalid {param_name} type {type(fixed_value)}. Expected enum or string. Defaulting to random"
      )
      self.config_params[param_name] = random_choice.name
      return random_choice


class Question(abc.ABC):
  """
  Base class for all quiz questions with cross-format rendering support.

  CRITICAL: When implementing Question subclasses, ALWAYS use content AST elements
  for all content in get_body() and get_explanation() methods.

  NEVER create manual LaTeX, HTML, or Markdown strings. The content AST system
  ensures consistent rendering across PDF/LaTeX and Canvas/HTML formats.

  Required Methods:
    - _get_body(): Return Tuple[ca.Section, List[ca.Answer]] with body and answers
    - _get_explanation(): Return Tuple[ca.Section, List[ca.Answer]] with explanation

  Note: get_body() and get_explanation() are provided for backward compatibility
  and call the _get_* methods, returning just the first element of the tuple.

  Required Class Attributes:
    - VERSION (str): Question version number (e.g., "1.0")
      Increment when RNG logic changes to ensure reproducibility

  Content AST Usage Examples:
    def _get_body(self):
        body = ca.Section()
        answers = []
        body.add_element(ca.Paragraph(["Calculate the matrix:"]))

        # Use ca.Matrix for math, NOT manual LaTeX
        matrix_data = [[1, 2], [3, 4]]
        body.add_element(ca.Matrix(data=matrix_data, bracket_type="b"))

        # Answer extends ca.Leaf - add directly to body
        ans = ca.Answer.integer("result", 42, label="Result")
        answers.append(ans)
        body.add_element(ans)
        return body, answers

  Common Content AST Elements:
    - ca.Paragraph: Text blocks
    - ca.Equation: Mathematical expressions
    - ca.Matrix: Matrices and vectors (use instead of manual LaTeX!)
    - ca.Table: Data tables
    - ca.OnlyHtml/OnlyLatex: Platform-specific content

  Versioning Guidelines:
    - Increment VERSION when changing:
      * Order of random number generation calls
      * Question generation logic
      * Answer calculation methods
    - Do NOT increment for:
      * Cosmetic changes (formatting, wording)
      * Bug fixes that don't affect answer generation
      * Changes to get_explanation() only

  See existing questions in premade_questions/ for patterns and examples.
  """

  # Default version - subclasses should override this
  VERSION = "1.0"
  
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
    self.spacing = parse_spacing(kwargs.get("spacing", 0))
    self.answer_kind = ca.Answer.CanvasAnswerKind.BLANK

    # Support for multi-part questions (defaults to 1 for normal questions)
    self.num_subquestions = kwargs.get("num_subquestions", 1)

    self.extra_attrs = kwargs # clear page, etc.

    self.answers = {}
    self.possible_variations = float('inf')

    self.rng_seed_offset = kwargs.get("rng_seed_offset", 0)

    # Component caching for unified Answer architecture
    self._components: QuestionComponents = None

    # To be used throughout when generating random things
    self.rng = random.Random()

    # Track question-specific configuration parameters (excluding framework parameters)
    # These will be included in QR codes for exam regeneration
    framework_params = {
      'name', 'points_value', 'topic', 'spacing', 'num_subquestions',
      'rng_seed_offset', 'rng_seed', 'class', 'kwargs', 'kind'
    }
    self.config_params = {k: v for k, v in kwargs.items() if k not in framework_params}
  
  @classmethod
  def from_yaml(cls, path_to_yaml):
    with open(path_to_yaml) as fid:
      question_dicts = yaml.safe_load_all(fid)
  
  def get_question(self, **kwargs) -> ca.Question:
    """
    Gets the question in AST format
    :param kwargs:
    :return: (ca.Question) Containing question.
    """
    # Generate the question, retrying with incremented seeds until we get an interesting one
    with timer("question_refresh", question_name=self.name, question_type=self.__class__.__name__):
      base_seed = kwargs.get("rng_seed", None)

      # Pre-select any regenerable choices using the base seed
      # This ensures the policy/algorithm stays constant across backoff attempts
      if hasattr(self, '_regenerable_choices') and self._regenerable_choices:
        # Seed a temporary RNG with the base seed to make the choices
        choice_rng = random.Random(base_seed)
        for param_name, choice_info in self._regenerable_choices.items():
          if choice_info['fixed_value'] is None:
            # No fixed value - pick randomly and store it as fixed for this get_question() call
            enum_class = choice_info['enum_class']
            random_choice = choice_rng.choice(list(enum_class))
            # Temporarily set this as the fixed value so all refresh() calls use it
            choice_info['_temp_fixed_value'] = random_choice.name
            # Store in config_params
            self.config_params[param_name] = random_choice.name

      backoff_counter = 0
      is_interesting = False
      while not is_interesting:
        # Increment seed for each backoff attempt to maintain deterministic behavior
        current_seed = None if base_seed is None else base_seed + backoff_counter
        # Pass config_params to refresh so custom kwargs from YAML are available
        self.refresh(rng_seed=current_seed, hard_refresh=(backoff_counter > 0), **self.config_params)
        is_interesting = self.is_interesting()
        backoff_counter += 1

      # Clear temporary fixed values
      if hasattr(self, '_regenerable_choices') and self._regenerable_choices:
        for param_name, choice_info in self._regenerable_choices.items():
          if '_temp_fixed_value' in choice_info:
            del choice_info['_temp_fixed_value']

    with timer("question_body", question_name=self.name, question_type=self.__class__.__name__):
      body = self.get_body()

    with timer("question_explanation", question_name=self.name, question_type=self.__class__.__name__):
      explanation = self.get_explanation()

    # Store the actual seed used and question metadata for QR code generation
    actual_seed = None if base_seed is None else base_seed + backoff_counter - 1
    question_ast = ca.Question(
      body=body,
      explanation=explanation,
      value=self.points_value,
      spacing=self.spacing,
      topic=self.topic,
      
      can_be_numerical=self.can_be_numerical()
    )

    # Attach regeneration metadata to the question AST
    # Use the registered name instead of class name for better QR code regeneration
    question_ast.question_class_name = self._get_registered_name()
    question_ast.generation_seed = actual_seed
    question_ast.question_version = self.VERSION
    # Make a copy of config_params so each question AST has its own
    # (important when the same Question instance is reused for multiple PDFs)
    question_ast.config_params = dict(self.config_params)

    return question_ast
   
  @abc.abstractmethod
  def get_body(self, **kwargs) -> ca.Section:
    """
    Gets the body of the question during generation
    :param kwargs:
    :return: (ca.Section) Containing question body
    """
    pass
  
  def get_explanation(self, **kwargs) -> ca.Section:
    """
    Gets the body of the question during generation (backward compatible wrapper).
    Calls _get_explanation() and returns just the explanation.
    :param kwargs:
    :return: (ca.Section) Containing question explanation or None
    """
    # Try new pattern first
    if hasattr(self, '_get_explanation') and callable(getattr(self, '_get_explanation')):
      explanation, _ = self._get_explanation()
      return explanation
    # Fallback: default explanation
    return ca.Section(
      [ca.Text("[Please reach out to your professor for clarification]")]
    )

  def _get_body(self) -> Tuple[ca.Element, List[ca.Answer]]:
    """
    Build question body and collect answers (new pattern).
    Questions should override this to return (body, answers) tuple.

    Returns:
        Tuple of (body_ast, answers_list)
    """
    # Fallback: call old get_body() and return empty answers
    body = self.get_body()
    return body, []

  def _get_explanation(self) -> Tuple[ca.Element, List[ca.Answer]]:
    """
    Build question explanation and collect answers (new pattern).
    Questions can override this to include answers in explanations.

    Returns:
        Tuple of (explanation_ast, answers_list)
    """
    return ca.Section(
      [ca.Text("[Please reach out to your professor for clarification]")]
    ), []

  def build_question_components(self, **kwargs) -> QuestionComponents:
    """
    Build question components (body, answers, explanation) in single pass.

    Calls _get_body() and _get_explanation() which return tuples of
    (content, answers).
    """
    # Build body with its answers
    body, body_answers = self._get_body()

    # Build explanation with its answers
    explanation, explanation_answers = self._get_explanation()

    # Combine all answers
    all_answers = body_answers + explanation_answers

    return QuestionComponents(
      body=body,
      answers=all_answers,
      explanation=explanation
    )

  def get_answers(self, *args, **kwargs) -> Tuple[ca.Answer.CanvasAnswerKind, List[Dict[str,Any]]]:
    """
    Return answers from cached components (new pattern) or self.answers dict (old pattern).
    """
    # Try component-based approach first (new pattern)
    if self._components is None:
      try:
        self._components = self.build_question_components()
      except Exception as e:
        # If component building fails, fall back to dict
        log.debug(f"Failed to build question components: {e}, falling back to dict")
        pass

    # Use components if available and non-empty
    if self._components is not None and len(self._components.answers) > 0:
      answers = self._components.answers
      if self.can_be_numerical():
        return (
          ca.Answer.CanvasAnswerKind.NUMERICAL_QUESTION,
          list(itertools.chain(*[a.get_for_canvas(single_answer=True) for a in answers]))
        )
      return (
        self.answer_kind,
        list(itertools.chain(*[a.get_for_canvas() for a in answers]))
      )

    # Fall back to dict pattern (old pattern)
    if len(self.answers.values()) > 0:
      if self.can_be_numerical():
        return (
          ca.Answer.CanvasAnswerKind.NUMERICAL_QUESTION,
          list(itertools.chain(*[a.get_for_canvas(single_answer=True) for a in self.answers.values()]))
        )
      return (
        self.answer_kind,
        list(itertools.chain(*[a.get_for_canvas() for a in self.answers.values()]))
      )

    return (ca.Answer.CanvasAnswerKind.ESSAY, [])
    
  def refresh(self, rng_seed=None, *args, **kwargs):
    """If it is necessary to regenerate aspects between usages, this is the time to do it.
    This base implementation simply resets everything.
    :param rng_seed: random number generator seed to use when regenerating question
    :param *args:
    :param **kwargs:
    :return: bool - True if the generated question is interesting, False otherwise
    """
    self.answers = {}
    self._components = None  # Clear component cache
    # Seed the RNG directly with the provided seed (no offset)
    self.rng.seed(rng_seed)
    # Note: We don't call is_interesting() here because child classes need to
    # generate their workloads first. Child classes should call it at the end
    # of their refresh() and return the result.
    return self.is_interesting()  # Default: assume interesting if no override
    
  def is_interesting(self) -> bool:
    return True
  
  def get__canvas(self, course: canvasapi.course.Course, quiz : canvasapi.quiz.Quiz, interest_threshold=1.0, *args, **kwargs):
    # Get the AST for the question
    with timer("question_get_ast", question_name=self.name, question_type=self.__class__.__name__):
      questionAST = self.get_question(**kwargs)
    log.debug("got question ast")
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
  
  def can_be_numerical(self):
    if (len(self.answers.values()) == 1
          and isinstance(list(self.answers.values())[0], ca.AnswerTypes.Float)
    ):
      return True
    return False

  def _get_registered_name(self) -> str:
    """
    Get the registered name for this question class.

    Returns the name used when registering the question with @QuestionRegistry.register(),
    which may be different from the class name (e.g., "cst463.rnn.forward-pass" vs "RNNForwardPass").

    This is used for QR code generation to ensure regeneration works correctly.
    Falls back to class name if not found in registry (shouldn't happen in practice).
    """
    class_name_lower = self.__class__.__name__.lower()
    registered_name = QuestionRegistry._class_name_to_registered_name.get(class_name_lower)

    if registered_name is None:
      # Fallback to class name if not found (shouldn't happen but be defensive)
      log.warning(f"Question {self.__class__.__name__} not found in registry reverse mapping, using class name")
      return self.__class__.__name__

    return registered_name

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
