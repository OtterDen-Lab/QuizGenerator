#!env python
from __future__ import annotations

import abc
import dataclasses
import datetime
import enum
import importlib
import inspect
import io
import itertools
import logging
import os
import pathlib
import pkgutil
import random
import tempfile
import types
import uuid
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

import canvasapi.course
import canvasapi.quiz

import QuizGenerator.contentast as ca

log = logging.getLogger(__name__)


@dataclasses.dataclass
class QuestionComponents:
    """Bundle of question parts generated during construction."""
    body: ca.Element
    answers: List[ca.Answer]
    explanation: ca.Element


@dataclasses.dataclass(frozen=True)
class RegenerationFlags:
    """Minimal metadata needed to regenerate a question instance."""
    question_class_name: str
    generation_seed: Optional[int]
    question_version: str
    config_params: Dict[str, Any]
    context_extras: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class QuestionInstance:
    """Fully-instantiated question with content, answers, and regeneration metadata."""
    body: ca.Element
    explanation: ca.Element
    answers: List[ca.Answer]
    answer_kind: ca.Answer.CanvasAnswerKind
    can_be_numerical: bool
    value: float
    spacing: float
    topic: "Question.Topic"
    flags: RegenerationFlags


@dataclasses.dataclass
class QuestionContext:
  rng_seed: Optional[int]
  rng: random.Random
  data: MutableMapping[str, Any] | Mapping[str, Any] = dataclasses.field(default_factory=dict)
  frozen: bool = False
  question_cls: type | None = None

  def __getitem__(self, key: str) -> Any:
    if key == "rng_seed":
      return self.rng_seed
    if key == "rng":
      return self.rng
    return self.data[key]

  def __setitem__(self, key: str, value: Any) -> None:
    if self.frozen:
      raise TypeError("QuestionContext is frozen.")
    if key == "rng_seed":
      self.rng_seed = value
      return
    if key == "rng":
      self.rng = value
      return
    if isinstance(self.data, MappingProxyType):
      raise TypeError("QuestionContext is frozen.")
    self.data[key] = value

  def get(self, key: str, default: Any = None) -> Any:
    if key == "rng_seed":
      return self.rng_seed
    if key == "rng":
      return self.rng
    if hasattr(self.data, "get"):
      return self.data.get(key, default)
    return default

  def __contains__(self, key: object) -> bool:
    if key in ("rng_seed", "rng"):
      return True
    return key in self.data

  def __getattr__(self, name: str) -> Any:
    if name in self.data:
      return self.data[name]
    if self.question_cls is not None and hasattr(self.question_cls, name):
      raw_attr = inspect.getattr_static(self.question_cls, name)
      if isinstance(raw_attr, staticmethod):
        return raw_attr.__func__
      if isinstance(raw_attr, classmethod):
        return raw_attr.__func__.__get__(self.question_cls, self.question_cls)
      attr = getattr(self.question_cls, name)
      if callable(attr):
        return types.MethodType(attr, self)
      return attr
    raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

  def keys(self):
    return self.data.keys()

  def items(self):
    return self.data.items()

  def values(self):
    return self.data.values()

  def freeze(self) -> "QuestionContext":
    frozen_data = MappingProxyType(dict(self.data))
    return QuestionContext(
      rng_seed=self.rng_seed,
      rng=self.rng,
      data=frozen_data,
      frozen=True,
      question_cls=self.question_cls,
    )


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
        parse_spacing("SHORT") -> 4.0
        parse_spacing("NONE") -> 0.0
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
    # Note: Don't build context here - instantiate() handles it
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
        try:
          importlib.import_module(f"{package_prefix}.{module_name}")
        except ImportError as e:
          log.warning(
            f"Skipping module '{package_prefix}.{module_name}' due to import error: {e}"
          )

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
  1. Different across multiple builds (when the same Question instance is reused for multiple PDFs)
  2. Reproducible from QR code config_params

  The Problem:
  ------------
  When generating multiple PDFs, Quiz.from_yaml() creates Question instances ONCE.
  These instances are then built multiple times with different RNG seeds.
  If a question randomly selects an algorithm/policy in __init__(), all PDFs get the same choice
  because __init__() only runs once with an unseeded RNG.

  The Solution:
  -------------
  1. In __init__(): Register choices with fixed values (if provided) or None (for random)
  2. In _build_context(): Make random selections using the seeded RNG, store in config_params
  3. Result: Each build gets a different random choice, and it's captured for QR codes

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

      def _build_context(cls, rng_seed=None, **kwargs):
          self.rng.seed(rng_seed)
          # Get the choice (randomly selected or from config_params)
          self.scheduler_algorithm = self.get_choice('scheduler_kind', self.Kind)
          # ... rest of build logic
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
    Should be called in _build_context() AFTER seeding the RNG.

    Args:
        param_name: The parameter name registered earlier
        enum_class: The enum class to choose from

    Returns:
        The selected enum value (either fixed or randomly chosen)
    """
    choice_info = self._regenerable_choices.get(param_name)
    if choice_info is None:
      raise ValueError(f"Choice '{param_name}' not registered. Call register_choice() in __init__() first.")

    # Check for temporary fixed value (set during backoff loop in instantiate())
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

  def pre_instantiate(self, base_seed, **kwargs):
    if not (hasattr(self, '_regenerable_choices') and self._regenerable_choices):
      return
    choice_rng = random.Random(base_seed)
    for param_name, choice_info in self._regenerable_choices.items():
      if choice_info['fixed_value'] is None:
        enum_class = choice_info['enum_class']
        random_choice = choice_rng.choice(list(enum_class))
        # Temporarily set this as the fixed value so all builds use it
        choice_info['_temp_fixed_value'] = random_choice.name
        # Store in config_params
        self.config_params[param_name] = random_choice.name

  def post_instantiate(self, instance, **kwargs):
    if not (hasattr(self, '_regenerable_choices') and self._regenerable_choices):
      return
    for param_name, choice_info in self._regenerable_choices.items():
      if '_temp_fixed_value' in choice_info:
        del choice_info['_temp_fixed_value']

class Question(abc.ABC):
  AUTO_ENTRY_WARNINGS = True
  """
  Base class for all quiz questions with cross-format rendering support.

  CRITICAL: When implementing Question subclasses, ALWAYS use content AST elements
  for all content in _build_body() and _build_explanation() methods.

  NEVER create manual LaTeX, HTML, or Markdown strings. The content AST system
  ensures consistent rendering across PDF/LaTeX and Canvas/HTML formats.

  Primary extension points:
    - build(...): Simple path. Override to generate body + explanation in one place.
    - _build_context/_build_body/_build_explanation: Context path.
      Default _build_context is required for all questions.

  Required Class Attributes:
    - VERSION (str): Question version number (e.g., "1.0")
      Increment when RNG logic changes to ensure reproducibility

  Content AST Usage Examples:
    def _build_body(cls, context):
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
        return body

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
      * Changes to _build_explanation() only

  See existing questions in premade_questions/ for patterns and examples.
  """

  # Default version - subclasses should override this
  VERSION = "1.0"
  FREEZE_CONTEXT = False
  
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

    self.possible_variations = float('inf')

    self.rng_seed_offset = kwargs.get("rng_seed_offset", 0)

    # To be used throughout when generating random things
    self.rng = random.Random()

    # Track question-specific configuration parameters (excluding framework parameters)
    # These will be included in QR codes for exam regeneration
    framework_params = {
      'name', 'points_value', 'topic', 'spacing', 'num_subquestions',
      'rng_seed_offset', 'rng_seed', 'class', 'kwargs', 'kind'
    }
    self.config_params = {k: v for k, v in kwargs.items() if k not in framework_params}
  
  def instantiate(self, **kwargs) -> QuestionInstance:
    """
    Instantiate a question once, returning content, answers, and regeneration metadata.
    """
    # Generate the question, retrying with incremented seeds until we get an interesting one
    base_seed = kwargs.get("rng_seed", None)
    max_backoff_attempts = kwargs.get("max_backoff_attempts", None)
    build_kwargs = dict(kwargs)
    build_kwargs.pop("rng_seed", None)
    build_kwargs.pop("max_backoff_attempts", None)
    # Include config params so build() implementations can access YAML-provided settings.
    build_kwargs = {**self.config_params, **build_kwargs}

    # Pre-select any regenerable choices using the base seed
    # This ensures the policy/algorithm stays constant across backoff attempts
    self.pre_instantiate(base_seed, **kwargs)

    instance = None
    try:
      backoff_counter = 0
      is_interesting = False
      ctx = None
      while not is_interesting:
        if max_backoff_attempts is not None and backoff_counter >= max_backoff_attempts:
          raise RuntimeError(
            f"Exceeded max_backoff_attempts={max_backoff_attempts} for {self.__class__.__name__}"
          )
        # Increment seed for each backoff attempt to maintain deterministic behavior
        current_seed = None if base_seed is None else base_seed + backoff_counter
        ctx = self._build_context(
          rng_seed=current_seed,
          **self.config_params
        )
        is_interesting = self.is_interesting_ctx(ctx)
        backoff_counter += 1

      # Store the actual seed used and question metadata for QR code generation
      actual_seed = None if base_seed is None else base_seed + backoff_counter - 1

      # Keep instance rng in sync for any legacy usage.
      if isinstance(ctx, QuestionContext):
        self.rng = ctx.rng
      elif isinstance(ctx, dict) and "rng" in ctx:
        self.rng = ctx["rng"]

      components = self.__class__.build(
        rng_seed=current_seed,
        context=ctx,
        **build_kwargs
      )

      # Collect answers from explicit lists and inline AST
      inline_body_answers = self._collect_answers_from_ast(components.body)
      answers = self._merge_answers(
        components.answers,
        inline_body_answers
      )

      can_be_numerical = self._can_be_numerical_from_answers(answers)

      if self.AUTO_ENTRY_WARNINGS:
        warnings = self._entry_warnings_from_answers(answers)
        components.body = self._append_entry_warnings(components.body, warnings)

      config_params = dict(self.config_params)
      if isinstance(ctx, dict) and ctx.get("_config_params"):
        config_params.update(ctx.get("_config_params"))

      context_extras: Dict[str, Any] = {}
      if isinstance(ctx, QuestionContext):
        include_list = ctx.get("qr_include_list", None)
        if isinstance(include_list, (list, tuple)):
          for key in include_list:
            if key in ctx:
              context_extras[key] = ctx[key]
      elif isinstance(ctx, dict):
        include_list = ctx.get("qr_include_list", None)
        if isinstance(include_list, (list, tuple)):
          for key in include_list:
            if key in ctx:
              context_extras[key] = ctx[key]

      instance = QuestionInstance(
        body=components.body,
        explanation=components.explanation,
        answers=answers,
        answer_kind=self.answer_kind,
        can_be_numerical=can_be_numerical,
        value=self.points_value,
        spacing=self.spacing,
        topic=self.topic,
        flags=RegenerationFlags(
          question_class_name=self._get_registered_name(),
          generation_seed=actual_seed,
          question_version=self.VERSION,
          config_params=config_params,
          context_extras=context_extras
        )
      )
      return instance
    finally:
      self.post_instantiate(instance, **kwargs)

  def pre_instantiate(self, base_seed, **kwargs):
    pass

  def post_instantiate(self, instance, **kwargs):
    pass
   
  @classmethod
  def build(cls, *, rng_seed=None, context=None, **kwargs) -> QuestionComponents:
    """
    Build question content (body, answers, explanation) for a given seed.

    This should only generate content; metadata like points/spacing belong in instantiate().
    """
    if context is None:
      context = cls._coerce_context(
        cls._build_context(rng_seed=rng_seed, **kwargs),
        rng_seed=rng_seed
      )
    else:
      context = cls._coerce_context(context, rng_seed=rng_seed)

    if cls.FREEZE_CONTEXT:
      context = context.freeze()

    # Build body + explanation. Each may return just an Element or (Element, answers).
    body, body_answers = cls._normalize_build_output(cls._build_body(context))
    explanation, explanation_answers = cls._normalize_build_output(cls._build_explanation(context))

    # Collect inline answers from both body and explanation.
    inline_body_answers = cls._collect_answers_from_ast(body)
    inline_explanation_answers = cls._collect_answers_from_ast(explanation)

    answers = cls._merge_answers(
      body_answers,
      explanation_answers,
      inline_body_answers,
      inline_explanation_answers
    )

    return QuestionComponents(
      body=body,
      answers=answers,
      explanation=explanation
    )

  @classmethod
  def _coerce_context(cls, context, *, rng_seed=None) -> QuestionContext:
    if isinstance(context, QuestionContext):
      return context
    if isinstance(context, dict):
      ctx_seed = context.get("rng_seed", rng_seed)
      rng = context.get("rng") or random.Random(ctx_seed)
      ctx = QuestionContext(rng_seed=ctx_seed, rng=rng)
      for key, value in context.items():
        if key in ("rng_seed", "rng"):
          continue
        ctx.data[key] = value
      return ctx
    raise TypeError(f"Unsupported context type: {type(context)}")


  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs) -> QuestionContext:
    """
    Build the deterministic context for a question instance.

    Override to return a QuestionContext and avoid persistent self.* state.
    """
    rng = random.Random(rng_seed)
    return QuestionContext(
      rng_seed=rng_seed,
      rng=rng,
      question_cls=cls,
    )

  @classmethod
  def is_interesting_ctx(cls, context) -> bool:
    """Context-aware hook; defaults to existing is_interesting()."""
    return True

  @classmethod
  def _build_body(cls, context) -> ca.Element | Tuple[ca.Element, List[ca.Answer]]:
    """Context-aware body builder."""
    raise NotImplementedError("Questions must implement _build_body().")

  @classmethod
  def _build_explanation(cls, context) -> ca.Element | Tuple[ca.Element, List[ca.Answer]]:
    """Context-aware explanation builder."""
    raise NotImplementedError("Questions must implement _build_explanation().")

  @classmethod
  def _collect_answers_from_ast(cls, element: ca.Element) -> List[ca.Answer]:
    """Traverse AST and collect embedded Answer elements."""
    answers: List[ca.Answer] = []

    def visit(node):
      if node is None:
        return
      if isinstance(node, ca.Answer):
        answers.append(node)
        return
      if isinstance(node, ca.Table):
        if getattr(node, "headers", None):
          for header in node.headers:
            visit(header)
        for row in node.data:
          for cell in row:
            visit(cell)
        return
      if isinstance(node, ca.TableGroup):
        for _, table in node.tables:
          visit(table)
        return
      if isinstance(node, ca.Container):
        for child in node.elements:
          visit(child)
        return
      if isinstance(node, (list, tuple)):
        for child in node:
          visit(child)

    visit(element)
    return answers

  @classmethod
  def _merge_answers(cls, *answer_lists: List[ca.Answer]) -> List[ca.Answer]:
    """Merge answers while preserving order and removing duplicates by key/id."""
    merged: List[ca.Answer] = []
    seen: set[str] = set()

    for answers in answer_lists:
      for ans in answers:
        key = getattr(ans, "key", None)
        if key is None:
          key = str(id(ans))
        if key in seen:
          continue
        seen.add(key)
        merged.append(ans)
    return merged

  @classmethod
  def _entry_warnings_from_answers(cls, answers: List[ca.Answer]) -> List[str]:
    warnings: List[str] = []
    seen: set[str] = set()
    for answer in answers:
      warning = None
      if hasattr(answer.__class__, "get_entry_warning"):
        warning = answer.__class__.get_entry_warning()
      if not warning:
        continue
      if isinstance(warning, str):
        warning_list = [warning]
      else:
        warning_list = list(warning)
      for item in warning_list:
        if item and item not in seen:
          warnings.append(item)
          seen.add(item)
    return warnings

  @classmethod
  def _append_entry_warnings(cls, body: ca.Element, warnings: List[str]) -> ca.Element:
    if not warnings:
      return body
    notes_lines = ["**Notes for answer entry**", ""]
    notes_lines.extend(f"- {warning}" for warning in warnings)
    warning_elements = ca.OnlyHtml([ca.Text("\n".join(notes_lines))])
    if isinstance(body, ca.Container):
      body.add_element(warning_elements)
      return body
    return ca.Section([body, warning_elements])

  @classmethod
  def _can_be_numerical_from_answers(cls, answers: List[ca.Answer]) -> bool:
    return (
      len(answers) == 1
      and isinstance(answers[0], ca.AnswerTypes.Float)
    )

  @classmethod
  def _normalize_build_output(
    cls,
    result: ca.Element | Tuple[ca.Element, List[ca.Answer]]
  ) -> Tuple[ca.Element, List[ca.Answer]]:
    if isinstance(result, tuple):
      body, answers = result
      return body, list(answers or [])
    return result, []

  def _answers_for_canvas(
      self,
      answers: List[ca.Answer],
      can_be_numerical: bool
  ) -> Tuple[ca.Answer.CanvasAnswerKind, List[Dict[str, Any]]]:
    if len(answers) == 0:
      return (ca.Answer.CanvasAnswerKind.ESSAY, [])

    if can_be_numerical:
      return (
        ca.Answer.CanvasAnswerKind.NUMERICAL_QUESTION,
        list(itertools.chain(*[a.get_for_canvas(single_answer=True) for a in answers]))
      )

    return (
      self.answer_kind,
      list(itertools.chain(*[a.get_for_canvas() for a in answers]))
    )

  def _build_question_ast(self, instance: QuestionInstance) -> ca.Question:
    question_ast = ca.Question(
      body=instance.body,
      explanation=instance.explanation,
      value=instance.value,
      spacing=instance.spacing,
      topic=instance.topic,
      can_be_numerical=instance.can_be_numerical
    )

    # Attach regeneration metadata to the question AST
    question_ast.question_class_name = instance.flags.question_class_name
    question_ast.generation_seed = instance.flags.generation_seed
    question_ast.question_version = instance.flags.question_version
    question_ast.config_params = dict(instance.flags.config_params)
    question_ast.qr_context_extras = dict(instance.flags.context_extras)
    if hasattr(self, "layout_reserved_height"):
      question_ast.reserve_height_cm = self.layout_reserved_height

    return question_ast

  def refresh(self, rng_seed=None, *args, **kwargs):
    raise NotImplementedError("refresh() has been removed; use _build_context().")
    
  def is_interesting(self) -> bool:
    return True
  
  def get__canvas(self, course: canvasapi.course.Course, quiz : canvasapi.quiz.Quiz, interest_threshold=1.0, *args, **kwargs):
    # Instantiate once for both content and answers
    instance = self.instantiate(**kwargs)
    log.debug("got question instance")

    questionAST = self._build_question_ast(instance)
    question_type, answers = self._answers_for_canvas(
      instance.answers,
      instance.can_be_numerical
    )

    # Define a helper function for uploading images to canvas
    def image_upload(img_data) -> str:

      course.create_folder(f"{quiz.id}", parent_folder_path="Quiz Files")

      temp_dir = os.path.join(tempfile.gettempdir(), "quiz_canvas_uploads")
      os.makedirs(temp_dir, exist_ok=True)
      temp_file = tempfile.NamedTemporaryFile(
        mode="w+b",
        suffix=".png",
        delete=False,
        dir=temp_dir
      )

      try:
        temp_file.write(img_data.getbuffer())
        temp_file.flush()
        temp_file.seek(0)
        upload_success, f = course.upload(temp_file, parent_folder_path=f"Quiz Files/{quiz.id}")
      finally:
        temp_file.close()
        try:
          os.remove(temp_file.name)
        except OSError:
          log.warning(f"Failed to remove temp image {temp_file.name}")

      img_data.name = "img.png"
      log.debug("path: " + f"/courses/{course.id}/files/{f['id']}/preview")
      return f"/courses/{course.id}/files/{f['id']}/preview"

    # Render AST to HTML for Canvas
    question_html = questionAST.render(
      "html",
      upload_func=image_upload
    )
    explanation_html = questionAST.explanation.render(
      "html",
      upload_func=image_upload
    )

    # Build appropriate dictionary to send to canvas
    return {
      "question_name": f"{self.name} ({datetime.datetime.now().strftime('%m/%d/%y %H:%M:%S.%f')})",
      "question_text": question_html,
      "question_type": question_type.value,
      "points_possible": self.points_value,
      "answers": answers,
      "neutral_comments_html": explanation_html
    }
  
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

  def __init__(self, questions_in_group: List[Question], pick_once: bool, name: Optional[str] = None):
    self.questions = questions_in_group
    self.pick_once = pick_once
    self.name = name or "QuestionGroup"

    # Deterministic metadata without selecting a specific question.
    first_question = questions_in_group[0] if questions_in_group else None
    self.points_value = getattr(first_question, "points_value", 0)
    self.topic = getattr(first_question, "topic", Question.Topic.MISC)
    self.spacing = max((q.spacing for q in questions_in_group), default=0)
    self.possible_variations = float("inf")

    self._current_question: Optional[Question] = None

  def instantiate(self, *args, **kwargs):
    # Use a local RNG to avoid global side effects.
    rng = random.Random(kwargs.get("rng_seed", None))

    if not self.pick_once or self._current_question is None:
      self._current_question = rng.choice(self.questions)

  def __getattr__(self, name):
    # Avoid instantiating without a seed when accessing metadata.
    if name in {"points_value", "topic", "spacing", "name", "possible_variations"}:
      return getattr(self, name)

    if self._current_question is None:
      representative = self.questions[0] if self.questions else None
      if representative is None:
        raise AttributeError(
          f"Neither QuestionGroup nor Question has attribute '{name}'"
        )
      rep_attr = getattr(representative, name, None)
      if callable(rep_attr):
        def wrapped_method(*args, **kwargs):
          if self._current_question is None or not self.pick_once:
            self.instantiate(*args, **kwargs)
          return getattr(self._current_question, name)(*args, **kwargs)
        return wrapped_method
      if rep_attr is not None:
        return rep_attr

    attr = getattr(self._current_question, name)
    if callable(attr):
      def wrapped_method(*args, **kwargs):
        if self._current_question is None or not self.pick_once:
          self.instantiate(*args, **kwargs)
        return attr(*args, **kwargs)
      return wrapped_method

    return attr
