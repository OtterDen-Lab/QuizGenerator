#!env python
from __future__ import annotations

import logging
import os
import random
from types import SimpleNamespace
from typing import List, Tuple

import QuizGenerator.contentast as ca
from QuizGenerator.mixins import TableQuestionMixin
from QuizGenerator.question import Question, QuestionRegistry

log = logging.getLogger(__name__)

ALLOW_GENERATOR = False
_GENERATOR_WARNING_EMITTED = False


def _generator_allowed() -> bool:
  if os.environ.get("QUIZGEN_ALLOW_GENERATOR", "").lower() in {"1", "true", "yes"}:
    return True
  return ALLOW_GENERATOR


def _warn_generator_enabled() -> None:
  global _GENERATOR_WARNING_EMITTED
  if not _GENERATOR_WARNING_EMITTED:
    log.warning(
      "FromGenerator is enabled. Generator code is executed with full Python permissions."
    )
    _GENERATOR_WARNING_EMITTED = True


@QuestionRegistry.register()
class FromText(Question):
  
  def __init__(self, *args, text, **kwargs):
    kwargs["text"] = text
    super().__init__(*args, **kwargs)
    self.text = text
    self.possible_variations = 1
  
  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    context["text"] = kwargs.get("text", "")
    return context
  
  @classmethod
  def _build_body(cls, context) -> Tuple[ca.Element, List[ca.Answer]]:
    return ca.Section([ca.Text(context["text"])]), []

  @classmethod
  def _build_explanation(cls, context) -> Tuple[ca.Element, List[ca.Answer]]:
    return ca.Section(), []


@QuestionRegistry.register()
class FromGenerator(FromText, TableQuestionMixin):
  
  def __init__(self, *args, generator=None, text=None, **kwargs):
    if not _generator_allowed():
      raise ValueError(
        "FromGenerator is disabled by default. "
        "Use --allow_generator or set QUIZGEN_ALLOW_GENERATOR=1 to enable."
      )
    _warn_generator_enabled()
    if generator is None and text is None:
      raise TypeError(f"Must supply either generator or text kwarg for {self.__class__.__name__}")
    
    if generator is None:
      generator = text
    
    kwargs["generator"] = generator
    super().__init__(*args, text="", **kwargs)
    self.possible_variations = kwargs.get("possible_variations", float('inf'))
    
    def attach_function_to_object(obj, function_code, function_name='get_body_lines'):
      # Provide a deterministic RNG handle for generator snippets.
      function_code = "rng = self.rng\n" + function_code

      # Create a local namespace for exec with content AST helpers available
      local_namespace = {
        'ca': ca,
        'Section': ca.Section,
        'Text': ca.Text,
        'Table': ca.Table,
        'Paragraph': ca.Paragraph
      }

      # Define the function dynamically using exec
      # Merge current globals with our local namespace for the exec
      exec_globals = {**globals(), **local_namespace}
      exec(f"def {function_name}(self):\n" + "\n".join(f"    {line}" for line in function_code.splitlines()), exec_globals, local_namespace)

      # Get the function and bind it to the object
      function = local_namespace[function_name]
      setattr(obj, function_name, function.__get__(obj))
    
    self.generator_text = generator
    # Attach the function dynamically
    attach_function_to_object(self, generator, "generator")

  @staticmethod
  def _compile_generator(function_code, function_name="generator"):
    # Provide a deterministic RNG handle for generator snippets.
    function_code = "rng = self.rng\n" + function_code

    local_namespace = {
      'ca': ca,
      'Section': ca.Section,
      'Text': ca.Text,
      'Table': ca.Table,
      'Paragraph': ca.Paragraph
    }

    exec_globals = {**globals(), **local_namespace}
    exec(
      f"def {function_name}(self):\n" + "\n".join(f"    {line}" for line in function_code.splitlines()),
      exec_globals,
      local_namespace
    )
    return local_namespace[function_name]

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    for key, value in kwargs.items():
      if key not in context:
        context[key] = value
    # Preserve prior behavior for generators that use the global random module.
    random.seed(rng_seed)
    generator_text = kwargs.get("generator")
    if generator_text is not None:
      context["generator_text"] = generator_text
      context["generator_fn"] = cls._compile_generator(generator_text)
      context["generator_scope"] = SimpleNamespace(
        rng=context.rng,
        **context.data
      )
    return context

  @classmethod
  def _build_body(cls, context) -> Tuple[ca.Element, List[ca.Answer]]:
    try:
      generator_fn = context.get("generator_fn")
      if generator_fn is None:
        raise TypeError("No generator provided for FromGenerator.")
      generated_content = generator_fn(context.get("generator_scope"))
      if isinstance(generated_content, ca.Section):
        body = generated_content
      elif isinstance(generated_content, str):
        body = ca.Section([ca.Text(generated_content)])
      else:
        body = ca.Section([ca.Text(str(generated_content))])
    except TypeError as e:
      log.error(f"Error generating from text: {e}")
      log.debug(context.get("generator_text"))
      exit(8)

    return body, []
    
class TrueFalse(FromText):
  pass
