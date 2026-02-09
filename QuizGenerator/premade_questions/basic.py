#!/usr/bin/env python
from __future__ import annotations

import logging
import os
import random
import sys
from types import SimpleNamespace

import QuizGenerator.contentast as ca
from QuizGenerator import yaml_question
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
    # Print prominent warning to stderr (not just log)
    print("\n" + "=" * 70, file=sys.stderr)
    print("WARNING: FromGenerator executes arbitrary Python code from YAML.", file=sys.stderr)
    print("Only use this with YAML files you trust completely.", file=sys.stderr)
    print("=" * 70 + "\n", file=sys.stderr)
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
  def _build_body(cls, context) -> tuple[ca.Element, list[ca.Answer]]:
    return ca.Section([ca.Text(context["text"])]), []

  @classmethod
  def _build_explanation(cls, context) -> tuple[ca.Element, list[ca.Answer]]:
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

      random_proxy = _LocalRandomProxy(obj.rng)
      # Define the function dynamically using exec
      # Merge current globals with our local namespace for the exec
      exec_globals = {**globals(), **local_namespace, "random": random_proxy}
      exec(f"def {function_name}(self):\n" + "\n".join(f"    {line}" for line in function_code.splitlines()), exec_globals, local_namespace)

      # Get the function and bind it to the object
      function = local_namespace[function_name]
      setattr(obj, function_name, function.__get__(obj))
    
    self.generator_text = generator
    # Attach the function dynamically
    attach_function_to_object(self, generator, "generator")

  @staticmethod
  def _compile_generator(function_code, function_name="generator", rng_seed=None):
    rng = random.Random(rng_seed)
    random_proxy = _LocalRandomProxy(rng)

    # Provide a deterministic RNG handle for generator snippets.
    function_code = "rng = self.rng\n" + function_code

    local_namespace = {
      'ca': ca,
      'Section': ca.Section,
      'Text': ca.Text,
      'Table': ca.Table,
      'Paragraph': ca.Paragraph
    }

    exec_globals = {**globals(), **local_namespace, "random": random_proxy}
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
    # Provide a local RNG for generator snippets without touching global state.
    generator_text = kwargs.get("generator")
    if generator_text is not None:
      context["generator_text"] = generator_text
      context["generator_fn"] = cls._compile_generator(generator_text, rng_seed=rng_seed)
      context["generator_scope"] = SimpleNamespace(
        rng=context.rng,
        **context.data
      )
    return context

  @classmethod
  def _build_body(cls, context) -> tuple[ca.Element, list[ca.Answer]]:
    generator_fn = context.get("generator_fn")
    if generator_fn is None:
      raise ValueError("No generator provided for FromGenerator.")

    try:
      generated_content = generator_fn(context.get("generator_scope"))
    except Exception as e:
      generator_text = context.get("generator_text", "<unknown>")
      # Truncate long generator text for readability
      if len(generator_text) > 200:
        generator_text = generator_text[:200] + "..."
      raise RuntimeError(
        f"FromGenerator failed to execute:\n"
        f"  Error: {type(e).__name__}: {e}\n"
        f"  Generator code:\n    {generator_text.replace(chr(10), chr(10) + '    ')}"
      ) from e

    if isinstance(generated_content, ca.Section):
      body = generated_content
    elif isinstance(generated_content, str):
      body = ca.Section([ca.Text(generated_content)])
    else:
      body = ca.Section([ca.Text(str(generated_content))])

    return body, []

class _LocalRandomProxy:
  """Proxy random-like functions to a local RNG while preserving module attributes."""

  def __init__(self, rng: random.Random):
    self._rng = rng

  def __getattr__(self, name):
    if hasattr(self._rng, name):
      return getattr(self._rng, name)
    return getattr(random, name)


@QuestionRegistry.register()
class FromYaml(Question):
  def __init__(self, *args, yaml_path=None, yaml_text=None, yaml_spec=None, yaml_base_dir=None, **kwargs):
    if yaml_path is None and yaml_text is None and yaml_spec is None:
      raise TypeError(f"Must supply yaml_path, yaml_text, or yaml_spec for {self.__class__.__name__}")
    kwargs["yaml_path"] = yaml_path
    kwargs["yaml_text"] = yaml_text
    kwargs["yaml_spec"] = yaml_spec
    kwargs["yaml_base_dir"] = yaml_base_dir
    super().__init__(*args, **kwargs)

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    spec = yaml_question.load_question_spec(
      yaml_path=kwargs.get("yaml_path"),
      yaml_text=kwargs.get("yaml_text"),
      yaml_spec=kwargs.get("yaml_spec"),
      base_dir=kwargs.get("yaml_base_dir")
    )
    if kwargs.get("yaml_base_dir"):
      context["yaml_base_dir"] = kwargs.get("yaml_base_dir")
    context["yaml_spec"] = spec
    yaml_question.apply_context_spec(spec, context)
    templates = yaml_question.parse_question_templates(spec)
    context["yaml_body_template"] = templates["body"]
    context["yaml_explanation_template"] = templates["explanation"]
    return context

  @classmethod
  def _build_body(cls, context):
    return context.get("yaml_body_template", ca.Section())

  @classmethod
  def _build_explanation(cls, context):
    return context.get("yaml_explanation_template", ca.Section())
