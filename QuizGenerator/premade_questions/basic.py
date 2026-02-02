#!env python
from __future__ import annotations

from typing import Tuple, List
import random

import logging

import QuizGenerator.contentast as ca
from QuizGenerator.question import Question, QuestionRegistry
from QuizGenerator.mixins import TableQuestionMixin

log = logging.getLogger(__name__)


@QuestionRegistry.register()
class FromText(Question):
  
  def __init__(self, *args, text, **kwargs):
    super().__init__(*args, **kwargs)
    self.text = text
    self.possible_variations = 1
  
  def _build_context(self, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    context["text"] = self.text
    return context
  
  def _build_body(self, context) -> Tuple[ca.Element, List[ca.Answer]]:
    return ca.Section([ca.Text(context["text"])]), []

  def _build_explanation(self, context) -> Tuple[ca.Element, List[ca.Answer]]:
    return ca.Section(), []


@QuestionRegistry.register()
class FromGenerator(FromText, TableQuestionMixin):
  
  def __init__(self, *args, generator=None, text=None, **kwargs):
    if generator is None and text is None:
      raise TypeError(f"Must supply either generator or text kwarg for {self.__class__.__name__}")
    
    if generator is None:
      generator = text
    
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

  def _build_context(self, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    # Preserve prior behavior for generators that use the global random module.
    random.seed(rng_seed)
    return context

  def _build_body(self, context) -> Tuple[ca.Element, List[ca.Answer]]:
    try:
      generated_content = self.generator()
      if isinstance(generated_content, ca.Section):
        body = generated_content
      elif isinstance(generated_content, str):
        body = ca.Section([ca.Text(generated_content)])
      else:
        body = ca.Section([ca.Text(str(generated_content))])
    except TypeError as e:
      log.error(f"Error generating from text: {e}")
      log.debug(self.generator_text)
      exit(8)

    return body, []
    
class TrueFalse(FromText):
  pass
