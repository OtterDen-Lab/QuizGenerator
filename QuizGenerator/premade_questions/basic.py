#!env python
from __future__ import annotations

from typing import Tuple, List

import logging

import QuizGenerator.contentast as ca
from QuizGenerator.question import Question, QuestionRegistry, QuestionComponents
from QuizGenerator.mixins import TableQuestionMixin

log = logging.getLogger(__name__)


@QuestionRegistry.register()
class FromText(Question):
  
  def __init__(self, *args, text, **kwargs):
    super().__init__(*args, **kwargs)
    self.text = text
    self.possible_variations = 1
  
  def _build_body(self, context) -> Tuple[ca.Element, List[ca.Answer]]:
    return ca.Section([ca.Text(self.text)]), []


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
      function_code = "import random\n" + function_code

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
    
  def build(self, *, rng_seed=None, context=None, **kwargs) -> QuestionComponents:
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

    explanation = self._build_explanation(context or {})
    answers = self._merge_answers(
      self._collect_answers_from_ast(body),
      self._collect_answers_from_ast(explanation)
    )
    return QuestionComponents(
      body=body,
      answers=answers,
      explanation=explanation
    )


class TrueFalse(FromText):
  pass
