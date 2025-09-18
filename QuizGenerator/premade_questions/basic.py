#!env python
from __future__ import annotations

from typing import List, Dict, Any, Tuple

import logging

from QuizGenerator.misc import ContentAST
from QuizGenerator.question import Question, QuestionRegistry, Answer
from QuizGenerator.mixins import TableQuestionMixin

log = logging.getLogger(__name__)


@QuestionRegistry.register()
class FromText(Question):
  
  def __init__(self, *args, text, **kwargs):
    super().__init__(*args, **kwargs)
    self.text = text
    self.answers = []
    self.possible_variations = 1
  
  def get_body(self, **kwargs) -> ContentAST.Section:
    return ContentAST.Section([ContentAST.Text(self.text)])
  
  def get_answers(self, *args, **kwargs) -> Tuple[Answer.AnswerKind, List[Dict[str,Any]]]:
    return Answer.AnswerKind.ESSAY, []


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

      # Create a local namespace for exec with ContentAST available
      local_namespace = {
        'ContentAST': ContentAST,
        'Section': ContentAST.Section,
        'Text': ContentAST.Text,
        'Table': ContentAST.Table,
        'Paragraph': ContentAST.Paragraph
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
    
    self.answers = {}


  def get_body(self, **kwargs) -> ContentAST.Section:
    return super().get_body()

  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)
    try:
      generated_content = self.generator()
      # Expect generator to return a ContentAST.Section or convert string to Section
      if isinstance(generated_content, ContentAST.Section):
        self.text = ""  # Clear text since we'll override get_body
        self._generated_section = generated_content
      elif isinstance(generated_content, str):
        self.text = generated_content
        self._generated_section = None
      else:
        # Fallback
        self.text = str(generated_content)
        self._generated_section = None
    except TypeError as e:
      log.error(f"Error generating from text: {e}")
      log.debug(self.generator_text)
      exit(8)

  def get_body(self, **kwargs) -> ContentAST.Section:
    if hasattr(self, '_generated_section') and self._generated_section:
      return self._generated_section
    return super().get_body()


class TrueFalse(FromText):
  pass