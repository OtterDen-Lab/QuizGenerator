#!env python
from __future__ import annotations

import logging
import os
from typing import List

import pydantic
from pydantic import constr

from QuizGenerator.core.question_ast import NodeRegistry, Node

log = logging.getLogger(__name__)


class StrictBase(pydantic.BaseModel):
  pass
  # model_config = pydantic.ConfigDict(extra='allow')


class Assessment(StrictBase):
  """
  Base class for assessments.  At it's core, it contains a list of questions and a list of submissions.
  Note that questions are Question objects, but are essentially AST that define the question.  They are ordered.
  
  fields:
    - questions: List[Question] -> List of Questions that are on this Assessment
    - submissions: List[Submission] -> List of Submissions for this assessment
  """
  questions: List[Question] = []
  submissions: List[Submission] = []


class Question(pydantic.BaseModel):
  """
  Question class for generating and grading exams.  Should be able to be rebuilt from pieces.
  Essentially is the AST.
  """
  pass
  body : Node
  explanation : Node
  
  @classmethod
  def from_dict(cls, data: dict) -> Question:
    return cls(
      body=NodeRegistry.from_dict(data["body"]),
      explanation=NodeRegistry.from_dict(data["explanation"])
    )
  


class Submission(StrictBase):
  """
  Submission of an exam.  For now, we assume it came from a file that has a name and a hash.
  The goal is that this should identify a unique submission without identifying the submitter
  """
  file_name: str
  file_hash: constr(pattern=r'^[0-9a-fA-F]{64}$')
  answers: List[Answer] = []

  class Answer(StrictBase):
    """
    An answer to a question on an exam.  Will be ripped out of the exam.  Likely will have extra info added later (e.g. feedback)
    """
    raw_data: bytes # Raw data of the input, probably an image -- tbd
  
  class Feedback(StrictBase):
    """
    Feedback from either grader, default, ai, or generated
    """
    text: str
