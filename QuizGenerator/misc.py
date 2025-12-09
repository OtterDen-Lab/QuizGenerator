#!env python
from __future__ import annotations

import decimal
import enum
import itertools
import logging
import math
import numpy as np
from typing import List, Dict, Tuple, Any

import fractions

from QuizGenerator.contentast import ContentAST

log = logging.getLogger(__name__)


def fix_negative_zero(value):
  """Convert -0.0 to 0.0 to avoid confusing display."""
  if isinstance(value, (int, float)):
    return 0.0 if value == 0 else value
  return value


class Answer:
  DEFAULT_ROUNDING_DIGITS = 4
  
  class AnswerKind(enum.Enum):
    BLANK = "fill_in_multiple_blanks_question"
    MULTIPLE_ANSWER = "multiple_answers_question"
    ESSAY = "essay_question"
    MULTIPLE_DROPDOWN = "multiple_dropdowns_question"
    NUMERICAL_QUESTION = "numerical_question" # note: these can only be single answers as far as I can tell
    
  class VariableKind(enum.Enum):
    STR = enum.auto()
    INT = enum.auto()
    FLOAT = enum.auto()
    BINARY = enum.auto()
    HEX = enum.auto()
    BINARY_OR_HEX = enum.auto()
    AUTOFLOAT = enum.auto()
    LIST = enum.auto()
    VECTOR = enum.auto()
    MATRIX = enum.auto()
    
    
  def __init__(
      self, key:str,
      value,
      kind : Answer.AnswerKind = AnswerKind.BLANK,
      variable_kind : Answer.VariableKind = VariableKind.STR,
      display=None,
      length=None,
      correct=True,
      baffles=None,
      pdf_only=False
  ):
    self.key = key
    self.value = value
    self.kind = kind
    self.variable_kind = variable_kind
    # For list values in display, show the first option (or join them with /)
    if display is not None:
      self.display = display
    elif isinstance(value, list) and variable_kind == Answer.VariableKind.STR:
      self.display = value[0] if len(value) == 1 else " / ".join(value)
    else:
      self.display = value
    self.length = length # Used for bits and hex to be printed appropriately
    self.correct = correct
    self.baffles = baffles
    self.pdf_only = pdf_only
  
  def get_for_canvas(self, single_answer=False) -> List[Dict]:
    # If this answer is marked as PDF-only, don't send it to Canvas
    if self.pdf_only:
      return []

    canvas_answers : List[Dict] = []
    if self.variable_kind == Answer.VariableKind.BINARY:
      canvas_answers = [
        {
          "blank_id": self.key,
          "answer_text": f"{self.value:0{self.length if self.length is not None else 0}b}",
          "answer_weight": 100 if self.correct else 0,
        },
        {
          "blank_id": self.key,
          "answer_text": f"0b{self.value:0{self.length if self.length is not None else 0}b}",
          "answer_weight": 100 if self.correct else 0,
        }
      ]
    elif self.variable_kind == Answer.VariableKind.HEX:
      canvas_answers = [
        {
          "blank_id": self.key,
          "answer_text": f"{self.value:0{(self.length // 8) + 1 if self.length is not None else 0}X}",
          "answer_weight": 100 if self.correct else 0,
        },{
          "blank_id": self.key,
          "answer_text": f"0x{self.value:0{(self.length // 8) + 1 if self.length is not None else 0}X}",
          "answer_weight": 100 if self.correct else 0,
        }
      ]
    elif self.variable_kind == Answer.VariableKind.BINARY_OR_HEX:
      canvas_answers = [
        {
          "blank_id": self.key,
          "answer_text": f"{self.value:0{self.length if self.length is not None else 0}b}",
          "answer_weight": 100 if self.correct else 0,
        },
        {
          "blank_id": self.key,
          "answer_text": f"0b{self.value:0{self.length if self.length is not None else 0}b}",
          "answer_weight": 100 if self.correct else 0,
        },
        {
          "blank_id": self.key,
          "answer_text": f"{self.value:0{math.ceil(self.length / 8) if self.length is not None else 0}X}",
          "answer_weight": 100 if self.correct else 0,
        },
        {
          "blank_id": self.key,
          "answer_text": f"0x{self.value:0{math.ceil(self.length / 8) if self.length is not None else 0}X}",
          "answer_weight": 100 if self.correct else 0,
        },
        {
          "blank_id": self.key,
          "answer_text": f"{self.value}",
          "answer_weight": 100 if self.correct else 0,
        },
      
      ]
    elif self.variable_kind in [
      Answer.VariableKind.AUTOFLOAT,
      Answer.VariableKind.FLOAT,
      Answer.VariableKind.INT
    ]:
      if single_answer:
        canvas_answers = [
          {
            "numerical_answer_type": "exact_answer",
            "answer_text": round(self.value, self.DEFAULT_ROUNDING_DIGITS),
            "answer_exact": round(self.value, self.DEFAULT_ROUNDING_DIGITS),
            "answer_error_margin": 0.1,
            "answer_weight": 100 if self.correct else 0,
          }
        ]
      else:
        # Use the accepted_strings helper with settings that match the original AUTOFLOAT behavior
        answer_strings = self.__class__.accepted_strings(
          self.value,
          allow_integer=True,
          allow_simple_fraction=True,
          max_denominator=3*4*5,  # For process questions, these are the numbers of jobs we'd have
          allow_mixed=True,
          include_spaces=False,
          include_fixed_even_if_integer=True
        )
  
        canvas_answers = [
        {
          "blank_id": self.key,
          "answer_text": answer_string,
          "answer_weight": 100 if self.correct else 0,
        }
        for answer_string in answer_strings
      ]
      
    elif self.variable_kind == Answer.VariableKind.VECTOR:

      # Get all answer variations
      answer_variations = [
        self.__class__.accepted_strings(dimension_value)
        for dimension_value in self.value
      ]

      canvas_answers = []
      for combination in itertools.product(*answer_variations):
        # Add parentheses format for all vectors: (1, 2, 3)
        canvas_answers.append({
          "blank_id" : self.key,
          "answer_weight": 100 if self.correct else 0,
          "answer_text": f"({', '.join(list(combination))})",
        })

        # Add non-parentheses format only for single-element vectors: 5
        if len(combination) == 1:
          canvas_answers.append(
            {
              "blank_id": self.key,
              "answer_weight": 100 if self.correct else 0,
              "answer_text": f"{', '.join(combination)}",
            }
          )
      return canvas_answers
        
    elif self.variable_kind == Answer.VariableKind.LIST:
      canvas_answers = [
        {
          "blank_id": self.key,
          "answer_text": ', '.join(map(str, possible_state)),
          "answer_weight": 100 if self.correct else 0,
        }
        for possible_state in [self.value] #itertools.permutations(self.value)
      ]
    
    else:
      # For string answers, check if value is a list of acceptable alternatives
      if isinstance(self.value, list):
        canvas_answers = [
          {
            "blank_id": self.key,
            "answer_text": str(alt),
            "answer_weight": 100 if self.correct else 0,
          }
          for alt in self.value
        ]
      else:
        canvas_answers = [{
          "blank_id": self.key,
          "answer_text": self.value,
          "answer_weight": 100 if self.correct else 0,
        }]
    
    if self.baffles is not None:
      for baffle in self.baffles:
        canvas_answers.append({
          "blank_id": self.key,
          "answer_text": baffle,
          "answer_weight": 0,
        })
    
    return canvas_answers

  def get_display_string(self) -> str:
    """
    Get the formatted display string for this answer (for grading/answer keys).

    Returns the answer in the most appropriate format for human readability:
    - BINARY_OR_HEX: hex format (0x...)
    - BINARY: binary format (0b...)
    - HEX: hex format (0x...)
    - AUTOFLOAT/FLOAT: rounded to DEFAULT_ROUNDING_DIGITS
    - INT: integer
    - STR/LIST/VECTOR: as-is
    """
    if self.variable_kind == Answer.VariableKind.BINARY_OR_HEX:
      # For binary_hex answers, show hex format (more compact and readable)
      hex_digits = math.ceil(self.length / 4) if self.length is not None else 0
      return f"0x{self.value:0{hex_digits}X}"

    elif self.variable_kind == Answer.VariableKind.BINARY:
      # Show binary format
      return f"0b{self.value:0{self.length if self.length is not None else 0}b}"

    elif self.variable_kind == Answer.VariableKind.HEX:
      # Show hex format
      hex_digits = (self.length // 4) + 1 if self.length is not None else 0
      return f"0x{self.value:0{hex_digits}X}"

    elif self.variable_kind == Answer.VariableKind.AUTOFLOAT:
      # Round to default precision for readability
      rounded = round(self.value, self.DEFAULT_ROUNDING_DIGITS)
      return f"{fix_negative_zero(rounded)}"

    elif self.variable_kind == Answer.VariableKind.FLOAT:
      # Round to default precision
      if isinstance(self.value, (list, tuple)):
        rounded = round(self.value[0], self.DEFAULT_ROUNDING_DIGITS)
        return f"{fix_negative_zero(rounded)}"
      rounded = round(self.value, self.DEFAULT_ROUNDING_DIGITS)
      return f"{fix_negative_zero(rounded)}"

    elif self.variable_kind == Answer.VariableKind.INT:
      return str(int(self.value))

    elif self.variable_kind == Answer.VariableKind.LIST:
      return ", ".join(str(v) for v in self.value)

    elif self.variable_kind == Answer.VariableKind.VECTOR:
      # Format as comma-separated rounded values
      return ", ".join(str(fix_negative_zero(round(v, self.DEFAULT_ROUNDING_DIGITS))) for v in self.value)

    else:
      # Default: use display or value
      return str(self.display if hasattr(self, 'display') else self.value)

  def get_ast_element(self, label=None):
    from QuizGenerator.contentast import ContentAST
    
    return ContentAST.Answer(answer=self, label=label) # todo fix label

  # Factory methods for common answer types
  @classmethod
  def binary_hex(cls, key: str, value: int, length: int = None, **kwargs) -> 'Answer':
    """Create an answer that accepts binary or hex format"""
    return cls(
      key=key,
      value=value,
      variable_kind=cls.VariableKind.BINARY_OR_HEX,
      length=length,
      **kwargs
    )
  
  @classmethod
  def auto_float(cls, key: str, value: float, **kwargs) -> 'Answer':
    """Create an answer that accepts multiple float formats (decimal, fraction, mixed)"""
    return cls(
      key=key,
      value=value,
      variable_kind=cls.VariableKind.AUTOFLOAT,
      **kwargs
    )
  
  @classmethod
  def integer(cls, key: str, value: int, **kwargs) -> 'Answer':
    """Create an integer answer"""
    return cls(
      key=key,
      value=value,
      variable_kind=cls.VariableKind.INT,
      **kwargs
    )
  
  @classmethod
  def string(cls, key: str, value: str, **kwargs) -> 'Answer':
    """Create a string answer"""
    return cls(
      key=key,
      value=value,
      variable_kind=cls.VariableKind.STR,
      **kwargs
    )
  
  @classmethod
  def binary(cls, key: str, value: int, length: int = None, **kwargs) -> 'Answer':
    """Create a binary-only answer"""
    return cls(
      key=key,
      value=value,
      variable_kind=cls.VariableKind.BINARY,
      length=length,
      **kwargs
    )
  
  @classmethod
  def hex_value(cls, key: str, value: int, length: int = None, **kwargs) -> 'Answer':
    """Create a hex-only answer"""
    return cls(
      key=key,
      value=value,
      variable_kind=cls.VariableKind.HEX,
      length=length,
      **kwargs
    )
  
  @classmethod
  def float_value(cls, key: str, value: Tuple[float], **kwargs) -> 'Answer':
    """Create a simple float answer (no fraction conversion)"""
    return cls(
      key=key,
      value=value,
      variable_kind=cls.VariableKind.FLOAT,
      **kwargs
    )
  
  @classmethod
  def list_value(cls, key: str, value: list, **kwargs) -> 'Answer':
    """Create a list answer (comma-separated values)"""
    return cls(
      key=key,
      value=value,
      variable_kind=cls.VariableKind.LIST,
      **kwargs
    )

  @classmethod
  def vector_value(cls, key: str, value: List[float], **kwargs) -> 'Answer':
    """Create a simple float answer (no fraction conversion)"""
    return cls(
      key=key,
      value=value,
      variable_kind=cls.VariableKind.VECTOR,
      **kwargs
    )

  @classmethod
  def dropdown(cls, key: str, value: str, baffles: list = None, **kwargs) -> 'Answer':
    """Create a dropdown answer with wrong answer choices (baffles)"""
    return cls(
      key=key,
      value=value,
      kind=cls.AnswerKind.MULTIPLE_DROPDOWN,
      baffles=baffles,
      **kwargs
    )

  @classmethod
  def multiple_choice(cls, key: str, value: str, baffles: list = None, **kwargs) -> 'Answer':
    """Create a multiple choice answer with wrong answer choices (baffles)"""
    return cls(
      key=key,
      value=value,
      kind=cls.AnswerKind.MULTIPLE_ANSWER,
      baffles=baffles,
      **kwargs
    )

  @classmethod
  def essay(cls, key: str, **kwargs) -> 'Answer':
    """Create an essay question (no specific correct answer)"""
    return cls(
      key=key,
      value="",  # Essays don't have predetermined answers
      kind=cls.AnswerKind.ESSAY,
      **kwargs
    )
  
  @classmethod
  def matrix(cls, key: str, value: np.array|List, **kwargs ):
    return MatrixAnswer(
      key=key,
      value=value,
      variable_kind=cls.VariableKind.MATRIX
    )
  
  @staticmethod
  def _to_fraction(x):
    """Convert int/float/decimal.Decimal/fractions.Fraction/str('a/b' or decimal) to fractions.Fraction exactly."""
    log.debug(f"x: {x} {x.__class__}")
    if isinstance(x, fractions.Fraction):
      return x
    if isinstance(x, int):
      return fractions.Fraction(x, 1)
    if isinstance(x, decimal.Decimal):
      # exact conversion of decimal.Decimal to fractions.Fraction
      sign, digits, exp = x.as_tuple()
      n = 0
      for d in digits:
        n = n * 10 + d
      n = -n if sign else n
      if exp >= 0:
        return fractions.Fraction(n * (10 ** exp), 1)
      else:
        return fractions.Fraction(n, 10 ** (-exp))
    if isinstance(x, str):
      s = x.strip()
      if '/' in s:
        a, b = s.split('/', 1)
        return fractions.Fraction(int(a.strip()), int(b.strip()))
      return fractions.Fraction(decimal.Decimal(s))
    # float or other numerics
    return fractions.Fraction(decimal.Decimal(str(x)))
  
  @staticmethod
  def accepted_strings(
      value,
      *,
      allow_integer=True,  # allow "whole numbers as whole numbers"
      allow_simple_fraction=True,  # allow simple a/b when denominator small
      max_denominator=720,  # how "simple" the fraction is
      allow_mixed=False,  # also allow "1 1/2" for 3/2
      include_spaces=False,  # also accept "1 / 2"
      include_fixed_even_if_integer=False  # include "1.0000" when value is 1 and fixed_decimals is set
  ):
    """
    Return a sorted list of strings you can paste into Canvas as alternate correct answers.
    """
    decimal.getcontext().prec = max(34, (Answer.DEFAULT_ROUNDING_DIGITS or 0) + 10)
    f = Answer._to_fraction(value)
    outs = set()
    
    # Integer form
    if f.denominator == 1 and allow_integer:
      outs.add(str(f.numerator))
      if include_fixed_even_if_integer:
        q = decimal.Decimal(1).scaleb(-Answer.DEFAULT_ROUNDING_DIGITS)  # 1e-<fixed_decimals>
        d = decimal.Decimal(f.numerator).quantize(q, rounding=decimal.ROUND_HALF_UP)
        outs.add(format(d, 'f'))
    
    # Fixed-decimal form (exactly N decimals)
    q = decimal.Decimal(1).scaleb(-Answer.DEFAULT_ROUNDING_DIGITS)
    d = (decimal.Decimal(f.numerator) / decimal.Decimal(f.denominator)).quantize(q, rounding=decimal.ROUND_HALF_UP)
    outs.add(format(d, 'f'))
    
    # Trimmed decimal (no trailing zeros; up to max_trimmed_decimals)
    if Answer.DEFAULT_ROUNDING_DIGITS:
      q = decimal.Decimal(1).scaleb(-Answer.DEFAULT_ROUNDING_DIGITS)
      d = (decimal.Decimal(f.numerator) / decimal.Decimal(f.denominator)).quantize(q, rounding=decimal.ROUND_HALF_UP)
      s = format(d, 'f').rstrip('0').rstrip('.')
      # ensure we keep leading zero like "0.5"
      if s.startswith('.'):
        s = '0' + s
      if s == '-0':  # tidy negative zero
        s = '0'
      outs.add(s)
    
    # Simple fraction (reduced, with small denominator)
    if allow_simple_fraction:
      fr = f.limit_denominator(max_denominator)
      if fr == f:
        a, b = fr.numerator, fr.denominator
        outs.add(f"{a}/{b}")
        if include_spaces:
          outs.add(f"{a} / {b}")
        if allow_mixed and b != 1 and abs(a) > b:
          sign = '-' if a < 0 else ''
          A = abs(a)
          whole, rem = divmod(A, b)
          outs.add(f"{sign}{whole} {rem}/{b}")
    
    return sorted(outs, key=lambda s: (len(s), s))
  

class MatrixAnswer(Answer):
  def get_for_canvas(self, single_answer=False) -> List[Dict]:
    canvas_answers = []
    
    """
    The core idea is that we will walk through and generate each one for X_i,j

    The big remaining question is how we will get all these names to the outside world.
    It might have to be a pretty big challenge, or rather re-write.
    """
    
    # The core idea is that we will be walking through and generating a per-index set of answers.
    # Boy will this get messy.  Poor canvas.
    for i, j in np.ndindex(self.value.shape):
      entry_strings = self.__class__.accepted_strings(
        self.value[i,j],
        allow_integer=True,
        allow_simple_fraction=True,
        max_denominator=3 * 4 * 5,
        allow_mixed=True,
        include_spaces=False,
        include_fixed_even_if_integer=True
      )
      canvas_answers.extend(
        [
          {
            "blank_id": f"{self.key}_{i}_{j}",  # Give each an index associated with it so we can track it
            "answer_text": answer_string,
            "answer_weight": 100 if self.correct else 0,
          }
          for answer_string in entry_strings
        ]
      )
    
    return canvas_answers
  
  def get_ast_element(self, label=None):
    from QuizGenerator.contentast import ContentAST
    
    log.debug(f"self.value: {self.value}")
    
    data = [
      [
        ContentAST.Answer(
          Answer.float_value(
            key=f"{self.key}_{i}_{j}",
            value=self.value[i,j]
          )
        )
        for j in range(self.value.shape[1])
      ]
      for i in range(self.value.shape[0])
    ]
    table = ContentAST.Table(data)
    
    if label is not None:
      return ContentAST.Container([
        ContentAST.Text(f"{label} = "),
        table
      ])
    else:
      return table
