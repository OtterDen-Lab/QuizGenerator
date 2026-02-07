#!/usr/bin/env python
import abc
import logging
import math
import random

import QuizGenerator.contentast as ca
from QuizGenerator.question import Question, QuestionRegistry

log = logging.getLogger(__name__)


class VectorMathQuestion(Question):

  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.MATH)
    super().__init__(*args, **kwargs)

  @staticmethod
  def _generate_vector(rng, dimension, min_val=-10, max_val=10):
    """Generate a vector with random integer values."""
    return [rng.randint(min_val, max_val) for _ in range(dimension)]

  @staticmethod
  def _format_vector(vector):
    """Return a ca.Matrix element for the vector (format-independent).

    The Matrix element will render appropriately for each output format:
    - HTML: LaTeX bmatrix (for MathJax)
    - Typst: mat() with square bracket delimiter
    - LaTeX: bmatrix environment
    """
    # Convert to column matrix format: [[v1], [v2], [v3]]
    matrix_data = [[v] for v in vector]
    return ca.Matrix(data=matrix_data, bracket_type="b")

  @staticmethod
  def _format_vector_inline(vector):
    """Format vector for inline display."""
    elements = [str(v) for v in vector]
    return f"({', '.join(elements)})"

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)
    num_subquestions = kwargs.get("num_subquestions", 1)
    if num_subquestions > 1:
      raise NotImplementedError("Multipart not supported")

    dimension = kwargs.get("dimension", rng.randint(cls.MIN_DIMENSION, cls.MAX_DIMENSION))
    vector_a = cls._generate_vector(rng, dimension)
    vector_b = cls._generate_vector(rng, dimension)
    result = cls.calculate_single_result(vector_a, vector_b)

    return {
      "dimension": dimension,
      "vector_a": vector_a,
      "vector_b": vector_b,
      "result": result,
      "num_subquestions": num_subquestions,
    }

  # Abstract methods that subclasses must still implement
  @staticmethod
  @abc.abstractmethod
  def get_operator():
    """Return the LaTeX operator for this operation."""
    pass

  @staticmethod
  @abc.abstractmethod
  def calculate_single_result(vector_a, vector_b):
    """Calculate the result for a single question with two vectors."""
    pass

  @abc.abstractmethod
  def create_subquestion_answers(self, subpart_index, result):
    """Create answer objects for a subquestion result."""
    pass


@QuestionRegistry.register()
class VectorAddition(VectorMathQuestion):

  MIN_DIMENSION = 2
  MAX_DIMENSION = 4

  @staticmethod
  def get_operator():
    return "+"

  @staticmethod
  def calculate_single_result(vector_a, vector_b):
    return [vector_a[i] + vector_b[i] for i in range(len(vector_a))]

  def create_subquestion_answers(self, subpart_index, result):
    raise NotImplementedError("Multipart not supported")

  def create_single_answers(self, result):
    answer = ca.AnswerTypes.Vector(result)
    self._single_answer = answer
    return [answer]

  @classmethod
  def _build_body(cls, context):
    """Build question body and collect answers."""
    body = ca.Section()

    body.add_element(ca.Paragraph(["Calculate the following:"]))

    # Equation display using MathExpression for format-independent rendering
    vector_a_elem = cls._format_vector(context["vector_a"])
    vector_b_elem = cls._format_vector(context["vector_b"])
    body.add_element(ca.MathExpression([
        vector_a_elem,
        " + ",
        vector_b_elem,
        " = "
    ]))

    answer = ca.AnswerTypes.Vector(context["result"])
    body.add_element(ca.OnlyHtml([ca.Paragraph(["Enter your answer as a column vector:"])]))
    body.add_element(ca.OnlyHtml([answer]))

    return body, [answer]

  @classmethod
  def _build_explanation(cls, context):
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph(["To add vectors, we add corresponding components:"]))

    # Use LaTeX syntax for make_block_equation__multiline_equals
    vector_a_str = r" \\ ".join([str(v) for v in context["vector_a"]])
    vector_b_str = r" \\ ".join([str(v) for v in context["vector_b"]])
    result_str = r" \\ ".join([str(v) for v in context["result"]])
    addition_str = r" \\ ".join([
      f"{context['vector_a'][i]}+{context['vector_b'][i]}"
      for i in range(context["dimension"])
    ])

    explanation.add_element(
        ca.Equation.make_block_equation__multiline_equals(
            lhs=r"\vec{a} + \vec{b}",
            rhs=[
                f"\\begin{{bmatrix}} {vector_a_str} \\end{{bmatrix}} + \\begin{{bmatrix}} {vector_b_str} \\end{{bmatrix}}",
                f"\\begin{{bmatrix}} {addition_str} \\end{{bmatrix}}",
                f"\\begin{{bmatrix}} {result_str} \\end{{bmatrix}}"
            ]
        )
    )

    return explanation, []


@QuestionRegistry.register()
class VectorScalarMultiplication(VectorMathQuestion):

  MIN_DIMENSION = 2
  MAX_DIMENSION = 4

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)
    num_subquestions = kwargs.get("num_subquestions", 1)
    if num_subquestions > 1:
      raise NotImplementedError("Multipart not supported")

    dimension = kwargs.get("dimension", rng.randint(cls.MIN_DIMENSION, cls.MAX_DIMENSION))
    vector_a = cls._generate_vector(rng, dimension)
    scalar = cls._generate_scalar(rng)
    result = cls.calculate_single_result(vector_a, scalar)

    return {
      "dimension": dimension,
      "vector_a": vector_a,
      "scalar": scalar,
      "result": result,
      "num_subquestions": num_subquestions,
    }

  @staticmethod
  def _generate_scalar(rng):
    """Generate a non-zero scalar for multiplication."""
    scalar = rng.randint(-5, 5)
    while scalar == 0:
      scalar = rng.randint(-5, 5)
    return scalar

  @staticmethod
  def get_operator():
    return "\\cdot"

  @staticmethod
  def calculate_single_result(vector_a, vector_b):
    return [vector_b * component for component in vector_a]

  def create_subquestion_answers(self, subpart_index, result):
    raise NotImplementedError("Multipart not supported")

  def create_single_answers(self, result):
    answer = ca.AnswerTypes.Vector(result)
    self._single_answer = answer
    return [answer]

  @classmethod
  def _build_body(cls, context):
    """Build question body and collect answers."""
    body = ca.Section()

    body.add_element(ca.Paragraph(["Calculate the following:"]))

    # Equation display using MathExpression
    vector_elem = cls._format_vector(context["vector_a"])
    body.add_element(ca.MathExpression([
        f"{context['scalar']} \\cdot ",
        vector_elem,
        " = "
    ]))

    answer = ca.AnswerTypes.Vector(context["result"])
    body.add_element(ca.OnlyHtml([ca.Paragraph(["Enter your answer as a column vector:"])]))
    body.add_element(ca.OnlyHtml([answer]))

    return body, [answer]

  @classmethod
  def _build_explanation(cls, context):
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph(["To multiply a vector by a scalar, we multiply each component by the scalar:"]))

    vector_str = r" \\ ".join([str(v) for v in context["vector_a"]])
    multiplication_str = r" \\ ".join([f"{context['scalar']} \\cdot {v}" for v in context["vector_a"]])
    result_str = r" \\ ".join([str(v) for v in context["result"]])

    explanation.add_element(
        ca.Equation.make_block_equation__multiline_equals(
            lhs=f"{context['scalar']} \\cdot \\vec{{v}}",
            rhs=[
                f"{context['scalar']} \\cdot \\begin{{bmatrix}} {vector_str} \\end{{bmatrix}}",
                f"\\begin{{bmatrix}} {multiplication_str} \\end{{bmatrix}}",
                f"\\begin{{bmatrix}} {result_str} \\end{{bmatrix}}"
            ]
        )
    )

    return explanation, []


@QuestionRegistry.register()
class VectorDotProduct(VectorMathQuestion):

  MIN_DIMENSION = 2
  MAX_DIMENSION = 4

  @staticmethod
  def get_operator():
    return "\\cdot"

  @staticmethod
  def calculate_single_result(vector_a, vector_b):
    return sum(vector_a[i] * vector_b[i] for i in range(len(vector_a)))

  def create_subquestion_answers(self, subpart_index, result):
    raise NotImplementedError("Multipart not supported")

  def create_single_answers(self, result):
    answer = ca.AnswerTypes.Int(result)
    self._single_answer = answer
    return [answer]

  @classmethod
  def _build_body(cls, context):
    """Build question body and collect answers."""
    body = ca.Section()

    body.add_element(ca.Paragraph(["Calculate the following:"]))

    # Equation display using MathExpression
    vector_a_elem = cls._format_vector(context["vector_a"])
    vector_b_elem = cls._format_vector(context["vector_b"])
    body.add_element(ca.MathExpression([
        vector_a_elem,
        " \\cdot ",
        vector_b_elem,
        " = "
    ]))

    answer = ca.AnswerTypes.Int(context["result"])
    body.add_element(ca.OnlyHtml([answer]))

    return body, [answer]

  @classmethod
  def _build_explanation(cls, context):
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph(["The dot product is calculated by multiplying corresponding components and summing the results:"]))

    vector_a_str = r" \\ ".join([str(v) for v in context["vector_a"]])
    vector_b_str = r" \\ ".join([str(v) for v in context["vector_b"]])
    products_str = " + ".join([
      f"({context['vector_a'][i]} \\cdot {context['vector_b'][i]})"
      for i in range(context["dimension"])
    ])
    calculation_str = " + ".join([
      str(context["vector_a"][i] * context["vector_b"][i])
      for i in range(context["dimension"])
    ])

    explanation.add_element(
        ca.Equation.make_block_equation__multiline_equals(
            lhs="\\vec{a} \\cdot \\vec{b}",
            rhs=[
                f"\\begin{{bmatrix}} {vector_a_str} \\end{{bmatrix}} \\cdot \\begin{{bmatrix}} {vector_b_str} \\end{{bmatrix}}",
                products_str,
                calculation_str,
                str(context["result"])
            ]
          )
      )

    return explanation, []  # Explanations don't have answers


@QuestionRegistry.register()
class VectorMagnitude(VectorMathQuestion):

  MIN_DIMENSION = 2
  MAX_DIMENSION = 3

  @staticmethod
  def get_operator():
    return "||"

  @staticmethod
  def calculate_single_result(vector_a, vector_b):
    magnitude_squared = sum(component ** 2 for component in vector_a)
    return math.sqrt(magnitude_squared)

  def create_subquestion_answers(self, subpart_index, result):
    raise NotImplementedError("Multipart not supported")

  def create_single_answers(self, result):
    answer = ca.AnswerTypes.Float(result)
    self._single_answer = answer
    return [answer]

  @classmethod
  def _build_body(cls, context):
    """Build question body and collect answers."""
    body = ca.Section()

    body.add_element(ca.Paragraph(["Calculate the following:"]))

    # Equation display using MathExpression
    vector_elem = cls._format_vector(context["vector_a"])
    body.add_element(ca.MathExpression([
        "||",
        vector_elem,
        "|| = "
    ]))

    answer = ca.AnswerTypes.Float(context["result"])
    body.add_element(ca.OnlyHtml([answer]))

    return body, [answer]

  @classmethod
  def _build_explanation(cls, context):
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph(["The magnitude of a vector is calculated using the formula:"]))
    explanation.add_element(ca.Equation(
        r"||\vec{v}|| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}", inline=False
    ))

    # Use LaTeX syntax for make_block_equation__multiline_equals
    vector_str = r" \\ ".join([str(v) for v in context["vector_a"]])
    squares_str = " + ".join([f"{v}^2" for v in context["vector_a"]])
    calculation_str = " + ".join([str(v**2) for v in context["vector_a"]])
    sum_of_squares = sum(component ** 2 for component in context["vector_a"])
    result_formatted = sorted(ca.Answer.accepted_strings(context["result"]), key=lambda s: len(s))[0]

    explanation.add_element(
        ca.Equation.make_block_equation__multiline_equals(
            lhs=r"||\vec{v}||",
            rhs=[
                f"\\left|\\left| \\begin{{bmatrix}} {vector_str} \\end{{bmatrix}} \\right|\\right|",
                f"\\sqrt{{{squares_str}}}",
                f"\\sqrt{{{calculation_str}}}",
                f"\\sqrt{{{sum_of_squares}}}",
                result_formatted
            ]
        )
    )

    return explanation, []
