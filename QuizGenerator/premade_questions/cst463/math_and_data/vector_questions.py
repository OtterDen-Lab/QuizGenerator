#!env python
import abc
import logging
import math
from typing import List

from QuizGenerator.question import Question, QuestionRegistry
import QuizGenerator.contentast as ca
from QuizGenerator.mixins import MathOperationQuestion

log = logging.getLogger(__name__)


class VectorMathQuestion(MathOperationQuestion, Question):

  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.MATH)
    super().__init__(*args, **kwargs)

  def _generate_vector(self, dimension, min_val=-10, max_val=10):
    """Generate a vector with random integer values."""
    return [self.rng.randint(min_val, max_val) for _ in range(dimension)]

  def _format_vector(self, vector):
    """Return a ca.Matrix element for the vector (format-independent).

    The Matrix element will render appropriately for each output format:
    - HTML: LaTeX bmatrix (for MathJax)
    - Typst: mat() with square bracket delimiter
    - LaTeX: bmatrix environment
    """
    # Convert to column matrix format: [[v1], [v2], [v3]]
    matrix_data = [[v] for v in vector]
    return ca.Matrix(data=matrix_data, bracket_type="b")

  def _format_vector_inline(self, vector):
    """Format vector for inline display."""
    elements = [str(v) for v in vector]
    return f"({', '.join(elements)})"

  # Implement MathOperationQuestion abstract methods

  def generate_operands(self):
    """Generate two vectors for the operation."""
    if not hasattr(self, 'dimension'):
      self.dimension = self.rng.randint(self.MIN_DIMENSION, self.MAX_DIMENSION)
    vector_a = self._generate_vector(self.dimension)
    vector_b = self._generate_vector(self.dimension)
    return vector_a, vector_b

  def format_operand_latex(self, operand):
    """Format a vector for LaTeX display."""
    return self._format_vector(operand)

  def format_single_equation(self, operand_a, operand_b):
    """Format the equation for single questions."""
    operand_a_latex = self.format_operand_latex(operand_a)
    operand_b_latex = self.format_operand_latex(operand_b)
    return f"{operand_a_latex} {self.get_operator()} {operand_b_latex}"

  # Vector-specific overrides

  def refresh(self, *args, **kwargs):
    # Generate vector dimension first
    self.dimension = self.rng.randint(self.MIN_DIMENSION, self.MAX_DIMENSION)

    # Call parent refresh which will use our generate_operands method
    super().refresh(*args, **kwargs)

    self.vector_a = self.operand_a
    self.vector_b = self.operand_b

  def generate_subquestion_data(self):
    """Generate LaTeX content for each subpart of the question.
    Override to handle vector-specific keys in subquestion_data."""
    subparts = []
    for data in self.subquestion_data:
      # Map generic operand names to vector names for compatibility
      vector_a = data.get('vector_a', data['operand_a'])
      vector_b = data.get('vector_b', data['operand_b'])

      vector_a_latex = self._format_vector(vector_a)
      vector_b_latex = self._format_vector(vector_b)
      # Return as tuple of (matrix_a, operator, matrix_b)
      subparts.append((vector_a_latex, self.get_operator(), vector_b_latex))
    return subparts

  # Abstract methods that subclasses must still implement
  @abc.abstractmethod
  def get_operator(self):
    """Return the LaTeX operator for this operation."""
    pass

  @abc.abstractmethod
  def calculate_single_result(self, vector_a, vector_b):
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

  def get_operator(self):
    return "+"

  def calculate_single_result(self, vector_a, vector_b):
    return [vector_a[i] + vector_b[i] for i in range(len(vector_a))]

  def create_subquestion_answers(self, subpart_index, result):
    raise NotImplementedError("Multipart not supported")

  def create_single_answers(self, result):
    self.answers["result"] = ca.AnswerTypes.Vector(result)

  def _get_body(self):
    """Build question body and collect answers."""
    body = ca.Section()

    body.add_element(ca.Paragraph([self.get_intro_text()]))

    # Equation display using MathExpression for format-independent rendering
    vector_a_elem = self._format_vector(self.vector_a)
    vector_b_elem = self._format_vector(self.vector_b)
    body.add_element(ca.MathExpression([
        vector_a_elem,
        " + ",
        vector_b_elem,
        " = "
    ]))

    # Canvas-only answer field - use stored answer for consistent UUID
    answer = self.answers["result"]
    body.add_element(ca.OnlyHtml([ca.Paragraph(["Enter your answer as a column vector:"])]))
    body.add_element(ca.OnlyHtml([answer]))

    return body, list(self.answers.values())

  def _get_explanation(self):
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph(["To add vectors, we add corresponding components:"]))

    # Use LaTeX syntax for make_block_equation__multiline_equals
    vector_a_str = r" \\ ".join([str(v) for v in self.vector_a])
    vector_b_str = r" \\ ".join([str(v) for v in self.vector_b])
    result_str = r" \\ ".join([str(v) for v in self.result])
    addition_str = r" \\ ".join([f"{self.vector_a[i]}+{self.vector_b[i]}" for i in range(self.dimension)])

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

  def _generate_scalar(self):
    """Generate a non-zero scalar for multiplication."""
    scalar = self.rng.randint(-5, 5)
    while scalar == 0:
      scalar = self.rng.randint(-5, 5)
    return scalar

  def refresh(self, *args, **kwargs):
    # Generate scalar first, then call parent refresh
    self.scalar = self._generate_scalar()
    super().refresh(*args, **kwargs)

  def get_operator(self):
    return f"{self.scalar} \\cdot"

  def calculate_single_result(self, vector_a, vector_b):
    return [self.scalar * component for component in vector_a]

  def create_subquestion_answers(self, subpart_index, result):
    raise NotImplementedError("Multipart not supported")

  def create_single_answers(self, result):
    self.answers["result"] = ca.AnswerTypes.Vector(result)

  def _get_body(self):
    """Build question body and collect answers."""
    body = ca.Section()

    body.add_element(ca.Paragraph([self.get_intro_text()]))

    # Equation display using MathExpression
    vector_elem = self._format_vector(self.vector_a)
    body.add_element(ca.MathExpression([
        f"{self.scalar} \\cdot ",
        vector_elem,
        " = "
    ]))

    # Canvas-only answer field - use stored answer
    answer = self.answers["result"]
    body.add_element(ca.OnlyHtml([ca.Paragraph(["Enter your answer as a column vector:"])]))
    body.add_element(ca.OnlyHtml([answer]))

    return body, list(self.answers.values())

  def _get_explanation(self):
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph(["To multiply a vector by a scalar, we multiply each component by the scalar:"]))

    vector_str = r" \\ ".join([str(v) for v in self.vector_a])
    multiplication_str = r" \\ ".join([f"{self.scalar} \\cdot {v}" for v in self.vector_a])
    result_str = r" \\ ".join([str(v) for v in self.result])

    explanation.add_element(
        ca.Equation.make_block_equation__multiline_equals(
            lhs=f"{self.scalar} \\cdot \\vec{{v}}",
            rhs=[
                f"{self.scalar} \\cdot \\begin{{bmatrix}} {vector_str} \\end{{bmatrix}}",
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

  def get_operator(self):
    return "\\cdot"

  def calculate_single_result(self, vector_a, vector_b):
    return sum(vector_a[i] * vector_b[i] for i in range(len(vector_a)))

  def create_subquestion_answers(self, subpart_index, result):
    raise NotImplementedError("Multipart not supported")

  def create_single_answers(self, result):
    self.answers["dot_product"] = ca.AnswerTypes.Int(result)

  def _get_body(self):
    """Build question body and collect answers."""
    body = ca.Section()

    body.add_element(ca.Paragraph([self.get_intro_text()]))

    # Equation display using MathExpression
    vector_a_elem = self._format_vector(self.vector_a)
    vector_b_elem = self._format_vector(self.vector_b)
    body.add_element(ca.MathExpression([
        vector_a_elem,
        " \\cdot ",
        vector_b_elem,
        " = "
    ]))

    # Canvas-only answer field - use stored answer
    answer = self.answers["dot_product"]
    body.add_element(ca.OnlyHtml([answer]))

    return body, list(self.answers.values())

  def _get_explanation(self):
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph(["The dot product is calculated by multiplying corresponding components and summing the results:"]))

    vector_a_str = r" \\ ".join([str(v) for v in self.vector_a])
    vector_b_str = r" \\ ".join([str(v) for v in self.vector_b])
    products_str = " + ".join([f"({self.vector_a[i]} \\cdot {self.vector_b[i]})" for i in range(self.dimension)])
    calculation_str = " + ".join([str(self.vector_a[i] * self.vector_b[i]) for i in range(self.dimension)])

    explanation.add_element(
        ca.Equation.make_block_equation__multiline_equals(
            lhs="\\vec{a} \\cdot \\vec{b}",
            rhs=[
                f"\\begin{{bmatrix}} {vector_a_str} \\end{{bmatrix}} \\cdot \\begin{{bmatrix}} {vector_b_str} \\end{{bmatrix}}",
                products_str,
                calculation_str,
                str(self.result)
            ]
          )
      )

    return explanation, []  # Explanations don't have answers


@QuestionRegistry.register()
class VectorMagnitude(VectorMathQuestion):

  MIN_DIMENSION = 2
  MAX_DIMENSION = 3

  def get_operator(self):
    return "||"

  def calculate_single_result(self, vector_a, vector_b):
    magnitude_squared = sum(component ** 2 for component in vector_a)
    return math.sqrt(magnitude_squared)

  def create_subquestion_answers(self, subpart_index, result):
    raise NotImplementedError("Multipart not supported")

  def create_single_answers(self, result):
    self.answers["magnitude"] = ca.AnswerTypes.Float(result)

  def _get_body(self):
    """Build question body and collect answers."""
    body = ca.Section()

    body.add_element(ca.Paragraph([self.get_intro_text()]))

    # Equation display using MathExpression
    vector_elem = self._format_vector(self.vector_a)
    body.add_element(ca.MathExpression([
        "||",
        vector_elem,
        "|| = "
    ]))

    # Canvas-only answer field - use stored answer
    answer = self.answers["magnitude"]
    body.add_element(ca.OnlyHtml([answer]))

    return body, list(self.answers.values())

  def _get_explanation(self):
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph(["The magnitude of a vector is calculated using the formula:"]))
    explanation.add_element(ca.Equation(
        r"||\vec{v}|| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}", inline=False
    ))

    # Use LaTeX syntax for make_block_equation__multiline_equals
    vector_str = r" \\ ".join([str(v) for v in self.vector_a])
    squares_str = " + ".join([f"{v}^2" for v in self.vector_a])
    calculation_str = " + ".join([str(v**2) for v in self.vector_a])
    sum_of_squares = sum(component ** 2 for component in self.vector_a)
    result_formatted = sorted(ca.Answer.accepted_strings(self.result), key=lambda s: len(s))[0]

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
