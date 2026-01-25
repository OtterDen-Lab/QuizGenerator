#!env python
import abc
import logging
import math
from typing import List

from QuizGenerator.question import Question, QuestionRegistry
from QuizGenerator.contentast import ContentAST, AnswerTypes
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
    """Return a ContentAST.Matrix element for the vector (format-independent).

    The Matrix element will render appropriately for each output format:
    - HTML: LaTeX bmatrix (for MathJax)
    - Typst: mat() with square bracket delimiter
    - LaTeX: bmatrix environment
    """
    # Convert to column matrix format: [[v1], [v2], [v3]]
    matrix_data = [[v] for v in vector]
    return ContentAST.Matrix(data=matrix_data, bracket_type="b")

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

    # For backward compatibility, set vector_a/vector_b for single questions
    if not self.is_multipart():
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
    letter = chr(ord('a') + subpart_index)
    for j in range(len(result)):
      self.answers[f"subpart_{letter}_{j}"] = AnswerTypes.IntAnswer(result[j])

  def create_single_answers(self, result):
    # Backward compatibility - still populate dict for old pattern
    for i in range(len(result)):
      self.answers[f"result_{i}"] = AnswerTypes.IntAnswer(result[i])

  def get_body(self):
    """Override parent get_body() to use our custom formatting."""
    body, _ = self._get_body()
    return body

  def _get_body(self):
    """Build question body and collect answers (new pattern)."""
    from typing import Tuple, List

    body = ContentAST.Section()
    answers = []

    body.add_element(ContentAST.Paragraph([self.get_intro_text()]))

    if self.is_multipart():
      # Use multipart formatting with repeated problem parts
      subpart_data = self.generate_subquestion_data()
      repeated_part = self.create_repeated_problem_part(subpart_data)
      body.add_element(repeated_part)

      # Collect all subpart answers
      for i, data in enumerate(self.subquestion_data):
        letter = chr(ord('a') + i)
        result = data['result']
        for j in range(len(result)):
          ans = AnswerTypes.IntAnswer(result[j])
          answers.append(ans)
    else:
      # Single equation display using MathExpression for format-independent rendering
      vector_a_elem = self._format_vector(self.vector_a)
      vector_b_elem = self._format_vector(self.vector_b)
      body.add_element(ContentAST.MathExpression([
          vector_a_elem,
          " + ",
          vector_b_elem,
          " = "
      ]))

      # Canvas-only answer fields (hidden from PDF)
      body.add_element(ContentAST.OnlyHtml([ContentAST.Paragraph(["Enter your answer as a column vector:"])]))
      table_data = []
      for i in range(self.dimension):
        ans = AnswerTypes.IntAnswer(self.result[i])
        answers.append(ans)
        table_data.append([ans])
      body.add_element(ContentAST.OnlyHtml([ContentAST.Table(data=table_data, padding=True)]))

    return body, answers

  def _get_explanation(self):
    """Build question explanation (new pattern)."""
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph(["To add vectors, we add corresponding components:"]))

    if self.is_multipart():
      # Handle multipart explanations
      for i, data in enumerate(self.subquestion_data):
        letter = chr(ord('a') + i)
        vector_a = data['vector_a']
        vector_b = data['vector_b']
        result = data['result']

        # Format vectors using Typst syntax
        vector_a_str = self._format_vector(vector_a)
        vector_b_str = self._format_vector(vector_b)
        result_str = self._format_vector(result)

        # Build addition step-by-step
        addition_elements = "; ".join([f"{vector_a[j]}+{vector_b[j]}" for j in range(self.dimension)])
        addition_str = f"mat({addition_elements})"

        # Add explanation for this subpart
        explanation.add_element(ContentAST.Paragraph([f"Part ({letter}):"]))
        explanation.add_element(
            ContentAST.Equation.make_block_equation__multiline_equals(
                lhs="arrow(a) + arrow(b)",  # Typst uses arrow() for vectors
                rhs=[
                    f"{vector_a_str} + {vector_b_str}",
                    addition_str,
                    result_str
                ]
            )
        )
    else:
      # Single part explanation - use Typst syntax
      vector_a_str = self._format_vector(self.vector_a)
      vector_b_str = self._format_vector(self.vector_b)
      result_str = self._format_vector(self.result)

      # Build addition step-by-step
      addition_elements = "; ".join([f"{self.vector_a[i]}+{self.vector_b[i]}" for i in range(self.dimension)])
      addition_str = f"mat({addition_elements})"

      explanation.add_element(
          ContentAST.Equation.make_block_equation__multiline_equals(
              lhs="arrow(a) + arrow(b)",
              rhs=[
                  f"{vector_a_str} + {vector_b_str}",
                  addition_str,
                  result_str
              ]
          )
      )

    return explanation, []  # Explanations don't have answers


@QuestionRegistry.register()
class VectorScalarMultiplication(VectorMathQuestion):

  MIN_DIMENSION = 2
  MAX_DIMENSION = 4

  def _generate_scalar(self):
    """Generate a non-zero scalar for multiplication."""
    scalar = self.rng.randint(-5, 5)
    while scalar == 0:  # Avoid zero scalar for more interesting problems
      scalar = self.rng.randint(-5, 5)
    return scalar

  def generate_operands(self):
    """Override to generate scalar and vector."""
    if not hasattr(self, 'dimension'):
      self.dimension = self.rng.randint(self.MIN_DIMENSION, self.MAX_DIMENSION)
    vector_a = self._generate_vector(self.dimension)
    vector_b = self._generate_vector(self.dimension)  # Not used, but kept for consistency
    return vector_a, vector_b

  def refresh(self, *args, **kwargs):
    if self.is_multipart():
      # For multipart questions, we handle everything ourselves
      # Don't call super() because we need different scalars per subpart

      # Call Question.refresh() directly to get basic setup
      Question.refresh(self, *args, **kwargs)

      # Generate vector dimension
      self.dimension = self.rng.randint(self.MIN_DIMENSION, self.MAX_DIMENSION)

      # Clear any existing data
      self.answers = {}

      # Generate multiple subquestions with their own scalars
      self.subquestion_data = []
      for i in range(self.num_subquestions):
        # Generate unique vectors and scalar for each subquestion
        vector_a = self._generate_vector(self.dimension)
        vector_b = self._generate_vector(self.dimension)  # Not used, but kept for consistency
        scalar = self._generate_scalar()
        result = [scalar * component for component in vector_a]

        self.subquestion_data.append({
          'operand_a': vector_a,
          'operand_b': vector_b,
          'vector_a': vector_a,
          'vector_b': vector_b,
          'scalar': scalar,
          'result': result
        })

        # Create answers for this subpart
        self.create_subquestion_answers(i, result)
    else:
      # For single questions, generate scalar first
      self.scalar = self._generate_scalar()
      # Then call super() normally
      super().refresh(*args, **kwargs)

  def get_operator(self):
    return f"{self.scalar} \\cdot"

  def calculate_single_result(self, vector_a, vector_b):
    # For scalar multiplication, we only use vector_a and ignore vector_b
    return [self.scalar * component for component in vector_a]

  def create_subquestion_answers(self, subpart_index, result):
    letter = chr(ord('a') + subpart_index)
    for j in range(len(result)):
      self.answers[f"subpart_{letter}_{j}"] = AnswerTypes.IntAnswer(result[j])

  def create_single_answers(self, result):
    for i in range(len(result)):
      self.answers[f"result_{i}"] = AnswerTypes.IntAnswer(result[i])

  def generate_subquestion_data(self):
    """Override to handle scalar multiplication format."""
    subparts = []
    for data in self.subquestion_data:
      vector_elem = self._format_vector(data['vector_a'])
      scalar = data['scalar']
      # Return MathExpression for format-independent rendering
      subparts.append(ContentAST.MathExpression([
          f"{scalar} \\cdot ",
          vector_elem
      ], inline=True))
    return subparts

  def get_body(self):
    """Override parent get_body() to use our custom formatting."""
    body, _ = self._get_body()
    return body

  def _get_body(self):
    """Build question body and collect answers (new pattern)."""
    body = ContentAST.Section()
    answers = []

    body.add_element(ContentAST.Paragraph([self.get_intro_text()]))

    if self.is_multipart():
      # Use multipart formatting with repeated problem parts
      subpart_data = self.generate_subquestion_data()
      repeated_part = self.create_repeated_problem_part(subpart_data)
      body.add_element(repeated_part)

      # Collect all subpart answers
      for i, data in enumerate(self.subquestion_data):
        letter = chr(ord('a') + i)
        result = data['result']
        for j in range(len(result)):
          ans = AnswerTypes.IntAnswer(result[j])
          answers.append(ans)
    else:
      # Single equation display using MathExpression
      vector_elem = self._format_vector(self.vector_a)
      body.add_element(ContentAST.MathExpression([
          f"{self.scalar} \\cdot ",
          vector_elem,
          " = "
      ]))

      # Canvas-only answer fields (hidden from PDF)
      body.add_element(ContentAST.OnlyHtml([ContentAST.Paragraph(["Enter your answer as a column vector:"])]))
      table_data = []
      for i in range(self.dimension):
        ans = AnswerTypes.IntAnswer(self.result[i])
        answers.append(ans)
        table_data.append([ans])
      body.add_element(ContentAST.OnlyHtml([ContentAST.Table(data=table_data, padding=True)]))

    return body, answers

  def _get_explanation(self):
    """Build question explanation (new pattern)."""
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph(["To multiply a vector by a scalar, we multiply each component by the scalar:"]))

    if self.is_multipart():
      # Handle multipart explanations
      for i, data in enumerate(self.subquestion_data):
        letter = chr(ord('a') + i)
        vector_a = data['vector_a']
        result = data['result']
        scalar = data['scalar']

        # Use LaTeX syntax (will be converted by Equation._latex_to_typst for Typst output)
        vector_str = r" \\ ".join([str(v) for v in vector_a])
        multiplication_str = r" \\ ".join([f"{scalar} \\cdot {v}" for v in vector_a])
        result_str = r" \\ ".join([str(v) for v in result])

        explanation.add_element(ContentAST.Paragraph([f"Part ({letter}):"]))
        explanation.add_element(
            ContentAST.Equation.make_block_equation__multiline_equals(
                lhs=f"{scalar} \\cdot \\vec{{v}}",
                rhs=[
                    f"{scalar} \\cdot \\begin{{bmatrix}} {vector_str} \\end{{bmatrix}}",
                    f"\\begin{{bmatrix}} {multiplication_str} \\end{{bmatrix}}",
                    f"\\begin{{bmatrix}} {result_str} \\end{{bmatrix}}"
                ]
            )
        )
    else:
      vector_str = r" \\ ".join([str(v) for v in self.vector_a])
      multiplication_str = r" \\ ".join([f"{self.scalar} \\cdot {v}" for v in self.vector_a])
      result_str = r" \\ ".join([str(v) for v in self.result])

      explanation.add_element(
          ContentAST.Equation.make_block_equation__multiline_equals(
              lhs=f"{self.scalar} \\cdot \\vec{{v}}",
              rhs=[
                  f"{self.scalar} \\cdot \\begin{{bmatrix}} {vector_str} \\end{{bmatrix}}",
                  f"\\begin{{bmatrix}} {multiplication_str} \\end{{bmatrix}}",
                  f"\\begin{{bmatrix}} {result_str} \\end{{bmatrix}}"
              ]
          )
      )

    return explanation, []  # Explanations don't have answers


@QuestionRegistry.register()
class VectorDotProduct(VectorMathQuestion):

  MIN_DIMENSION = 2
  MAX_DIMENSION = 4

  def get_operator(self):
    return "\\cdot"

  def calculate_single_result(self, vector_a, vector_b):
    return sum(vector_a[i] * vector_b[i] for i in range(len(vector_a)))

  def create_subquestion_answers(self, subpart_index, result):
    letter = chr(ord('a') + subpart_index)
    self.answers[f"subpart_{letter}"] = AnswerTypes.IntAnswer(result)

  def create_single_answers(self, result):
    self.answers = {
      "dot_product": AnswerTypes.IntAnswer(result)
    }

  def get_body(self):
    """Override parent get_body() to use our custom formatting."""
    body, _ = self._get_body()
    return body

  def _get_body(self):
    """Build question body and collect answers (new pattern)."""
    body = ContentAST.Section()
    answers = []

    body.add_element(ContentAST.Paragraph([self.get_intro_text()]))

    if self.is_multipart():
      # Use multipart formatting with repeated problem parts
      subpart_data = self.generate_subquestion_data()
      repeated_part = self.create_repeated_problem_part(subpart_data)
      body.add_element(repeated_part)

      # Collect all subpart answers (scalar results)
      for i, data in enumerate(self.subquestion_data):
        letter = chr(ord('a') + i)
        result = data['result']
        ans = AnswerTypes.IntAnswer(result)
        answers.append(ans)
    else:
      # Single equation display using MathExpression
      vector_a_elem = self._format_vector(self.vector_a)
      vector_b_elem = self._format_vector(self.vector_b)
      body.add_element(ContentAST.MathExpression([
          vector_a_elem,
          " \\cdot ",
          vector_b_elem,
          " = "
      ]))

      # Canvas-only answer field (single scalar result)
      ans = AnswerTypes.IntAnswer(self.result)
      answers.append(ans)
      body.add_element(ContentAST.OnlyHtml([ans]))

    return body, answers

  def _get_explanation(self):
    """Build question explanation (new pattern)."""
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph(["The dot product is calculated by multiplying corresponding components and summing the results:"]))

    if self.is_multipart():
      for i, data in enumerate(self.subquestion_data):
        letter = chr(ord('a') + i)
        vector_a = data['vector_a']
        vector_b = data['vector_b']
        result = data['result']

        vector_a_str = r" \\ ".join([str(v) for v in vector_a])
        vector_b_str = r" \\ ".join([str(v) for v in vector_b])
        products_str = " + ".join([f"({vector_a[j]} \\cdot {vector_b[j]})" for j in range(self.dimension)])
        calculation_str = " + ".join([str(vector_a[j] * vector_b[j]) for j in range(self.dimension)])

        explanation.add_element(ContentAST.Paragraph([f"Part ({letter}):"]))
        explanation.add_element(
            ContentAST.Equation.make_block_equation__multiline_equals(
                lhs="\\vec{a} \\cdot \\vec{b}",
                rhs=[
                    f"\\begin{{bmatrix}} {vector_a_str} \\end{{bmatrix}} \\cdot \\begin{{bmatrix}} {vector_b_str} \\end{{bmatrix}}",
                    products_str,
                    calculation_str,
                    str(result)
                ]
            )
        )
    else:
      vector_a_str = r" \\ ".join([str(v) for v in self.vector_a])
      vector_b_str = r" \\ ".join([str(v) for v in self.vector_b])
      products_str = " + ".join([f"({self.vector_a[i]} \\cdot {self.vector_b[i]})" for i in range(self.dimension)])
      calculation_str = " + ".join([str(self.vector_a[i] * self.vector_b[i]) for i in range(self.dimension)])

      explanation.add_element(
          ContentAST.Equation.make_block_equation__multiline_equals(
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
    # Magnitude uses ||...|| notation, not an operator between vectors
    return "||"

  def calculate_single_result(self, vector_a, vector_b):
    # For magnitude, we only use vector_a and ignore vector_b
    magnitude_squared = sum(component ** 2 for component in vector_a)
    return math.sqrt(magnitude_squared)

  def create_subquestion_answers(self, subpart_index, result):
    # Backward compatibility
    letter = chr(ord('a') + subpart_index)
    self.answers[f"subpart_{letter}"] = AnswerTypes.FloatAnswer(result)

  def create_single_answers(self, result):
    # Backward compatibility
    self.answers = {
      "magnitude": AnswerTypes.FloatAnswer(result)
    }

  def generate_subquestion_data(self):
    """Override to handle magnitude format ||vector||.

    Returns MathExpression elements for format-independent rendering.
    """
    subparts = []
    for data in self.subquestion_data:
      vector_elem = self._format_vector(data['vector_a'])
      # Create MathExpression for ||vector||
      subparts.append(ContentAST.MathExpression([
          "||",
          vector_elem,
          "||"
      ], inline=True))
    return subparts

  def get_body(self):
    """Override parent get_body() to use our custom formatting."""
    body, _ = self._get_body()
    return body

  def _get_body(self):
    """Build question body and collect answers (new pattern)."""
    from typing import Tuple, List

    body = ContentAST.Section()
    answers = []

    body.add_element(ContentAST.Paragraph([self.get_intro_text()]))

    if self.is_multipart():
      # Use multipart formatting with repeated problem parts
      subpart_data = self.generate_subquestion_data()
      repeated_part = self.create_repeated_problem_part(subpart_data)
      body.add_element(repeated_part)

      # Collect all subpart answers
      for i, data in enumerate(self.subquestion_data):
        letter = chr(ord('a') + i)
        result = data['result']
        ans = AnswerTypes.FloatAnswer(result)
        answers.append(ans)
    else:
      # Single equation display using MathExpression for format-independent rendering
      vector_elem = self._format_vector(self.vector_a)
      body.add_element(ContentAST.MathExpression([
          "||",
          vector_elem,
          "|| = "
      ]))

      # Canvas-only answer field (hidden from PDF)
      ans = AnswerTypes.FloatAnswer(self.result)
      answers.append(ans)
      body.add_element(ContentAST.OnlyHtml([ans]))

    return body, answers

  def _get_explanation(self):
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph(["The magnitude of a vector is calculated using the formula:"]))
    explanation.add_element(ContentAST.Equation("||arrow(v)|| = sqrt(v_1^2 + v_2^2 + ... + v_n^2)", inline=False))

    if self.is_multipart():
      # Handle multipart explanations
      for i, data in enumerate(self.subquestion_data):
        letter = chr(ord('a') + i)
        vector_a = data['vector_a']
        result = data['result']

        # Format vectors using Typst syntax
        vector_str = self._format_vector(vector_a)
        squares_str = " + ".join([f"{v}^2" for v in vector_a])
        calculation_str = " + ".join([str(v**2) for v in vector_a])
        sum_of_squares = sum(component ** 2 for component in vector_a)
        result_formatted = sorted(ContentAST.Answer.accepted_strings(result), key=lambda s: len(s))[0]

        # Add explanation for this subpart
        explanation.add_element(ContentAST.Paragraph([f"Part ({letter}):"]))
        explanation.add_element(
            ContentAST.Equation.make_block_equation__multiline_equals(
                lhs="||arrow(v)||",
                rhs=[
                    f"||{vector_str}||",
                    f"sqrt({squares_str})",
                    f"sqrt({calculation_str})",
                    f"sqrt({sum_of_squares})",
                    result_formatted
                ]
            )
        )
    else:
      # Single part explanation - use Typst syntax
      vector_str = self._format_vector(self.vector_a)
      squares_str = " + ".join([f"{v}^2" for v in self.vector_a])
      calculation_str = " + ".join([str(v**2) for v in self.vector_a])
      sum_of_squares = sum(component ** 2 for component in self.vector_a)
      result_formatted = sorted(ContentAST.Answer.accepted_strings(self.result), key=lambda s: len(s))[0]

      explanation.add_element(
          ContentAST.Equation.make_block_equation__multiline_equals(
              lhs="||arrow(v)||",
              rhs=[
                  f"||{vector_str}||",
                  f"sqrt({squares_str})",
                  f"sqrt({calculation_str})",
                  f"sqrt({sum_of_squares})",
                  result_formatted
              ]
          )
      )

    return explanation, []  # Explanations don't have answers
