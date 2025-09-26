#!env python
import abc
import logging
import math
from typing import List

from QuizGenerator.question import Question, QuestionRegistry, Answer
from QuizGenerator.contentast import ContentAST
from QuizGenerator.mixins import MultiPartQuestionMixin

log = logging.getLogger(__name__)


class VectorMathQuestion(Question, MultiPartQuestionMixin, abc.ABC):
  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.MATH)
    super().__init__(*args, **kwargs)

  def _generate_vector(self, dimension, min_val=-10, max_val=10):
    """Generate a vector with random integer values."""
    return [self.rng.randint(min_val, max_val) for _ in range(dimension)]

  def _format_vector(self, vector):
    """Format vector for display as column vector using ContentAST.Matrix."""
    # Convert to column matrix format
    matrix_data = [[v] for v in vector]
    return ContentAST.Matrix.to_latex(matrix_data, "b")

  def _format_vector_inline(self, vector):
    """Format vector for inline display."""
    elements = [str(v) for v in vector]
    return f"({', '.join(elements)})"

  def get_intro_text(self):
    """Default intro text - subclasses can override."""
    return "Calculate the following:"

  @abc.abstractmethod
  def get_operator(self):
    """Return the LaTeX operator for this operation (e.g., '+', '\\cdot', '\\times')."""
    pass

  @abc.abstractmethod
  def calculate_single_result(self, vector_a, vector_b):
    """Calculate the result for a single question with two vectors."""
    pass

  @abc.abstractmethod
  def create_subquestion_answers(self, subpart_index, result):
    """Create answer objects for a subquestion result."""
    pass

  def create_single_answers(self, result):
    """Create answers for single questions - just delegate to subquestion method."""
    return self.create_subquestion_answers(0, result)

  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)

    # Generate vector dimension
    self.dimension = self.rng.randint(self.MIN_DIMENSION, self.MAX_DIMENSION)

    # Clear any existing data
    self.answers = {}

    if self.is_multipart():
      # Generate multiple subquestions
      self.subquestion_data = []
      for i in range(self.num_subquestions):
        # Generate unique vectors for each subquestion
        vector_a = self._generate_vector(self.dimension)
        vector_b = self._generate_vector(self.dimension)
        result = self.calculate_single_result(vector_a, vector_b)

        self.subquestion_data.append({
          'vector_a': vector_a,
          'vector_b': vector_b,
          'result': result
        })

        # Create answers for this subpart
        self.create_subquestion_answers(i, result)
    else:
      # Single question (original behavior)
      self.vector_a = self._generate_vector(self.dimension)
      self.vector_b = self._generate_vector(self.dimension)
      self.result = self.calculate_single_result(self.vector_a, self.vector_b)

      # Create answers
      self.create_single_answers(self.result)

  def generate_subquestion_data(self):
    """Generate LaTeX content for each subpart of the question."""
    subparts = []
    for data in self.subquestion_data:
      vector_a_latex = self._format_vector(data['vector_a'])
      vector_b_latex = self._format_vector(data['vector_b'])
      # Return as tuple of (matrix_a, operator, matrix_b)
      subparts.append((vector_a_latex, self.get_operator(), vector_b_latex))
    return subparts

  def get_body(self):
    body = ContentAST.Section()

    body.add_element(ContentAST.Paragraph([self.get_intro_text()]))

    if self.is_multipart():
      # Use multipart formatting with repeated problem parts
      subpart_data = self.generate_subquestion_data()
      repeated_part = self.create_repeated_problem_part(subpart_data)
      body.add_element(repeated_part)
    else:
      # Single equation display
      vector_a_latex = self._format_vector(self.vector_a)
      vector_b_latex = self._format_vector(self.vector_b)
      body.add_element(ContentAST.Equation(f"{vector_a_latex} {self.get_operator()} {vector_b_latex} = ", inline=False))

      # Canvas-only answer fields (hidden from PDF)
      self._add_single_question_answers(body)

    return body

  def _add_single_question_answers(self, body):
    """Add Canvas-only answer fields for single questions. Subclasses can override."""
    # Check if it's a scalar result (like dot product)
    if hasattr(self, 'answers') and len(self.answers) == 1:
      # Single scalar answer
      answer_key = list(self.answers.keys())[0]
      body.add_element(ContentAST.OnlyHtml([ContentAST.Answer(answer=self.answers[answer_key])]))
    else:
      # Vector results (like addition) - show table
      body.add_element(ContentAST.OnlyHtml([ContentAST.Paragraph(["Enter your answer as a column vector:"])]))
      table_data = []
      for i in range(self.dimension):
        if f"result_{i}" in self.answers:
          table_data.append([ContentAST.Answer(answer=self.answers[f"result_{i}"])])
      if table_data:
        body.add_element(ContentAST.OnlyHtml([ContentAST.Table(data=table_data, padding=True)]))

  def get_explanation_intro(self):
    """Get the intro text for explanations. Subclasses should override."""
    return "The calculation is performed as follows:"

  def create_explanation_for_subpart(self, subpart_data, letter):
    """Create explanation for a single subpart. Subclasses should override."""
    return ContentAST.Paragraph([f"Part ({letter}): Calculation details would go here."])

  def get_explanation(self):
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph([self.get_explanation_intro()]))

    if self.is_multipart():
      # Handle multipart explanations
      for i, data in enumerate(self.subquestion_data):
        letter = chr(ord('a') + i)
        explanation.add_element(self.create_explanation_for_subpart(data, letter))
    else:
      # Single part explanation - subclasses should override this
      explanation.add_element(ContentAST.Paragraph(["Single question explanation would go here."]))

    return explanation


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
      self.answers[f"subpart_{letter}_{j}"] = Answer.integer(f"subpart_{letter}_{j}", result[j])

  def create_single_answers(self, result):
    for i in range(len(result)):
      self.answers[f"result_{i}"] = Answer.integer(f"result_{i}", result[i])

  def get_explanation(self):
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph(["To add vectors, we add corresponding components:"]))

    if self.is_multipart():
      # Handle multipart explanations
      for i, data in enumerate(self.subquestion_data):
        letter = chr(ord('a') + i)
        vector_a = data['vector_a']
        vector_b = data['vector_b']
        result = data['result']

        # Create LaTeX strings for multiline equation
        vector_a_str = r" \\ ".join([str(v) for v in vector_a])
        vector_b_str = r" \\ ".join([str(v) for v in vector_b])
        addition_str = r" \\ ".join([f"{vector_a[j]}+{vector_b[j]}" for j in range(self.dimension)])
        result_str = r" \\ ".join([str(v) for v in result])

        # Add explanation for this subpart
        explanation.add_element(ContentAST.Paragraph([f"Part ({letter}):"]))
        explanation.add_element(
            ContentAST.Equation.make_block_equation__multiline_equals(
                lhs="\\vec{a} + \\vec{b}",
                rhs=[
                    f"\\begin{{bmatrix}} {vector_a_str} \\end{{bmatrix}} + \\begin{{bmatrix}} {vector_b_str} \\end{{bmatrix}}",
                    f"\\begin{{bmatrix}} {addition_str} \\end{{bmatrix}}",
                    f"\\begin{{bmatrix}} {result_str} \\end{{bmatrix}}"
                ]
            )
        )
    else:
      # Single part explanation (original behavior)
      vector_a_str = r" \\ ".join([str(v) for v in self.vector_a])
      vector_b_str = r" \\ ".join([str(v) for v in self.vector_b])
      addition_str = r" \\ ".join([f"{self.vector_a[i]}+{self.vector_b[i]}" for i in range(self.dimension)])
      result_str = r" \\ ".join([str(v) for v in self.result])

      explanation.add_element(
          ContentAST.Equation.make_block_equation__multiline_equals(
              lhs="\\vec{a} + \\vec{b}",
              rhs=[
                  f"\\begin{{bmatrix}} {vector_a_str} \\end{{bmatrix}} + \\begin{{bmatrix}} {vector_b_str} \\end{{bmatrix}}",
                  f"\\begin{{bmatrix}} {addition_str} \\end{{bmatrix}}",
                  f"\\begin{{bmatrix}} {result_str} \\end{{bmatrix}}"
              ]
          )
      )

    return explanation


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

  def refresh(self, *args, **kwargs):
    if self.is_multipart():
      # For multipart questions, we handle everything ourselves
      # Don't call super() because it would try to use calculate_single_result without scalars

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
      self.answers[f"subpart_{letter}_{j}"] = Answer.integer(f"subpart_{letter}_{j}", result[j])

  def create_single_answers(self, result):
    for i in range(len(result)):
      self.answers[f"result_{i}"] = Answer.integer(f"result_{i}", result[i])

  def generate_subquestion_data(self):
    """Override to handle scalar multiplication format."""
    subparts = []
    for data in self.subquestion_data:
      vector_a_latex = self._format_vector(data['vector_a'])
      # For scalar multiplication, we show scalar * vector as a single string
      # Use the scalar from this specific subquestion's data
      scalar = data['scalar']
      subparts.append(f"{scalar} \\cdot {vector_a_latex}")
    return subparts

  def get_body(self):
    body = ContentAST.Section()

    body.add_element(ContentAST.Paragraph([self.get_intro_text()]))

    if self.is_multipart():
      # Use multipart formatting with repeated problem parts
      subpart_data = self.generate_subquestion_data()
      repeated_part = self.create_repeated_problem_part(subpart_data)
      body.add_element(repeated_part)
    else:
      # Single equation display
      vector_a_latex = self._format_vector(self.vector_a)
      body.add_element(ContentAST.Equation(f"{self.scalar} \\cdot {vector_a_latex} = ", inline=False))

      # Canvas-only answer fields (hidden from PDF)
      self._add_single_question_answers(body)

    return body

  def get_explanation(self):
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph(["To multiply a vector by a scalar, we multiply each component by the scalar:"]))

    if self.is_multipart():
      # Handle multipart explanations
      for i, data in enumerate(self.subquestion_data):
        letter = chr(ord('a') + i)
        vector_a = data['vector_a']
        result = data['result']

        # Get the scalar for this specific subpart
        scalar = data['scalar']

        # Create LaTeX strings for multiline equation
        vector_str = r" \\ ".join([str(v) for v in vector_a])
        multiplication_str = r" \\ ".join([f"{scalar} \\cdot {v}" for v in vector_a])
        result_str = r" \\ ".join([str(v) for v in result])

        # Add explanation for this subpart
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
      # Single part explanation - use the correct attributes
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

    return explanation


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
    self.answers[f"subpart_{letter}"] = Answer.integer(f"subpart_{letter}", result)

  def create_single_answers(self, result):
    self.answers = {
      "dot_product": Answer.integer("dot_product", result)
    }

  def get_explanation(self):
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph(["The dot product is calculated by multiplying corresponding components and summing the results:"]))

    if self.is_multipart():
      # Handle multipart explanations
      for i, data in enumerate(self.subquestion_data):
        letter = chr(ord('a') + i)
        vector_a = data['vector_a']
        vector_b = data['vector_b']
        result = data['result']

        # Create LaTeX strings for multiline equation
        vector_a_str = r" \\ ".join([str(v) for v in vector_a])
        vector_b_str = r" \\ ".join([str(v) for v in vector_b])
        products_str = " + ".join([f"({vector_a[j]} \\cdot {vector_b[j]})" for j in range(self.dimension)])
        calculation_str = " + ".join([str(vector_a[j] * vector_b[j]) for j in range(self.dimension)])

        # Add explanation for this subpart
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
      # Single part explanation (original behavior)
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

    return explanation


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
    letter = chr(ord('a') + subpart_index)
    self.answers[f"subpart_{letter}"] = Answer.auto_float(f"subpart_{letter}", result)

  def create_single_answers(self, result):
    self.answers = {
      "magnitude": Answer.auto_float("magnitude", result)
    }

  def generate_subquestion_data(self):
    """Override to handle magnitude format ||vector||."""
    subparts = []
    for data in self.subquestion_data:
      vector_a_latex = self._format_vector(data['vector_a'])
      # For magnitude, we show ||vector|| as a single string
      subparts.append(f"\\left\\|{vector_a_latex}\\right\\|")
    return subparts

  def get_body(self):
    body = ContentAST.Section()

    body.add_element(ContentAST.Paragraph([self.get_intro_text()]))

    if self.is_multipart():
      # Use multipart formatting with repeated problem parts
      subpart_data = self.generate_subquestion_data()
      repeated_part = self.create_repeated_problem_part(subpart_data)
      body.add_element(repeated_part)
    else:
      # Single equation display
      vector_a_latex = self._format_vector(self.vector_a)
      body.add_element(ContentAST.Equation(f"\\left\\|{vector_a_latex}\\right\\| = ", inline=False))

      # Canvas-only answer field (hidden from PDF)
      self._add_single_question_answers(body)

    return body

  def get_explanation(self):
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph(["The magnitude of a vector is calculated using the formula:"]))
    explanation.add_element(ContentAST.Equation("\\left\\|\\vec{v}\\right\\| = \\sqrt{v_1^2 + v_2^2 + \\ldots + v_n^2}", inline=False))

    if self.is_multipart():
      # Handle multipart explanations
      for i, data in enumerate(self.subquestion_data):
        letter = chr(ord('a') + i)
        vector_a = data['vector_a']
        result = data['result']

        # Create LaTeX strings for multiline equation
        vector_str = r" \\ ".join([str(v) for v in vector_a])
        squares_str = " + ".join([f"{v}^2" for v in vector_a])
        calculation_str = " + ".join([str(v**2) for v in vector_a])
        sum_of_squares = sum(component ** 2 for component in vector_a)
        result_formatted = sorted(Answer.accepted_strings(result), key=lambda s: len(s))[0]

        # Add explanation for this subpart
        explanation.add_element(ContentAST.Paragraph([f"Part ({letter}):"]))
        explanation.add_element(
            ContentAST.Equation.make_block_equation__multiline_equals(
                lhs="\\left\\|\\vec{v}\\right\\|",
                rhs=[
                    f"\\left\\|\\begin{{bmatrix}} {vector_str} \\end{{bmatrix}}\\right\\|",
                    f"\\sqrt{{{squares_str}}}",
                    f"\\sqrt{{{calculation_str}}}",
                    f"\\sqrt{{{sum_of_squares}}}",
                    result_formatted
                ]
            )
        )
    else:
      # Single part explanation - use the correct attributes
      vector_str = r" \\ ".join([str(v) for v in self.vector_a])
      squares_str = " + ".join([f"{v}^2" for v in self.vector_a])
      calculation_str = " + ".join([str(v**2) for v in self.vector_a])
      sum_of_squares = sum(component ** 2 for component in self.vector_a)
      result_formatted = sorted(Answer.accepted_strings(self.result), key=lambda s: len(s))[0]

      explanation.add_element(
          ContentAST.Equation.make_block_equation__multiline_equals(
              lhs="\\left\\|\\vec{v}\\right\\|",
              rhs=[
                  f"\\left\\|\\begin{{bmatrix}} {vector_str} \\end{{bmatrix}}\\right\\|",
                  f"\\sqrt{{{squares_str}}}",
                  f"\\sqrt{{{calculation_str}}}",
                  f"\\sqrt{{{sum_of_squares}}}",
                  result_formatted
              ]
          )
      )

    return explanation
