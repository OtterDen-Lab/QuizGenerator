#!env python
import abc
import logging
import math
from typing import List

from QuizGenerator.question import Question, QuestionRegistry, Answer
from QuizGenerator.contentast import ContentAST
from QuizGenerator.mixins import MultiPartQuestionMixin

log = logging.getLogger(__name__)


class VectorMathQuestion(Question, abc.ABC):
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


@QuestionRegistry.register()
class VectorAddition(VectorMathQuestion):

  MIN_DIMENSION = 2
  MAX_DIMENSION = 4

  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)

    # Generate vector dimension
    self.dimension = self.rng.randint(self.MIN_DIMENSION, self.MAX_DIMENSION)

    # Generate two vectors
    self.vector_a = self._generate_vector(self.dimension)
    self.vector_b = self._generate_vector(self.dimension)

    # Calculate result
    self.result = [self.vector_a[i] + self.vector_b[i] for i in range(self.dimension)]

    # Create answers
    self.answers = {}
    for i in range(self.dimension):
      self.answers[f"result_{i}"] = Answer.integer(f"result_{i}", self.result[i])

  def get_body(self):
    body = ContentAST.Section()

    body.add_element(ContentAST.Paragraph(["Calculate the vector addition:"]))

    # Display the addition problem as a single equation
    vector_a_latex = self._format_vector(self.vector_a)
    vector_b_latex = self._format_vector(self.vector_b)
    body.add_element(ContentAST.Equation(f"{vector_a_latex} + {vector_b_latex} = ", inline=False))

    # Answer section
    body.add_element(ContentAST.OnlyHtml([ContentAST.Paragraph(["Enter your answer as a column vector:"])]))

    # Create answer table for vector components
    table_data = []
    for i in range(self.dimension):
      table_data.append([ContentAST.Answer(answer=self.answers[f"result_{i}"])])

    body.add_element(ContentAST.OnlyHtml([ContentAST.Table(data=table_data, padding=True)]))


    return body

  def get_explanation(self):
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph(["To add vectors, we add corresponding components:"]))

    # Create LaTeX strings for multiline equation
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

  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)

    # Generate vector dimension
    self.dimension = self.rng.randint(self.MIN_DIMENSION, self.MAX_DIMENSION)

    # Generate vector and scalar
    self.vector = self._generate_vector(self.dimension)
    self.scalar = self.rng.randint(-5, 5)
    while self.scalar == 0:  # Avoid zero scalar for more interesting problems
      self.scalar = self.rng.randint(-5, 5)

    # Calculate result
    self.result = [self.scalar * component for component in self.vector]

    # Create answers
    self.answers = {}
    for i in range(self.dimension):
      self.answers[f"result_{i}"] = Answer.integer(f"result_{i}", self.result[i])

  def get_body(self):
    body = ContentAST.Section()

    body.add_element(ContentAST.Paragraph(["Calculate the scalar multiplication:"]))

    # Display the multiplication problem as a single equation
    vector_latex = self._format_vector(self.vector)
    body.add_element(ContentAST.Equation(f"{self.scalar} \\cdot {vector_latex} = ", inline=False))

    # Answer section
    body.add_element(ContentAST.OnlyHtml([ContentAST.Paragraph(["Enter your answer as a column vector:"])]))

    # Create answer table for vector components
    table_data = []
    for i in range(self.dimension):
      table_data.append([ContentAST.Answer(answer=self.answers[f"result_{i}"])])

    body.add_element(ContentAST.OnlyHtml([ContentAST.Table(data=table_data, padding=True)]))


    return body

  def get_explanation(self):
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph(["To multiply a vector by a scalar, we multiply each component by the scalar:"]))

    # Create LaTeX strings for multiline equation
    vector_str = r" \\ ".join([str(v) for v in self.vector])
    multiplication_str = r" \\ ".join([f"{self.scalar} \\cdot {v}" for v in self.vector])
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
class VectorDotProduct(VectorMathQuestion, MultiPartQuestionMixin):

  MIN_DIMENSION = 2
  MAX_DIMENSION = 4

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
        result = sum(vector_a[j] * vector_b[j] for j in range(self.dimension))

        self.subquestion_data.append({
          'vector_a': vector_a,
          'vector_b': vector_b,
          'result': result
        })

        # Create answer for this subpart
        letter = chr(ord('a') + i)
        self.answers[f"subpart_{letter}"] = Answer.integer(f"subpart_{letter}", result)
    else:
      # Single question (original behavior)
      self.vector_a = self._generate_vector(self.dimension)
      self.vector_b = self._generate_vector(self.dimension)
      self.result = sum(self.vector_a[i] * self.vector_b[i] for i in range(self.dimension))
      self.answers = {
        "dot_product": Answer.integer("dot_product", self.result)
      }

  def generate_subquestion_data(self):
    """Generate LaTeX content for each subpart of the dot product question."""
    subparts = []
    for data in self.subquestion_data:
      vector_a_latex = self._format_vector(data['vector_a'])
      vector_b_latex = self._format_vector(data['vector_b'])
      # Return as tuple of (matrix_a, operator, matrix_b)
      subparts.append((vector_a_latex, "\\cdot", vector_b_latex))
    return subparts

  def get_body(self):
    body = ContentAST.Section()

    # Use consistent wording for both single and multipart questions
    # Randomly choose between "dot product" and "evaluate" phrasing
    if self.rng.random() < 0.5:
      intro_text = "Calculate the dot product of the following vectors:"
    else:
      intro_text = "Evaluate the following vector expression:"

    body.add_element(ContentAST.Paragraph([intro_text]))

    if self.is_multipart():
      # Use multipart formatting with repeated problem parts
      subpart_data = self.generate_subquestion_data()
      repeated_part = self.create_repeated_problem_part(subpart_data)
      body.add_element(repeated_part)
    else:
      # Single equation display
      vector_a_latex = self._format_vector(self.vector_a)
      vector_b_latex = self._format_vector(self.vector_b)
      body.add_element(ContentAST.Equation(f"{vector_a_latex} \\cdot {vector_b_latex} = ", inline=False))

      # Canvas-only answer field (hidden from PDF)
      body.add_element(ContentAST.OnlyHtml([ContentAST.Answer(answer=self.answers["dot_product"])]))

    return body

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

  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)

    # Generate vector dimension
    self.dimension = self.rng.randint(self.MIN_DIMENSION, self.MAX_DIMENSION)

    # Generate vector
    self.vector = self._generate_vector(self.dimension, min_val=-5, max_val=5)

    # Calculate magnitude
    magnitude_squared = sum(component ** 2 for component in self.vector)
    self.result = math.sqrt(magnitude_squared)

    # Create answer - use float_value for proper rounding
    self.answers = {
      "magnitude": Answer.auto_float("magnitude", self.result)
    }

  def get_body(self):
    body = ContentAST.Section()

    body.add_element(ContentAST.Paragraph(["Calculate the magnitude of the following vector:"]))

    # Display the vector as a single equation
    vector_latex = self._format_vector(self.vector)
    body.add_element(ContentAST.Equation(f"\\left\\|{vector_latex}\\right\\| = ", inline=False))

    # Canvas-only answer field (hidden from PDF)
    body.add_element(ContentAST.OnlyHtml([ContentAST.Answer(answer=self.answers["magnitude"])]))

    return body

  def get_explanation(self):
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph(["The magnitude of a vector is calculated using the formula:"]))
    explanation.add_element(ContentAST.Equation("\\left\\|\\vec{v}\\right\\| = \\sqrt{v_1^2 + v_2^2 + \\ldots + v_n^2}", inline=False))

    # Create LaTeX strings for multiline equation
    vector_str = r" \\ ".join([str(v) for v in self.vector])
    squares_str = " + ".join([f"{v}^2" for v in self.vector])
    calculation_str = " + ".join([str(v**2) for v in self.vector])
    sum_of_squares = sum(component ** 2 for component in self.vector)
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


@QuestionRegistry.register()
class VectorCrossProduct(VectorMathQuestion):

  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)

    # Cross product is only defined for 3D vectors
    self.dimension = 3

    # Generate two 3D vectors
    self.vector_a = self._generate_vector(self.dimension, min_val=-5, max_val=5)
    self.vector_b = self._generate_vector(self.dimension, min_val=-5, max_val=5)

    # Calculate cross product: a × b = (a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1)
    self.result = [
      self.vector_a[1] * self.vector_b[2] - self.vector_a[2] * self.vector_b[1],
      self.vector_a[2] * self.vector_b[0] - self.vector_a[0] * self.vector_b[2],
      self.vector_a[0] * self.vector_b[1] - self.vector_a[1] * self.vector_b[0]
    ]

    # Create answers
    self.answers = {}
    for i in range(self.dimension):
      self.answers[f"result_{i}"] = Answer.integer(f"result_{i}", self.result[i])

  def get_body(self):
    body = ContentAST.Section()

    body.add_element(ContentAST.Paragraph(["Calculate the cross product of the following 3D vectors:"]))

    # Display the cross product problem as a single equation
    vector_a_latex = self._format_vector(self.vector_a)
    vector_b_latex = self._format_vector(self.vector_b)
    body.add_element(ContentAST.Equation(f"{vector_a_latex} \\times {vector_b_latex} = ", inline=False))

    # Answer section
    body.add_element(ContentAST.OnlyHtml([ContentAST.Paragraph(["Enter your answer as a column vector:"])]))

    # Create answer table for vector components
    table_data = []
    for i in range(self.dimension):
      table_data.append([ContentAST.Answer(answer=self.answers[f"result_{i}"])])

    body.add_element(ContentAST.OnlyHtml([ContentAST.Table(data=table_data, padding=True)]))


    return body

  def get_explanation(self):
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph(["The cross product of two 3D vectors is calculated using the formula:"]))
    explanation.add_element(ContentAST.Equation("\\vec{a} \\times \\vec{b} = \\begin{bmatrix} a_2 b_3 - a_3 b_2 \\\\ a_3 b_1 - a_1 b_3 \\\\ a_1 b_2 - a_2 b_1 \\end{bmatrix}", inline=False))

    # Create LaTeX strings for multiline equation
    a1, a2, a3 = self.vector_a
    b1, b2, b3 = self.vector_b

    vector_a_str = r" \\ ".join([str(v) for v in self.vector_a])
    vector_b_str = r" \\ ".join([str(v) for v in self.vector_b])
    formula_str = f"{a2} \\cdot {b3} - {a3} \\cdot {b2} \\\\\\\\ {a3} \\cdot {b1} - {a1} \\cdot {b3} \\\\\\\\ {a1} \\cdot {b2} - {a2} \\cdot {b1}"
    calculation_str = f"{a2*b3} - {a3*b2} \\\\\\\\ {a3*b1} - {a1*b3} \\\\\\\\ {a1*b2} - {a2*b1}"
    result_str = r" \\ ".join([str(v) for v in self.result])

    explanation.add_element(
        ContentAST.Equation.make_block_equation__multiline_equals(
            lhs="\\vec{a} \\times \\vec{b}",
            rhs=[
                f"\\begin{{bmatrix}} {vector_a_str} \\end{{bmatrix}} \\times \\begin{{bmatrix}} {vector_b_str} \\end{{bmatrix}}",
                f"\\begin{{bmatrix}} {formula_str} \\end{{bmatrix}}",
                f"\\begin{{bmatrix}} {calculation_str} \\end{{bmatrix}}",
                f"\\begin{{bmatrix}} {result_str} \\end{{bmatrix}}"
            ]
        )
    )

    return explanation
