#!env python
import abc
import logging
import math
from typing import List

from QuizGenerator.question import Question, QuestionRegistry, Answer
from QuizGenerator.misc import ContentAST

log = logging.getLogger(__name__)


class VectorMathQuestion(Question, abc.ABC):
  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.MATH)
    super().__init__(*args, **kwargs)

  def _generate_vector(self, dimension, min_val=-10, max_val=10):
    """Generate a vector with random integer values."""
    return [self.rng.randint(min_val, max_val) for _ in range(dimension)]

  def _format_vector(self, vector):
    """Format vector for display as column vector."""
    elements = [str(v) for v in vector]
    return f"\\begin{{pmatrix}} {' \\\\\\\\ '.join(elements)} \\end{{pmatrix}}"

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

    # Display the addition problem
    vector_a_latex = self._format_vector(self.vector_a)
    vector_b_latex = self._format_vector(self.vector_b)
    body.add_element(ContentAST.Equation(f"{vector_a_latex} + {vector_b_latex} = ", inline=False))

    # Answer section
    body.add_element(ContentAST.Paragraph(["Enter your answer as a column vector:"]))

    # Create answer table for vector components
    table_data = []
    for i in range(self.dimension):
      table_data.append([ContentAST.Answer(answer=self.answers[f"result_{i}"])])

    body.add_element(ContentAST.OnlyHtml([ContentAST.Table(data=table_data, padding=True)]))

    # For PDF, show as a single column vector answer
    pdf_answer_text = "\\begin{pmatrix} "
    for i in range(self.dimension):
      if i > 0:
        pdf_answer_text += " \\\\\\\\ "
      pdf_answer_text += f"\\underline{{\\hspace{{2cm}}}}"
    pdf_answer_text += " \\end{pmatrix}"
    body.add_element(ContentAST.OnlyLatex([ContentAST.Equation(pdf_answer_text, inline=False)]))

    return body

  def get_explanation(self):
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph(["To add vectors, we add corresponding components:"]))

    # Show step-by-step calculation
    vector_a_inline = self._format_vector_inline(self.vector_a)
    vector_b_inline = self._format_vector_inline(self.vector_b)
    result_inline = self._format_vector_inline(self.result)

    explanation.add_element(ContentAST.Paragraph([f"Vector A: {vector_a_inline}"]))
    explanation.add_element(ContentAST.Paragraph([f"Vector B: {vector_b_inline}"]))

    # Show component-wise addition
    for i in range(self.dimension):
      explanation.add_element(ContentAST.Paragraph([f"Component {i+1}: {self.vector_a[i]} + {self.vector_b[i]} = {self.result[i]}"]))

    explanation.add_element(ContentAST.Paragraph([f"Result: {result_inline}"]))

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

    # Display the multiplication problem
    vector_latex = self._format_vector(self.vector)
    body.add_element(ContentAST.Equation(f"{self.scalar} \\cdot {vector_latex} = ", inline=False))

    # Answer section
    body.add_element(ContentAST.Paragraph(["Enter your answer as a column vector:"]))

    # Create answer table for vector components
    table_data = []
    for i in range(self.dimension):
      table_data.append([ContentAST.Answer(answer=self.answers[f"result_{i}"])])

    body.add_element(ContentAST.OnlyHtml([ContentAST.Table(data=table_data, padding=True)]))

    # For PDF, show as a single column vector answer
    pdf_answer_text = "\\begin{pmatrix} "
    for i in range(self.dimension):
      if i > 0:
        pdf_answer_text += " \\\\\\\\ "
      pdf_answer_text += f"\\underline{{\\hspace{{2cm}}}}"
    pdf_answer_text += " \\end{pmatrix}"
    body.add_element(ContentAST.OnlyLatex([ContentAST.Equation(pdf_answer_text, inline=False)]))

    return body

  def get_explanation(self):
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph(["To multiply a vector by a scalar, we multiply each component by the scalar:"]))

    # Show step-by-step calculation
    vector_inline = self._format_vector_inline(self.vector)
    result_inline = self._format_vector_inline(self.result)

    explanation.add_element(ContentAST.Paragraph([f"Scalar: {self.scalar}"]))
    explanation.add_element(ContentAST.Paragraph([f"Vector: {vector_inline}"]))

    # Show component-wise multiplication
    for i in range(self.dimension):
      explanation.add_element(ContentAST.Paragraph([f"Component {i+1}: {self.scalar} × {self.vector[i]} = {self.result[i]}"]))

    explanation.add_element(ContentAST.Paragraph([f"Result: {result_inline}"]))

    return explanation


@QuestionRegistry.register()
class VectorDotProduct(VectorMathQuestion):

  MIN_DIMENSION = 2
  MAX_DIMENSION = 4

  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)

    # Generate vector dimension
    self.dimension = self.rng.randint(self.MIN_DIMENSION, self.MAX_DIMENSION)

    # Generate two vectors
    self.vector_a = self._generate_vector(self.dimension)
    self.vector_b = self._generate_vector(self.dimension)

    # Calculate dot product
    self.result = sum(self.vector_a[i] * self.vector_b[i] for i in range(self.dimension))

    # Create answer
    self.answers = {
      "dot_product": Answer.integer("dot_product", self.result)
    }

  def get_body(self):
    body = ContentAST.Section()

    # Randomly choose between "dot product" and "evaluate" phrasing
    if self.rng.random() < 0.5:
      body.add_element(ContentAST.Paragraph(["Calculate the dot product of the following vectors:"]))
    else:
      body.add_element(ContentAST.Paragraph(["Evaluate the following vector expression:"]))

    # Display the problem
    vector_a_latex = self._format_vector(self.vector_a)
    vector_b_latex = self._format_vector(self.vector_b)
    body.add_element(ContentAST.Equation(f"{vector_a_latex} \\cdot {vector_b_latex} = ", inline=False))

    # Answer section
    body.add_element(ContentAST.Paragraph(["Answer: "]))
    body.add_element(ContentAST.Answer(answer=self.answers["dot_product"]))

    return body

  def get_explanation(self):
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph(["The dot product is calculated by multiplying corresponding components and summing the results:"]))

    # Show vectors
    vector_a_inline = self._format_vector_inline(self.vector_a)
    vector_b_inline = self._format_vector_inline(self.vector_b)

    explanation.add_element(ContentAST.Paragraph([f"Vector A: {vector_a_inline}"]))
    explanation.add_element(ContentAST.Paragraph([f"Vector B: {vector_b_inline}"]))

    # Show component-wise multiplication
    calculation_parts = []
    for i in range(self.dimension):
      product = self.vector_a[i] * self.vector_b[i]
      calculation_parts.append(f"({self.vector_a[i]} × {self.vector_b[i]}) = {product}")

    explanation.add_element(ContentAST.Paragraph(["Calculation:"]))
    explanation.add_element(ContentAST.Paragraph([" + ".join(calculation_parts)]))
    explanation.add_element(ContentAST.Paragraph([f"= {self.result}"]))

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
      "magnitude": Answer.float_value("magnitude", self.result)
    }

  def get_body(self):
    body = ContentAST.Section()

    body.add_element(ContentAST.Paragraph(["Calculate the magnitude (length) of the following vector:"]))

    # Display the vector
    vector_latex = self._format_vector(self.vector)
    body.add_element(ContentAST.Equation(f"\\|{vector_latex}\\| = ", inline=False))

    # Answer section
    body.add_element(ContentAST.Paragraph(["Answer: "]))
    body.add_element(ContentAST.Answer(answer=self.answers["magnitude"]))

    return body

  def get_explanation(self):
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph(["The magnitude of a vector is calculated using the formula:"]))
    explanation.add_element(ContentAST.Equation("\\|\\vec{v}\\| = \\sqrt{v_1^2 + v_2^2 + \\ldots + v_n^2}", inline=False))

    # Show vector
    vector_inline = self._format_vector_inline(self.vector)
    explanation.add_element(ContentAST.Paragraph([f"Vector: {vector_inline}"]))

    # Show calculation
    squared_terms = []
    sum_of_squares = 0
    for i, component in enumerate(self.vector):
      squared_term = component ** 2
      squared_terms.append(f"{component}^2 = {squared_term}")
      sum_of_squares += squared_term

    explanation.add_element(ContentAST.Paragraph(["Calculation:"]))
    explanation.add_element(ContentAST.Paragraph([" + ".join(squared_terms)]))
    explanation.add_element(ContentAST.Paragraph([f"= {sum_of_squares}"]))

    # Format the final result cleanly
    result_formatted = sorted(Answer.accepted_strings(self.result), key=lambda s: len(s))[0]
    explanation.add_element(ContentAST.Paragraph([f"= √{sum_of_squares} = {result_formatted}"]))

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

    # Display the cross product problem
    vector_a_latex = self._format_vector(self.vector_a)
    vector_b_latex = self._format_vector(self.vector_b)
    body.add_element(ContentAST.Equation(f"{vector_a_latex} \\times {vector_b_latex} = ", inline=False))

    # Answer section
    body.add_element(ContentAST.Paragraph(["Enter your answer as a column vector:"]))

    # Create answer table for vector components
    table_data = []
    for i in range(self.dimension):
      table_data.append([ContentAST.Answer(answer=self.answers[f"result_{i}"])])

    body.add_element(ContentAST.OnlyHtml([ContentAST.Table(data=table_data, padding=True)]))

    # For PDF, show as a single column vector answer
    pdf_answer_text = "\\begin{pmatrix} "
    for i in range(self.dimension):
      if i > 0:
        pdf_answer_text += " \\\\\\\\ "
      pdf_answer_text += f"\\underline{{\\hspace{{2cm}}}}"
    pdf_answer_text += " \\end{pmatrix}"
    body.add_element(ContentAST.OnlyLatex([ContentAST.Equation(pdf_answer_text, inline=False)]))

    return body

  def get_explanation(self):
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph(["The cross product of two 3D vectors is calculated using the formula:"]))
    explanation.add_element(ContentAST.Equation("\\vec{a} \\times \\vec{b} = \\begin{pmatrix} a_2 b_3 - a_3 b_2 \\\\\\\\ a_3 b_1 - a_1 b_3 \\\\\\\\ a_1 b_2 - a_2 b_1 \\end{pmatrix}", inline=False))

    # Show vectors
    vector_a_inline = self._format_vector_inline(self.vector_a)
    vector_b_inline = self._format_vector_inline(self.vector_b)

    explanation.add_element(ContentAST.Paragraph([f"Vector A: {vector_a_inline}"]))
    explanation.add_element(ContentAST.Paragraph([f"Vector B: {vector_b_inline}"]))

    # Show component-wise calculation
    a1, a2, a3 = self.vector_a
    b1, b2, b3 = self.vector_b

    explanation.add_element(ContentAST.Paragraph(["Component calculations:"]))
    explanation.add_element(ContentAST.Paragraph([f"x-component: ({a2}) × ({b3}) - ({a3}) × ({b2}) = {a2*b3} - {a3*b2} = {self.result[0]}"]))
    explanation.add_element(ContentAST.Paragraph([f"y-component: ({a3}) × ({b1}) - ({a1}) × ({b3}) = {a3*b1} - {a1*b3} = {self.result[1]}"]))
    explanation.add_element(ContentAST.Paragraph([f"z-component: ({a1}) × ({b2}) - ({a2}) × ({b1}) = {a1*b2} - {a2*b1} = {self.result[2]}"]))

    result_inline = self._format_vector_inline(self.result)
    explanation.add_element(ContentAST.Paragraph([f"Result: {result_inline}"]))

    return explanation