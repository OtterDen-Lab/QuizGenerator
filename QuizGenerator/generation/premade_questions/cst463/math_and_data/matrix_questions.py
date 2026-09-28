#!/usr/bin/env python
import logging

import QuizGenerator.generation.contentast as ca
from QuizGenerator.generation.question import Question, QuestionRegistry

log = logging.getLogger(__name__)


class MatrixMathQuestion(Question):
  """Base class for configurable matrix mathematics questions."""

  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.MATH)
    super().__init__(*args, **kwargs)

  @staticmethod
  def _generate_matrix(rng, rows, cols, min_val=1, max_val=9):
    """Generate a matrix with random integer values."""
    return [[rng.randint(min_val, max_val) for _ in range(cols)] for _ in range(rows)]

  @classmethod
  def _build_matrix_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    min_size = kwargs.get("min_size", cls.MIN_SIZE)
    max_size = kwargs.get("max_size", cls.MAX_SIZE)
    if min_size < 1 or max_size < min_size:
      raise ValueError("min_size and max_size must describe a positive size range")
    return context, min_size, max_size

  @staticmethod
  def _matrix_to_table(matrix, prefix=""):
    """Convert a matrix to content AST table format."""
    return [[f"{prefix}{matrix[i][j]}" for j in range(len(matrix[0]))] for i in range(len(matrix))]

  @staticmethod
  def _create_answer_table(answer_matrix):
    """Create a Canvas answer table while preserving PDF workspace."""
    table_data = []
    answers = []
    for row in answer_matrix:
      table_row = []
      for ans in row:
        table_row.append(ans)
        if isinstance(ans, ca.Answer):
          answers.append(ans)
      table_data.append(table_row)
    return ca.Table(data=table_data, padding=True), answers


@QuestionRegistry.register()
class MatrixAddition(MatrixMathQuestion):

    MIN_SIZE = 2
    MAX_SIZE = 4

    @classmethod
    def _build_context(cls, *, rng_seed=None, **kwargs):
      context, min_size, max_size = cls._build_matrix_context(
        rng_seed=rng_seed,
        **kwargs,
      )
      rows = kwargs.get("rows", context.rng.randint(min_size, max_size))
      cols = kwargs.get("cols", context.rng.randint(min_size, max_size))
      if not min_size <= rows <= max_size or not min_size <= cols <= max_size:
        raise ValueError("rows and cols must be within min_size and max_size")

      min_value = kwargs.get("min_value", 1)
      max_value = kwargs.get("max_value", 9)
      matrix_a = cls._generate_matrix(context.rng, rows, cols, min_value, max_value)
      matrix_b = cls._generate_matrix(context.rng, rows, cols, min_value, max_value)
      context["rows"] = rows
      context["cols"] = cols
      context["matrix_a"] = matrix_a
      context["matrix_b"] = matrix_b
      context["result"] = [
        [matrix_a[i][j] + matrix_b[i][j] for j in range(cols)]
        for i in range(rows)
      ]
      return context

    @classmethod
    def _build_body(cls, context):
        body = ca.Section()
        body.add_element(ca.Paragraph(["Calculate the following:"]))

        matrix_a_elem = ca.Matrix(data=context["matrix_a"], bracket_type="b")
        matrix_b_elem = ca.Matrix(data=context["matrix_b"], bracket_type="b")
        body.add_element(ca.MathExpression([matrix_a_elem, " + ", matrix_b_elem, " = "]))

        answer_matrix = [
            [ca.AnswerTypes.Int(value) for value in row]
            for row in context["result"]
        ]
        table, table_answers = cls._create_answer_table(answer_matrix)
        body.add_element(
            ca.OnlyHtml([
                ca.Paragraph(["Result matrix:"]),
                table
            ])
        )

        return body, table_answers

    @classmethod
    def _build_explanation(cls, context) -> tuple[ca.Section, list[ca.Answer]]:
        explanation = ca.Section()

        explanation.add_element(
            ca.Paragraph([
                "Matrix addition is performed element-wise. Each element in the result matrix "
                "is the sum of the corresponding elements in the input matrices."
            ])
        )

        explanation.add_element(ca.Paragraph(["Step-by-step calculation:"]))

        # Create properly formatted matrix strings
        matrix_a_str = r" \\ ".join([
            " & ".join([str(context["matrix_a"][i][j]) for j in range(context["cols"])])
            for i in range(context["rows"])
        ])
        matrix_b_str = r" \\ ".join([
            " & ".join([str(context["matrix_b"][i][j]) for j in range(context["cols"])])
            for i in range(context["rows"])
        ])
        addition_str = r" \\ ".join([
            " & ".join([f"{context['matrix_a'][i][j]}+{context['matrix_b'][i][j]}" for j in range(context["cols"])])
            for i in range(context["rows"])
        ])
        result_str = r" \\ ".join([
            " & ".join([str(context["result"][i][j]) for j in range(context["cols"])])
            for i in range(context["rows"])
        ])

        explanation.add_element(
            ca.Equation.make_block_equation__multiline_equals(
                lhs="A + B",
                rhs=[
                    f"\\begin{{bmatrix}} {matrix_a_str} \\end{{bmatrix}} + \\begin{{bmatrix}} {matrix_b_str} \\end{{bmatrix}}",
                    f"\\begin{{bmatrix}} {addition_str} \\end{{bmatrix}}",
                    f"\\begin{{bmatrix}} {result_str} \\end{{bmatrix}}"
                ]
            )
        )

        return explanation, []


@QuestionRegistry.register()
class MatrixScalarMultiplication(MatrixMathQuestion):

    MIN_SIZE = 2
    MAX_SIZE = 4
    MIN_SCALAR = 2
    MAX_SCALAR = 9

    @staticmethod
    def _generate_scalar(rng, min_scalar, max_scalar):
        """Generate a scalar for multiplication."""
        return rng.randint(min_scalar, max_scalar)

    @classmethod
    def _build_context(cls, *, rng_seed=None, **kwargs):
      context, min_size, max_size = cls._build_matrix_context(
        rng_seed=rng_seed,
        **kwargs,
      )
      rows = kwargs.get("rows", context.rng.randint(min_size, max_size))
      cols = kwargs.get("cols", context.rng.randint(min_size, max_size))
      if not min_size <= rows <= max_size or not min_size <= cols <= max_size:
        raise ValueError("rows and cols must be within min_size and max_size")

      min_value = kwargs.get("min_value", 1)
      max_value = kwargs.get("max_value", 9)
      min_scalar = kwargs.get("min_scalar", cls.MIN_SCALAR)
      max_scalar = kwargs.get("max_scalar", cls.MAX_SCALAR)
      if min_scalar > max_scalar:
        raise ValueError("min_scalar must not exceed max_scalar")

      matrix = cls._generate_matrix(context.rng, rows, cols, min_value, max_value)
      scalar = kwargs.get(
        "scalar",
        cls._generate_scalar(context.rng, min_scalar, max_scalar),
      )
      context["rows"] = rows
      context["cols"] = cols
      context["matrix"] = matrix
      context["scalar"] = scalar
      context["result"] = [
        [scalar * matrix[i][j] for j in range(cols)]
        for i in range(rows)
      ]
      return context

    @classmethod
    def _build_body(cls, context):
        body = ca.Section()
        body.add_element(ca.Paragraph(["Calculate the following:"]))

        matrix_elem = ca.Matrix(data=context["matrix"], bracket_type="b")
        body.add_element(ca.MathExpression([f"{context['scalar']} \\cdot ", matrix_elem, " = "]))

        answer_matrix = [
            [ca.AnswerTypes.Int(value) for value in row]
            for row in context["result"]
        ]
        table, table_answers = cls._create_answer_table(answer_matrix)
        body.add_element(
            ca.OnlyHtml([
                ca.Paragraph(["Result matrix:"]),
                table
            ])
        )

        return body, table_answers

    @classmethod
    def _build_explanation(cls, context) -> tuple[ca.Section, list[ca.Answer]]:
        explanation = ca.Section()

        explanation.add_element(
            ca.Paragraph([
                "Scalar multiplication involves multiplying every element in the matrix by the scalar value."
            ])
        )

        explanation.add_element(ca.Paragraph(["Step-by-step calculation:"]))

        matrix_str = r" \\ ".join([
            " & ".join([str(context["matrix"][row][col]) for col in range(context["cols"])])
            for row in range(context["rows"])
        ])
        multiplication_str = r" \\ ".join([
            " & ".join([f"{context['scalar']} \\cdot {context['matrix'][row][col]}" for col in range(context["cols"])])
            for row in range(context["rows"])
        ])
        result_str = r" \\ ".join([
            " & ".join([str(context["result"][row][col]) for col in range(context["cols"])])
            for row in range(context["rows"])
        ])

        explanation.add_element(
            ca.Equation.make_block_equation__multiline_equals(
                lhs=f"{context['scalar']} \\cdot A",
                rhs=[
                    f"{context['scalar']} \\cdot \\begin{{bmatrix}} {matrix_str} \\end{{bmatrix}}",
                    f"\\begin{{bmatrix}} {multiplication_str} \\end{{bmatrix}}",
                    f"\\begin{{bmatrix}} {result_str} \\end{{bmatrix}}"
                ]
            )
        )

        return explanation, []


@QuestionRegistry.register()
class MatrixMultiplication(MatrixMathQuestion):
  """Compute a compatible matrix product of manageable size."""

  MIN_SIZE = 2
  MAX_SIZE = 3

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context, min_size, max_size = cls._build_matrix_context(
      rng_seed=rng_seed,
      **kwargs,
    )
    rows_a = kwargs.get("rows_a", context.rng.randint(min_size, max_size))
    cols_a = kwargs.get("cols_a", context.rng.randint(min_size, max_size))
    rows_b = kwargs.get("rows_b", cols_a)
    cols_b = kwargs.get("cols_b", context.rng.randint(min_size, max_size))
    if cols_a != rows_b:
      raise ValueError("MatrixMultiplication requires cols_a to equal rows_b")
    if not all(min_size <= size <= max_size for size in (rows_a, cols_a, rows_b, cols_b)):
      raise ValueError("matrix dimensions must be within min_size and max_size")

    min_value = kwargs.get("min_value", 1)
    max_value = kwargs.get("max_value", 9)
    matrix_a = cls._generate_matrix(context.rng, rows_a, cols_a, min_value, max_value)
    matrix_b = cls._generate_matrix(context.rng, rows_b, cols_b, min_value, max_value)
    context["rows_a"] = rows_a
    context["cols_a"] = cols_a
    context["rows_b"] = rows_b
    context["cols_b"] = cols_b
    context["matrix_a"] = matrix_a
    context["matrix_b"] = matrix_b
    context["result"] = [
      [sum(matrix_a[i][k] * matrix_b[k][j] for k in range(cols_a))
       for j in range(cols_b)]
      for i in range(rows_a)
    ]
    return context

  @classmethod
  def _build_body(cls, context):
    body = ca.Section()
    body.add_element(ca.Paragraph(["Compute the matrix product."]))
    body.add_element(ca.MathExpression([
      ca.Matrix(data=context["matrix_a"], bracket_type="b"),
      r" \cdot ",
      ca.Matrix(data=context["matrix_b"], bracket_type="b"),
      " = ",
    ]))
    answer_matrix = [
      [ca.AnswerTypes.Int(value) for value in row]
      for row in context["result"]
    ]
    table, answers = cls._create_answer_table(answer_matrix)
    body.add_element(ca.OnlyHtml([ca.Paragraph(["Result matrix:"]), table]))
    return body, answers

  @classmethod
  def _build_explanation(cls, context) -> tuple[ca.Section, list[ca.Answer]]:
    explanation = ca.Section()
    explanation.add_element(ca.Paragraph([
      f"The {context['cols_a']} columns of the first matrix match the "
      f"{context['rows_b']} rows of the second matrix. Each result entry is "
      "the dot product of a row of the first matrix and a column of the second."
    ]))
    for row_index in range(context["rows_a"]):
      for col_index in range(context["cols_b"]):
        terms = " + ".join(
          f"{context['matrix_a'][row_index][k]} \\cdot "
          f"{context['matrix_b'][k][col_index]}"
          for k in range(context["cols_a"])
        )
        explanation.add_element(ca.Equation(
          f"c_{{{row_index + 1},{col_index + 1}}} = {terms} "
          f"= {context['result'][row_index][col_index]}",
          inline=False,
        ))
    explanation.add_element(ca.Paragraph(["Final result:"]))
    explanation.add_element(ca.Matrix(data=context["result"], bracket_type="b"))
    return explanation, []


@QuestionRegistry.register()
class MatrixMultiplicationCompatibility(MatrixMathQuestion):
  """Practice identifying an incompatible product before doing arithmetic."""

  MIN_SIZE = 2
  MAX_SIZE = 4

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context, min_size, max_size = cls._build_matrix_context(
      rng_seed=rng_seed,
      **kwargs,
    )
    rows_a = kwargs.get("rows_a", context.rng.randint(min_size, max_size))
    cols_a = kwargs.get("cols_a", context.rng.randint(min_size, max_size))
    rows_b = kwargs.get("rows_b", context.rng.randint(min_size, max_size))
    if "rows_b" in kwargs and rows_b == cols_a:
      raise ValueError(
        "MatrixMultiplicationCompatibility requires cols_a to differ from rows_b"
      )
    while rows_b == cols_a:
      rows_b = context.rng.randint(min_size, max_size)
    cols_b = kwargs.get("cols_b", context.rng.randint(min_size, max_size))
    if not all(min_size <= size <= max_size for size in (rows_a, cols_a, rows_b, cols_b)):
      raise ValueError("matrix dimensions must be within min_size and max_size")
    min_value = kwargs.get("min_value", 1)
    max_value = kwargs.get("max_value", 9)
    context["rows_a"] = rows_a
    context["cols_a"] = cols_a
    context["rows_b"] = rows_b
    context["cols_b"] = cols_b
    context["matrix_a"] = cls._generate_matrix(context.rng, rows_a, cols_a, min_value, max_value)
    context["matrix_b"] = cls._generate_matrix(context.rng, rows_b, cols_b, min_value, max_value)
    return context

  @classmethod
  def _build_body(cls, context):
    body = ca.Section()
    body.add_element(ca.Paragraph([
      "Decide whether this product is defined. Enter Yes or No, then explain "
      "your decision."
    ]))
    body.add_element(ca.MathExpression([
      ca.Matrix(data=context["matrix_a"], bracket_type="b"),
      r" \cdot ",
      ca.Matrix(data=context["matrix_b"], bracket_type="b"),
    ]))
    answer = ca.AnswerTypes.String("No", label="Is the product defined?")
    body.add_element(ca.AnswerBlock([answer]))
    return body, [answer]

  @classmethod
  def _build_explanation(cls, context) -> tuple[ca.Section, list[ca.Answer]]:
    explanation = ca.Section()
    explanation.add_element(ca.Paragraph([
      f"No. The first matrix has {context['cols_a']} columns, while the "
      f"second matrix has {context['rows_b']} rows. Matrix multiplication "
      "requires these inner dimensions to be equal."
    ]))
    return explanation, []
