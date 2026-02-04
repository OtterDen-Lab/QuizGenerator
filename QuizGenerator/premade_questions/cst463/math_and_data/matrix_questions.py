#!env python
import logging
import random
from typing import List, Tuple

from QuizGenerator.question import Question, QuestionRegistry
import QuizGenerator.contentast as ca

log = logging.getLogger(__name__)


class MatrixMathQuestion(Question):
    """
    Base class for matrix mathematics questions with multipart support.

    NOTE: This class demonstrates proper content AST usage patterns.
    When implementing similar question types (vectors, equations, etc.),
    follow these patterns for consistent formatting across output formats.

    Key patterns demonstrated:
    - ca.Matrix for mathematical matrices
    - ca.Equation.make_block_equation__multiline_equals for step-by-step solutions
    - ca.OnlyHtml for Canvas-specific content
    - ca.Answer.integer for numerical answers
    """
    def __init__(self, *args, **kwargs):
        kwargs["topic"] = kwargs.get("topic", Question.Topic.MATH)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _generate_matrix(rng, rows, cols, min_val=1, max_val=9):
        """Generate a matrix with random integer values."""
        return [[rng.randint(min_val, max_val) for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def _matrix_to_table(matrix, prefix=""):
        """Convert a matrix to content AST table format."""
        return [[f"{prefix}{matrix[i][j]}" for j in range(len(matrix[0]))] for i in range(len(matrix))]

    @staticmethod
    def _create_answer_table(answer_matrix):
        """Create a table with answer blanks for matrix results.

        Returns:
            Tuple of (table, answers_list)
        """
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

    # Abstract methods retained for compatibility; subclasses handle build directly.


@QuestionRegistry.register()
class MatrixAddition(MatrixMathQuestion):

    MIN_SIZE = 2
    MAX_SIZE = 4

    @classmethod
    def _build_context(cls, *, rng_seed=None, **kwargs):
        rng = random.Random(rng_seed)
        num_subquestions = kwargs.get("num_subquestions", 1)
        if num_subquestions > 1:
            raise NotImplementedError("Multipart not supported")

        rows = rng.randint(cls.MIN_SIZE, cls.MAX_SIZE)
        cols = rng.randint(cls.MIN_SIZE, cls.MAX_SIZE)

        matrix_a = cls._generate_matrix(rng, rows, cols)
        matrix_b = cls._generate_matrix(rng, rows, cols)
        result = [[matrix_a[i][j] + matrix_b[i][j] for j in range(cols)] for i in range(rows)]

        return {
            "rows": rows,
            "cols": cols,
            "matrix_a": matrix_a,
            "matrix_b": matrix_b,
            "result": result,
            "num_subquestions": num_subquestions,
        }

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
    def _build_explanation(cls, context) -> Tuple[ca.Section, List[ca.Answer]]:
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
        rng = random.Random(rng_seed)
        num_subquestions = kwargs.get("num_subquestions", 1)
        if num_subquestions > 1:
            raise NotImplementedError("Multipart not supported")

        rows = rng.randint(cls.MIN_SIZE, cls.MAX_SIZE)
        cols = rng.randint(cls.MIN_SIZE, cls.MAX_SIZE)
        matrix = cls._generate_matrix(rng, rows, cols)
        scalar = cls._generate_scalar(rng, cls.MIN_SCALAR, cls.MAX_SCALAR)
        result = [[scalar * matrix[i][j] for j in range(cols)] for i in range(rows)]

        return {
            "rows": rows,
            "cols": cols,
            "matrix": matrix,
            "scalar": scalar,
            "result": result,
            "num_subquestions": num_subquestions,
        }

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
    def _build_explanation(cls, context) -> Tuple[ca.Section, List[ca.Answer]]:
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

    MIN_SIZE = 2
    MAX_SIZE = 4
    PROBABILITY_OF_VALID = 0.875  # 7/8 chance of success, 1/8 chance of failure

    @classmethod
    def _build_context(cls, *, rng_seed=None, **kwargs):
        rng = random.Random(rng_seed)
        num_subquestions = kwargs.get("num_subquestions", 1)
        if num_subquestions > 1:
            raise NotImplementedError("Multipart not supported")

        should_be_valid = rng.choices(
            [True, False],
            weights=[cls.PROBABILITY_OF_VALID, 1 - cls.PROBABILITY_OF_VALID],
            k=1,
        )[0]

        if should_be_valid:
            rows_a = rng.randint(cls.MIN_SIZE, cls.MAX_SIZE)
            cols_a = rng.randint(cls.MIN_SIZE, cls.MAX_SIZE)
            rows_b = cols_a
            cols_b = rng.randint(cls.MIN_SIZE, cls.MAX_SIZE)
        else:
            rows_a = rng.randint(cls.MIN_SIZE, cls.MAX_SIZE)
            cols_a = rng.randint(cls.MIN_SIZE, cls.MAX_SIZE)
            rows_b = rng.randint(cls.MIN_SIZE, cls.MAX_SIZE)
            cols_b = rng.randint(cls.MIN_SIZE, cls.MAX_SIZE)
            while cols_a == rows_b:
                rows_b = rng.randint(cls.MIN_SIZE, cls.MAX_SIZE)

        multiplication_possible = (cols_a == rows_b)

        matrix_a = cls._generate_matrix(rng, rows_a, cols_a)
        matrix_b = cls._generate_matrix(rng, rows_b, cols_b)
        max_dim = max(rows_a, cols_a, rows_b, cols_b)

        result = None
        result_rows = None
        result_cols = None
        if multiplication_possible:
            result = [[sum(matrix_a[i][k] * matrix_b[k][j] for k in range(cols_a))
                      for j in range(cols_b)] for i in range(rows_a)]
            result_rows = rows_a
            result_cols = cols_b

        return {
            "rows_a": rows_a,
            "cols_a": cols_a,
            "rows_b": rows_b,
            "cols_b": cols_b,
            "matrix_a": matrix_a,
            "matrix_b": matrix_b,
            "multiplication_possible": multiplication_possible,
            "result": result,
            "result_rows": result_rows,
            "result_cols": result_cols,
            "max_dim": max_dim,
            "num_subquestions": num_subquestions,
        }

    @classmethod
    def _build_body(cls, context):
        body = ca.Section()
        body.add_element(ca.Paragraph(["Calculate the following:"]))

        matrix_a_elem = ca.Matrix(data=context["matrix_a"], bracket_type="b")
        matrix_b_elem = ca.Matrix(data=context["matrix_b"], bracket_type="b")
        body.add_element(ca.MathExpression([matrix_a_elem, " \cdot ", matrix_b_elem, " = "]))

        if context["result"] is not None:
            rows_ans = ca.AnswerTypes.Int(context["result_rows"], label="Number of rows in result")
            cols_ans = ca.AnswerTypes.Int(context["result_cols"], label="Number of columns in result")
        else:
            rows_ans = ca.AnswerTypes.String("-", label="Number of rows in result")
            cols_ans = ca.AnswerTypes.String("-", label="Number of columns in result")

        answers = [rows_ans, cols_ans]
        body.add_element(
            ca.OnlyHtml([
                ca.AnswerBlock([rows_ans, cols_ans])
            ])
        )

        answer_matrix = []
        for i in range(context["max_dim"]):
            row = []
            for j in range(context["max_dim"]):
                if context["result"] is not None and i < context["result_rows"] and j < context["result_cols"]:
                    row.append(ca.AnswerTypes.Int(context["result"][i][j]))
                else:
                    row.append(ca.AnswerTypes.String("-"))
            answer_matrix.append(row)

        table, table_answers = cls._create_answer_table(answer_matrix)
        answers.extend(table_answers)
        body.add_element(
            ca.OnlyHtml([
                table
            ])
        )

        return body, answers

    @classmethod
    def _build_explanation(cls, context) -> Tuple[ca.Section, List[ca.Answer]]:
        explanation = ca.Section()

        if context["multiplication_possible"]:
            explanation.add_element(ca.Paragraph(["Given matrices:"]))
            matrix_a_latex = ca.Matrix.to_latex(context["matrix_a"], "b")
            matrix_b_latex = ca.Matrix.to_latex(context["matrix_b"], "b")
            explanation.add_element(ca.Equation(f"A = {matrix_a_latex}, \quad B = {matrix_b_latex}"))

            explanation.add_element(
                ca.Paragraph([
                    f"Matrix multiplication is possible because the number of columns in Matrix A ({context['cols_a']}) "
                    f"equals the number of rows in Matrix B ({context['rows_b']}). "
                    f"The result is a {context['result_rows']}×{context['result_cols']} matrix."
                ])
            )

            explanation.add_element(ca.Paragraph(["Step-by-step calculation:"]))
            explanation.add_element(ca.Paragraph([
                "Each element is calculated as the dot product of a row from Matrix A and a column from Matrix B:"
            ]))

            for i in range(min(2, context["result_rows"])):
                for j in range(min(2, context["result_cols"])):
                    row_a = [str(context["matrix_a"][i][k]) for k in range(context["cols_a"])]
                    col_b = [str(context["matrix_b"][k][j]) for k in range(context["cols_a"])]

                    row_latex = f"\begin{{bmatrix}} {' & '.join(row_a)} \end{{bmatrix}}"
                    col_latex = f"\begin{{bmatrix}} {' \\ '.join(col_b)} \end{{bmatrix}}"
                    element_calc = " + ".join([
                        f"{context['matrix_a'][i][k]} \cdot {context['matrix_b'][k][j]}"
                        for k in range(context["cols_a"])
                    ])

                    explanation.add_element(
                        ca.Equation(
                            f"({i+1},{j+1}): {row_latex} \cdot {col_latex} = {element_calc} = {context['result'][i][j]}"
                        )
                    )

            explanation.add_element(ca.Paragraph(["Final result:"]))
            explanation.add_element(ca.Matrix(data=context["result"], bracket_type="b"))
        else:
            explanation.add_element(
                ca.Paragraph([
                    f"Matrix multiplication is not possible because the number of columns in Matrix A ({context['cols_a']}) "
                    f"does not equal the number of rows in Matrix B ({context['rows_b']})."
                ])
            )

        return explanation, []
