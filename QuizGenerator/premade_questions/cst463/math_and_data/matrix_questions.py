#!env python
import abc
import logging

from QuizGenerator.question import Question, QuestionRegistry, Answer
from QuizGenerator.misc import ContentAST

log = logging.getLogger(__name__)


class MatrixMathQuestion(Question, abc.ABC):
    def __init__(self, *args, **kwargs):
        kwargs["topic"] = kwargs.get("topic", Question.Topic.MATH)
        super().__init__(*args, **kwargs)

    def _generate_matrix(self, rows, cols, min_val=1, max_val=9):
        """Generate a matrix with random integer values."""
        return [[self.rng.randint(min_val, max_val) for _ in range(cols)] for _ in range(rows)]

    def _matrix_to_table(self, matrix, prefix=""):
        """Convert a matrix to ContentAST table format."""
        return [[f"{prefix}{matrix[i][j]}" for j in range(len(matrix[0]))] for i in range(len(matrix))]

    def _create_answer_table(self, rows, cols, answers_dict, answer_prefix="answer"):
        """Create a table with answer blanks for matrix results."""
        table_data = []
        for i in range(rows):
            row = []
            for j in range(cols):
                answer_key = f"{answer_prefix}_{i}_{j}"
                row.append(ContentAST.Answer(answer=answers_dict[answer_key]))
            table_data.append(row)
        return ContentAST.Table(data=table_data, padding=True)


@QuestionRegistry.register()
class MatrixAddition(MatrixMathQuestion):

    MIN_SIZE = 2
    MAX_SIZE = 4

    def refresh(self, *args, **kwargs):
        super().refresh(*args, **kwargs)

        # Generate matrix dimensions (same for both matrices in addition)
        self.rows = self.rng.randint(self.MIN_SIZE, self.MAX_SIZE)
        self.cols = self.rng.randint(self.MIN_SIZE, self.MAX_SIZE)

        # Generate two matrices
        self.matrix_a = self._generate_matrix(self.rows, self.cols)
        self.matrix_b = self._generate_matrix(self.rows, self.cols)

        # Calculate result matrix
        self.result = [[self.matrix_a[i][j] + self.matrix_b[i][j]
                       for j in range(self.cols)] for i in range(self.rows)]

        # Create answers dictionary
        self.answers = {}
        for i in range(self.rows):
            for j in range(self.cols):
                answer_key = f"answer_{i}_{j}"
                self.answers[answer_key] = Answer.integer(answer_key, self.result[i][j])

    def get_body(self, **kwargs) -> ContentAST.Section:
        body = ContentAST.Section()

        # Concise question with matrices in single equation
        body.add_element(ContentAST.Paragraph(["Calculate:"]))

        # Create single equation with both matrices
        matrix_a_latex = ContentAST.Matrix.to_latex(self.matrix_a, "b")
        matrix_b_latex = ContentAST.Matrix.to_latex(self.matrix_b, "b")
        body.add_element(ContentAST.Equation(f"{matrix_a_latex} + {matrix_b_latex} = "))

        # Answer table
        body.add_element(ContentAST.Paragraph(["Result (A + B):"]))
        body.add_element(self._create_answer_table(self.rows, self.cols, self.answers))

        return body

    def get_explanation(self, **kwargs) -> ContentAST.Section:
        explanation = ContentAST.Section()

        explanation.add_element(
            ContentAST.Paragraph([
                "Matrix addition is performed element-wise. Each element in the result matrix "
                "is the sum of the corresponding elements in the input matrices."
            ])
        )

        # Comprehensive step-by-step walkthrough
        explanation.add_element(ContentAST.Paragraph(["Step-by-step calculation:"]))

        # Show matrix addition with symbolic representation
        # Create properly formatted matrix strings
        matrix_a_str = r" \\ ".join([" & ".join([str(self.matrix_a[i][j]) for j in range(self.cols)]) for i in range(self.rows)])
        matrix_b_str = r" \\ ".join([" & ".join([str(self.matrix_b[i][j]) for j in range(self.cols)]) for i in range(self.rows)])
        addition_str = r" \\ ".join([" & ".join([f"{self.matrix_a[i][j]}+{self.matrix_b[i][j]}" for j in range(self.cols)]) for i in range(self.rows)])
        result_str = r" \\ ".join([" & ".join([str(self.result[i][j]) for j in range(self.cols)]) for i in range(self.rows)])

        explanation.add_element(
            ContentAST.Equation.make_block_equation__multiline_equals(
                lhs="A + B",
                rhs=[
                    f"\\begin{{bmatrix}} {matrix_a_str} \\end{{bmatrix}} + \\begin{{bmatrix}} {matrix_b_str} \\end{{bmatrix}}",
                    f"\\begin{{bmatrix}} {addition_str} \\end{{bmatrix}}",
                    f"\\begin{{bmatrix}} {result_str} \\end{{bmatrix}}"
                ]
            )
        )

        return explanation


@QuestionRegistry.register()
class MatrixScalarMultiplication(MatrixMathQuestion):

    MIN_SIZE = 2
    MAX_SIZE = 4
    MIN_SCALAR = 2
    MAX_SCALAR = 9

    def refresh(self, *args, **kwargs):
        super().refresh(*args, **kwargs)

        # Generate matrix dimensions
        self.rows = self.rng.randint(self.MIN_SIZE, self.MAX_SIZE)
        self.cols = self.rng.randint(self.MIN_SIZE, self.MAX_SIZE)

        # Generate scalar and matrix
        self.scalar = self.rng.randint(self.MIN_SCALAR, self.MAX_SCALAR)
        self.matrix = self._generate_matrix(self.rows, self.cols)

        # Calculate result matrix
        self.result = [[self.scalar * self.matrix[i][j]
                       for j in range(self.cols)] for i in range(self.rows)]

        # Create answers dictionary
        self.answers = {}
        for i in range(self.rows):
            for j in range(self.cols):
                answer_key = f"answer_{i}_{j}"
                self.answers[answer_key] = Answer.integer(answer_key, self.result[i][j])

    def get_body(self, **kwargs) -> ContentAST.Section:
        body = ContentAST.Section()

        # Concise question with scalar and matrix using Matrix AST
        body.add_element(ContentAST.Paragraph(["Calculate:"]))

        # Create single equation with scalar and matrix
        matrix_latex = ContentAST.Matrix.to_latex(self.matrix, "b")
        body.add_element(ContentAST.Equation(f"{self.scalar} \\cdot {matrix_latex} = "))

        # Answer table
        body.add_element(ContentAST.Paragraph([f"Result ({self.scalar} × Matrix):"]))
        body.add_element(self._create_answer_table(self.rows, self.cols, self.answers))

        return body

    def get_explanation(self, **kwargs) -> ContentAST.Section:
        explanation = ContentAST.Section()

        explanation.add_element(
            ContentAST.Paragraph([
                "Scalar multiplication involves multiplying every element in the matrix by the scalar value."
            ])
        )

        # Comprehensive step-by-step walkthrough
        explanation.add_element(ContentAST.Paragraph(["Step-by-step calculation:"]))

        # Show scalar multiplication with symbolic representation
        # Create properly formatted matrix strings
        matrix_str = r" \\ ".join([" & ".join([str(self.matrix[i][j]) for j in range(self.cols)]) for i in range(self.rows)])
        multiplication_str = r" \\ ".join([" & ".join([f"{self.scalar} \\cdot {self.matrix[i][j]}" for j in range(self.cols)]) for i in range(self.rows)])
        result_str = r" \\ ".join([" & ".join([str(self.result[i][j]) for j in range(self.cols)]) for i in range(self.rows)])

        explanation.add_element(
            ContentAST.Equation.make_block_equation__multiline_equals(
                lhs=f"{self.scalar} \\cdot A",
                rhs=[
                    f"{self.scalar} \\cdot \\begin{{bmatrix}} {matrix_str} \\end{{bmatrix}}",
                    f"\\begin{{bmatrix}} {multiplication_str} \\end{{bmatrix}}",
                    f"\\begin{{bmatrix}} {result_str} \\end{{bmatrix}}"
                ]
            )
        )

        return explanation


@QuestionRegistry.register()
class MatrixMultiplication(MatrixMathQuestion):

    MIN_SIZE = 2
    MAX_SIZE = 4

    def refresh(self, *args, **kwargs):
        super().refresh(*args, **kwargs)

        # Generate matrix dimensions
        self.rows_a = self.rng.randint(self.MIN_SIZE, self.MAX_SIZE)
        self.cols_a = self.rng.randint(self.MIN_SIZE, self.MAX_SIZE)
        self.rows_b = self.rng.randint(self.MIN_SIZE, self.MAX_SIZE)
        self.cols_b = self.rng.randint(self.MIN_SIZE, self.MAX_SIZE)

        # Determine if multiplication is possible
        self.multiplication_possible = (self.cols_a == self.rows_b)

        # Generate matrices
        self.matrix_a = self._generate_matrix(self.rows_a, self.cols_a)
        self.matrix_b = self._generate_matrix(self.rows_b, self.cols_b)

        # Calculate max dimensions for answer table
        self.max_dim = max(self.rows_a, self.cols_a, self.rows_b, self.cols_b)

        # Calculate result if possible
        if self.multiplication_possible:
            self.result_rows = self.rows_a
            self.result_cols = self.cols_b
            self.result = [[sum(self.matrix_a[i][k] * self.matrix_b[k][j]
                              for k in range(self.cols_a))
                           for j in range(self.cols_b)] for i in range(self.rows_a)]
        else:
            self.result_rows = 0
            self.result_cols = 0
            self.result = []

        # Create answers dictionary
        self.answers = {}

        # Dimension answers - always ask for dimensions
        if self.multiplication_possible:
            self.answers["result_rows"] = Answer.integer("result_rows", self.result_rows)
            self.answers["result_cols"] = Answer.integer("result_cols", self.result_cols)
        else:
            self.answers["result_rows"] = Answer.string("result_rows", "-")
            self.answers["result_cols"] = Answer.string("result_cols", "-")

        # Matrix element answers
        for i in range(self.max_dim):
            for j in range(self.max_dim):
                answer_key = f"answer_{i}_{j}"
                if (self.multiplication_possible and
                    i < self.result_rows and j < self.result_cols):
                    self.answers[answer_key] = Answer.integer(answer_key, self.result[i][j])
                else:
                    self.answers[answer_key] = Answer.string(answer_key, "-")

    def get_body(self, **kwargs) -> ContentAST.Section:
        body = ContentAST.Section()

        # Concise question with matrices in single equation
        body.add_element(ContentAST.Paragraph(["Calculate:"]))

        # Create single equation with both matrices
        matrix_a_latex = ContentAST.Matrix.to_latex(self.matrix_a, "b")
        matrix_b_latex = ContentAST.Matrix.to_latex(self.matrix_b, "b")
        body.add_element(ContentAST.Equation(f"{matrix_a_latex} \\times {matrix_b_latex} = "))
        body.add_element(
            ContentAST.Paragraph([
                "(Use '-' for cells that don't exist in the result if multiplication is not possible.)"
            ])
        )

        # Always ask for result dimensions
        body.add_element(
            ContentAST.AnswerBlock([
                ContentAST.Answer(
                    answer=self.answers["result_rows"],
                    label="Number of rows in result (use '-' if not possible)"
                ),
                ContentAST.Answer(
                    answer=self.answers["result_cols"],
                    label="Number of columns in result (use '-' if not possible)"
                )
            ])
        )

        # Answer table (always max dimensions) - removing confusing grid size mention
        body.add_element(ContentAST.Paragraph(["Result matrix (A × B):"]))
        body.add_element(self._create_answer_table(self.max_dim, self.max_dim, self.answers))

        return body

    def get_explanation(self, **kwargs) -> ContentAST.Section:
        explanation = ContentAST.Section()

        if self.multiplication_possible:
            explanation.add_element(
                ContentAST.Paragraph([
                    f"Matrix multiplication is possible because the number of columns in Matrix A ({self.cols_a}) "
                    f"equals the number of rows in Matrix B ({self.rows_b}). "
                    f"The result is a {self.result_rows}×{self.result_cols} matrix."
                ])
            )

            # Comprehensive matrix multiplication walkthrough
            explanation.add_element(ContentAST.Paragraph(["Step-by-step calculation:"]))

            # Show detailed multiplication process
            explanation.add_element(ContentAST.Paragraph(["Element-by-element calculation:"]))

            # Show calculation for first few elements
            for i in range(min(2, self.result_rows)):
                for j in range(min(2, self.result_cols)):
                    element_calc = " + ".join([f"{self.matrix_a[i][k]} \\times {self.matrix_b[k][j]}" for k in range(self.cols_a)])
                    explanation.add_element(
                        ContentAST.Equation(f"({i+1},{j+1}): {element_calc} = {self.result[i][j]}", inline=True)
                    )

            explanation.add_element(ContentAST.Paragraph(["Final result:"]))
            explanation.add_element(ContentAST.Matrix(data=self.result, bracket_type="b"))
        else:
            explanation.add_element(
                ContentAST.Paragraph([
                    f"Matrix multiplication is not possible because the number of columns in Matrix A ({self.cols_a}) "
                    f"does not equal the number of rows in Matrix B ({self.rows_b})."
                ])
            )

        return explanation