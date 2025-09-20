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

        body.add_element(
            ContentAST.Paragraph([
                "Calculate the sum of the following two matrices. Fill in each cell of the result matrix."
            ])
        )

        # Display Matrix A
        body.add_element(ContentAST.Paragraph(["Matrix A:"]))
        body.add_element(ContentAST.Table(data=self._matrix_to_table(self.matrix_a), padding=True))

        # Display Matrix B
        body.add_element(ContentAST.Paragraph(["Matrix B:"]))
        body.add_element(ContentAST.Table(data=self._matrix_to_table(self.matrix_b), padding=True))

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

        explanation.add_element(ContentAST.Paragraph(["Result:"]))
        explanation.add_element(ContentAST.Table(data=self._matrix_to_table(self.result), padding=True))

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

        body.add_element(
            ContentAST.Paragraph([
                f"Calculate {self.scalar} times the following matrix. Fill in each cell of the result matrix."
            ])
        )

        # Display original matrix
        body.add_element(ContentAST.Paragraph(["Matrix:"]))
        body.add_element(ContentAST.Table(data=self._matrix_to_table(self.matrix), padding=True))

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

        explanation.add_element(ContentAST.Paragraph(["Result:"]))
        explanation.add_element(ContentAST.Table(data=self._matrix_to_table(self.result), padding=True))

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

        # Dimension answers
        if self.multiplication_possible:
            self.answers["result_rows"] = Answer.integer("result_rows", self.result_rows)
            self.answers["result_cols"] = Answer.integer("result_cols", self.result_cols)
        else:
            self.answers["multiplication_possible"] = Answer.multiple_choice(
                "multiplication_possible",
                "No, multiplication is not possible",
                ["Yes, multiplication is possible", "No, multiplication is not possible"]
            )

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

        body.add_element(
            ContentAST.Paragraph([
                f"Calculate the product of Matrix A × Matrix B. First determine if multiplication is possible, "
                f"then fill in the result matrix. Use '-' for cells that don't exist in the result."
            ])
        )

        # Display Matrix A
        body.add_element(ContentAST.Paragraph([f"Matrix A ({self.rows_a}×{self.cols_a}):"]))
        body.add_element(ContentAST.Table(data=self._matrix_to_table(self.matrix_a), padding=True))

        # Display Matrix B
        body.add_element(ContentAST.Paragraph([f"Matrix B ({self.rows_b}×{self.cols_b}):"]))
        body.add_element(ContentAST.Table(data=self._matrix_to_table(self.matrix_b), padding=True))

        # Questions about possibility and dimensions
        if self.multiplication_possible:
            body.add_element(
                ContentAST.AnswerBlock([
                    ContentAST.Answer(
                        answer=self.answers["result_rows"],
                        label="Number of rows in result"
                    ),
                    ContentAST.Answer(
                        answer=self.answers["result_cols"],
                        label="Number of columns in result"
                    )
                ])
            )
        else:
            body.add_element(
                ContentAST.AnswerBlock([
                    ContentAST.Answer(
                        answer=self.answers["multiplication_possible"],
                        label="Is matrix multiplication possible?"
                    )
                ])
            )

        # Answer table (always max dimensions)
        body.add_element(ContentAST.Paragraph([f"Result matrix (A × B) - {self.max_dim}×{self.max_dim} grid:"]))
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

            explanation.add_element(ContentAST.Paragraph(["Result:"]))
            explanation.add_element(ContentAST.Table(data=self._matrix_to_table(self.result), padding=True))
        else:
            explanation.add_element(
                ContentAST.Paragraph([
                    f"Matrix multiplication is not possible because the number of columns in Matrix A ({self.cols_a}) "
                    f"does not equal the number of rows in Matrix B ({self.rows_b})."
                ])
            )

        return explanation