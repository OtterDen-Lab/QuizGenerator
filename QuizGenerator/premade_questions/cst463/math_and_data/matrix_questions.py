#!env python
import abc
import logging

from QuizGenerator.question import Question, QuestionRegistry
import QuizGenerator.contentast as ca
from QuizGenerator.mixins import MathOperationQuestion

log = logging.getLogger(__name__)


class MatrixMathQuestion(MathOperationQuestion, Question):
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

    def _generate_matrix(self, rows, cols, min_val=1, max_val=9):
        """Generate a matrix with random integer values."""
        return [[self.rng.randint(min_val, max_val) for _ in range(cols)] for _ in range(rows)]

    def _matrix_to_table(self, matrix, prefix=""):
        """Convert a matrix to content AST table format."""
        return [[f"{prefix}{matrix[i][j]}" for j in range(len(matrix[0]))] for i in range(len(matrix))]

    def _create_answer_table(self, rows, cols, answers_dict, answer_prefix="answer"):
        """Create a table with answer blanks for matrix results.

        Returns:
            Tuple of (table, answers_list)
        """
        table_data = []
        answers = []
        for i in range(rows):
            row = []
            for j in range(cols):
                answer_key = f"{answer_prefix}_{i}_{j}"
                ans = answers_dict[answer_key]
                row.append(ans)
                answers.append(ans)
            table_data.append(row)
        return ca.Table(data=table_data, padding=True), answers

    # Implement MathOperationQuestion abstract methods

    @abc.abstractmethod
    def generate_operands(self):
        """Generate matrices for the operation. Subclasses must implement."""
        pass

    def format_operand_latex(self, operand):
        """Format a matrix for LaTeX display."""
        return ca.Matrix.to_latex(operand, "b")

    def format_single_equation(self, operand_a, operand_b):
        """Format the equation for single questions."""
        operand_a_latex = self.format_operand_latex(operand_a)
        operand_b_latex = self.format_operand_latex(operand_b)
        return f"{operand_a_latex} {self.get_operator()} {operand_b_latex}"

    def _add_single_question_answers(self, body):
        """Add Canvas-only answer fields for single questions.

        Returns:
            List of Answer objects that were added to the body.
        """
        answers = []

        # For matrices, we typically show result dimensions and answer table
        if hasattr(self, 'result_rows') and hasattr(self, 'result_cols'):
            # Matrix multiplication case with dimension answers
            if hasattr(self, 'answers') and "result_rows" in self.answers:
                rows_ans = self.answers["result_rows"]
                cols_ans = self.answers["result_cols"]
                answers.extend([rows_ans, cols_ans])
                body.add_element(
                    ca.OnlyHtml([
                        ca.AnswerBlock([rows_ans, cols_ans])
                    ])
                )

        # Matrix result table
        if hasattr(self, 'result') and self.result:
            rows = len(self.result)
            cols = len(self.result[0])
            table, table_answers = self._create_answer_table(rows, cols, self.answers)
            answers.extend(table_answers)
            body.add_element(
                ca.OnlyHtml([
                    ca.Paragraph(["Result matrix:"]),
                    table
                ])
            )
        elif hasattr(self, 'max_dim'):
            # Matrix multiplication with max dimensions
            table, table_answers = self._create_answer_table(self.max_dim, self.max_dim, self.answers)
            answers.extend(table_answers)
            body.add_element(
                ca.OnlyHtml([
                    ca.Paragraph(["Result matrix (use '-' if cell doesn't exist):"]),
                    table
                ])
            )

        return answers

    # Abstract methods that subclasses must implement
    @abc.abstractmethod
    def get_operator(self):
        """Return the LaTeX operator for this operation."""
        pass

    @abc.abstractmethod
    def calculate_single_result(self, matrix_a, matrix_b):
        """Calculate the result for a single question with two matrices."""
        pass

    @abc.abstractmethod
    def create_subquestion_answers(self, subpart_index, result):
        """Create answer objects for a subquestion result."""
        pass


@QuestionRegistry.register()
class MatrixAddition(MatrixMathQuestion):

    MIN_SIZE = 2
    MAX_SIZE = 4

    def generate_operands(self):
        """Generate two matrices with the same dimensions for addition."""
        # Generate matrix dimensions (same for both matrices in addition)
        self.rows = self.rng.randint(self.MIN_SIZE, self.MAX_SIZE)
        self.cols = self.rng.randint(self.MIN_SIZE, self.MAX_SIZE)

        # Generate two matrices
        matrix_a = self._generate_matrix(self.rows, self.cols)
        matrix_b = self._generate_matrix(self.rows, self.cols)
        return matrix_a, matrix_b

    def get_operator(self):
        """Return the addition operator."""
        return "+"

    def calculate_single_result(self, matrix_a, matrix_b):
        """Calculate matrix addition result."""
        rows = len(matrix_a)
        cols = len(matrix_a[0])
        return [[matrix_a[i][j] + matrix_b[i][j] for j in range(cols)] for i in range(rows)]

    def create_subquestion_answers(self, subpart_index, result):
        """Create answer objects for matrix addition result."""
        if subpart_index == 0 and not self.is_multipart():
            # For single questions, use the old answer format
            rows = len(result)
            cols = len(result[0])
            for i in range(rows):
                for j in range(cols):
                    answer_key = f"answer_{i}_{j}"
                    self.answers[answer_key] = ca.AnswerTypes.Int(result[i][j])
        else:
            # For multipart questions, use subpart letter format
            letter = chr(ord('a') + subpart_index)
            rows = len(result)
            cols = len(result[0])
            for i in range(rows):
                for j in range(cols):
                    answer_key = f"subpart_{letter}_{i}_{j}"
                    self.answers[answer_key] = ca.AnswerTypes.Int(result[i][j])

    def refresh(self, *args, **kwargs):
        """Override refresh to set rows/cols for compatibility."""
        super().refresh(*args, **kwargs)

        # For backward compatibility, set matrix attributes for single questions
        if not self.is_multipart():
            self.matrix_a = self.operand_a
            self.matrix_b = self.operand_b
            # rows and cols should already be set by generate_operands

    def get_explanation(self, **kwargs) -> ca.Section:
        explanation = ca.Section()

        explanation.add_element(
            ca.Paragraph([
                "Matrix addition is performed element-wise. Each element in the result matrix "
                "is the sum of the corresponding elements in the input matrices."
            ])
        )

        if self.is_multipart():
            # Handle multipart explanations
            explanation.add_element(ca.Paragraph(["Step-by-step calculation for each part:"]))
            for i, data in enumerate(self.subquestion_data):
                letter = chr(ord('a') + i)
                matrix_a = data.get('matrix_a', data['operand_a'])
                matrix_b = data.get('matrix_b', data['operand_b'])
                result = data['result']

                # Create LaTeX strings for multiline equation
                rows = len(matrix_a)
                cols = len(matrix_a[0])
                matrix_a_str = r" \\ ".join([" & ".join([str(matrix_a[row][col]) for col in range(cols)]) for row in range(rows)])
                matrix_b_str = r" \\ ".join([" & ".join([str(matrix_b[row][col]) for col in range(cols)]) for row in range(rows)])
                addition_str = r" \\ ".join([" & ".join([f"{matrix_a[row][col]}+{matrix_b[row][col]}" for col in range(cols)]) for row in range(rows)])
                result_str = r" \\ ".join([" & ".join([str(result[row][col]) for col in range(cols)]) for row in range(rows)])

                # Add explanation for this subpart
                explanation.add_element(ca.Paragraph([f"Part ({letter}):"]))
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
        else:
            # Single part explanation (original behavior)
            explanation.add_element(ca.Paragraph(["Step-by-step calculation:"]))

            # Create properly formatted matrix strings
            matrix_a_str = r" \\ ".join([" & ".join([str(self.matrix_a[i][j]) for j in range(self.cols)]) for i in range(self.rows)])
            matrix_b_str = r" \\ ".join([" & ".join([str(self.matrix_b[i][j]) for j in range(self.cols)]) for i in range(self.rows)])
            addition_str = r" \\ ".join([" & ".join([f"{self.matrix_a[i][j]}+{self.matrix_b[i][j]}" for j in range(self.cols)]) for i in range(self.rows)])
            result_str = r" \\ ".join([" & ".join([str(self.result[i][j]) for j in range(self.cols)]) for i in range(self.rows)])

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

        return explanation


@QuestionRegistry.register()
class MatrixScalarMultiplication(MatrixMathQuestion):

    MIN_SIZE = 2
    MAX_SIZE = 4
    MIN_SCALAR = 2
    MAX_SCALAR = 9

    def _generate_scalar(self):
        """Generate a scalar for multiplication."""
        return self.rng.randint(self.MIN_SCALAR, self.MAX_SCALAR)

    def generate_operands(self):
        """Generate scalar and matrix for scalar multiplication."""
        # Generate matrix dimensions
        self.rows = self.rng.randint(self.MIN_SIZE, self.MAX_SIZE)
        self.cols = self.rng.randint(self.MIN_SIZE, self.MAX_SIZE)

        # Generate matrix (we'll generate scalar per subpart in refresh)
        matrix = self._generate_matrix(self.rows, self.cols)
        dummy_matrix = matrix  # Not used but needed for interface compatibility
        return matrix, dummy_matrix

    def get_operator(self):
        """Return scalar multiplication operator with current scalar."""
        if hasattr(self, 'scalar'):
            return f"{self.scalar} \\cdot"
        else:
            return "k \\cdot"  # Fallback for multipart case

    def calculate_single_result(self, matrix_a, matrix_b):
        """Calculate scalar multiplication result."""
        # For scalar multiplication, we only use matrix_a and need self.scalar
        rows = len(matrix_a)
        cols = len(matrix_a[0])
        return [[self.scalar * matrix_a[i][j] for j in range(cols)] for i in range(rows)]

    def create_subquestion_answers(self, subpart_index, result):
        """Create answer objects for matrix scalar multiplication result."""
        if subpart_index == 0 and not self.is_multipart():
            # For single questions, use the old answer format
            rows = len(result)
            cols = len(result[0])
            for i in range(rows):
                for j in range(cols):
                    answer_key = f"answer_{i}_{j}"
                    self.answers[answer_key] = ca.AnswerTypes.Int(result[i][j])
        else:
            # For multipart questions, use subpart letter format
            letter = chr(ord('a') + subpart_index)
            rows = len(result)
            cols = len(result[0])
            for i in range(rows):
                for j in range(cols):
                    answer_key = f"subpart_{letter}_{i}_{j}"
                    self.answers[answer_key] = ca.AnswerTypes.Int(result[i][j])

    def refresh(self, *args, **kwargs):
        """Override refresh to handle different scalars per subpart."""
        if self.is_multipart():
            # For multipart questions, handle everything ourselves like VectorScalarMultiplication
            Question.refresh(self, *args, **kwargs)

            # Generate matrix dimensions
            self.rows = self.rng.randint(self.MIN_SIZE, self.MAX_SIZE)
            self.cols = self.rng.randint(self.MIN_SIZE, self.MAX_SIZE)

            # Clear any existing data
            self.answers = {}

            # Generate multiple subquestions with different scalars
            self.subquestion_data = []
            for i in range(self.num_subquestions):
                # Generate matrix and scalar for each subquestion
                matrix = self._generate_matrix(self.rows, self.cols)
                scalar = self._generate_scalar()
                result = [[scalar * matrix[i][j] for j in range(self.cols)] for i in range(self.rows)]

                self.subquestion_data.append({
                    'operand_a': matrix,
                    'operand_b': matrix,  # Not used but kept for consistency
                    'matrix': matrix,     # For compatibility
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

            # For backward compatibility
            if hasattr(self, 'operand_a'):
                self.matrix = self.operand_a

    def generate_subquestion_data(self):
        """Override to handle scalar multiplication format."""
        subparts = []
        for data in self.subquestion_data:
            matrix_latex = ca.Matrix.to_latex(data['matrix'], "b")
            scalar = data['scalar']
            # Return scalar * matrix as a single string
            subparts.append(f"{scalar} \\cdot {matrix_latex}")
        return subparts

    def format_single_equation(self, operand_a, operand_b):
        """Format the equation for single questions."""
        matrix_latex = ca.Matrix.to_latex(operand_a, "b")
        return f"{self.scalar} \\cdot {matrix_latex}"

    def get_explanation(self, **kwargs) -> ca.Section:
        explanation = ca.Section()

        explanation.add_element(
            ca.Paragraph([
                "Scalar multiplication involves multiplying every element in the matrix by the scalar value."
            ])
        )

        if self.is_multipart():
            # Handle multipart explanations
            explanation.add_element(ca.Paragraph(["Step-by-step calculation for each part:"]))
            for i, data in enumerate(self.subquestion_data):
                letter = chr(ord('a') + i)
                matrix = data.get('matrix', data['operand_a'])
                scalar = data['scalar']
                result = data['result']

                # Create LaTeX strings for multiline equation
                rows = len(matrix)
                cols = len(matrix[0])
                matrix_str = r" \\ ".join([" & ".join([str(matrix[row][col]) for col in range(cols)]) for row in range(rows)])
                multiplication_str = r" \\ ".join([" & ".join([f"{scalar} \\cdot {matrix[row][col]}" for col in range(cols)]) for row in range(rows)])
                result_str = r" \\ ".join([" & ".join([str(result[row][col]) for col in range(cols)]) for row in range(rows)])

                # Add explanation for this subpart
                explanation.add_element(ca.Paragraph([f"Part ({letter}):"]))
                explanation.add_element(
                    ca.Equation.make_block_equation__multiline_equals(
                        lhs=f"{scalar} \\cdot A",
                        rhs=[
                            f"{scalar} \\cdot \\begin{{bmatrix}} {matrix_str} \\end{{bmatrix}}",
                            f"\\begin{{bmatrix}} {multiplication_str} \\end{{bmatrix}}",
                            f"\\begin{{bmatrix}} {result_str} \\end{{bmatrix}}"
                        ]
                    )
                )
        else:
            # Single part explanation
            explanation.add_element(ca.Paragraph(["Step-by-step calculation:"]))

            # Create properly formatted matrix strings
            matrix_str = r" \\ ".join([" & ".join([str(self.matrix[i][j]) for j in range(self.cols)]) for i in range(self.rows)])
            multiplication_str = r" \\ ".join([" & ".join([f"{self.scalar} \\cdot {self.matrix[i][j]}" for j in range(self.cols)]) for i in range(self.rows)])
            result_str = r" \\ ".join([" & ".join([str(self.result[i][j]) for j in range(self.cols)]) for i in range(self.rows)])

            explanation.add_element(
                ca.Equation.make_block_equation__multiline_equals(
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
    PROBABILITY_OF_VALID = 0.875  # 7/8 chance of success, 1/8 chance of failure

    def generate_operands(self):
        """Generate two matrices for multiplication."""
        # For multipart questions, always generate valid multiplications
        # For single questions, use probability to determine validity
        if self.is_multipart():
            should_be_valid = True  # Always valid for multipart
        else:
            should_be_valid = self.rng.choices([True, False], weights=[self.PROBABILITY_OF_VALID, 1-self.PROBABILITY_OF_VALID], k=1)[0]

        if should_be_valid:
            # Generate dimensions that allow multiplication
            self.rows_a = self.rng.randint(self.MIN_SIZE, self.MAX_SIZE)
            self.cols_a = self.rng.randint(self.MIN_SIZE, self.MAX_SIZE)
            self.rows_b = self.cols_a  # Ensure multiplication is possible
            self.cols_b = self.rng.randint(self.MIN_SIZE, self.MAX_SIZE)
        else:
            # Generate dimensions that don't allow multiplication
            self.rows_a = self.rng.randint(self.MIN_SIZE, self.MAX_SIZE)
            self.cols_a = self.rng.randint(self.MIN_SIZE, self.MAX_SIZE)
            self.rows_b = self.rng.randint(self.MIN_SIZE, self.MAX_SIZE)
            self.cols_b = self.rng.randint(self.MIN_SIZE, self.MAX_SIZE)
            # Ensure they don't match by chance
            while self.cols_a == self.rows_b:
                self.rows_b = self.rng.randint(self.MIN_SIZE, self.MAX_SIZE)

        # Store multiplication possibility
        self.multiplication_possible = (self.cols_a == self.rows_b)

        # Generate matrices
        matrix_a = self._generate_matrix(self.rows_a, self.cols_a)
        matrix_b = self._generate_matrix(self.rows_b, self.cols_b)

        # Calculate max dimensions for answer table
        self.max_dim = max(self.rows_a, self.cols_a, self.rows_b, self.cols_b)

        return matrix_a, matrix_b

    def get_operator(self):
        """Return the multiplication operator."""
        return "\\cdot"

    def calculate_single_result(self, matrix_a, matrix_b):
        """Calculate matrix multiplication result."""
        rows_a = len(matrix_a)
        cols_a = len(matrix_a[0])
        rows_b = len(matrix_b)
        cols_b = len(matrix_b[0])

        # Check if multiplication is possible
        if cols_a != rows_b:
            return None  # Multiplication not possible

        # Calculate result
        result = [[sum(matrix_a[i][k] * matrix_b[k][j] for k in range(cols_a))
                  for j in range(cols_b)] for i in range(rows_a)]

        # Store result dimensions
        self.result_rows = rows_a
        self.result_cols = cols_b

        return result

    def create_subquestion_answers(self, subpart_index, result):
        """Create answer objects for matrix multiplication result."""
        if subpart_index == 0 and not self.is_multipart():
            # For single questions, use the old answer format
            # Dimension answers
            if result is not None:
                self.answers["result_rows"] = ca.AnswerTypes.Int(self.result_rows, label="Number of rows in result")
                self.answers["result_cols"] = ca.AnswerTypes.Int(self.result_cols, label="Number of columns in result")

                # Matrix element answers
                for i in range(self.max_dim):
                    for j in range(self.max_dim):
                        answer_key = f"answer_{i}_{j}"
                        if i < self.result_rows and j < self.result_cols:
                            self.answers[answer_key] = ca.AnswerTypes.Int(result[i][j])
                        else:
                            self.answers[answer_key] = ca.AnswerTypes.String("-")
            else:
                # Multiplication not possible
                self.answers["result_rows"] = ca.AnswerTypes.String("-", label="Number of rows in result")
                self.answers["result_cols"] = ca.AnswerTypes.String("-", label="Number of columns in result")

                # All matrix elements are "-"
                for i in range(self.max_dim):
                    for j in range(self.max_dim):
                        answer_key = f"answer_{i}_{j}"
                        self.answers[answer_key] = ca.AnswerTypes.String("-")
        else:
            # For multipart questions, use subpart letter format
            letter = chr(ord('a') + subpart_index)

            # For multipart, result should always be valid
            if result is not None:
                rows = len(result)
                cols = len(result[0])
                for i in range(rows):
                    for j in range(cols):
                        answer_key = f"subpart_{letter}_{i}_{j}"
                        self.answers[answer_key] = ca.AnswerTypes.Int(result[i][j])

    def _add_single_question_answers(self, body):
        """Add Canvas-only answer fields for MatrixMultiplication with dash instruction.

        Returns:
            List of Answer objects that were added to the body.
        """
        answers = []

        # Dimension answers for matrix multiplication
        if hasattr(self, 'answers') and "result_rows" in self.answers:
            rows_ans = self.answers["result_rows"]
            cols_ans = self.answers["result_cols"]
            answers.extend([rows_ans, cols_ans])
            body.add_element(
                ca.OnlyHtml([
                    ca.AnswerBlock([rows_ans, cols_ans])
                ])
            )

        # Matrix result table with dash instruction
        table, table_answers = self._create_answer_table(self.max_dim, self.max_dim, self.answers)
        answers.extend(table_answers)
        body.add_element(
            ca.OnlyHtml([
                ca.Paragraph(["Result matrix (use '-' if cell doesn't exist):"]),
                table
            ])
        )

        return answers

    def refresh(self, *args, **kwargs):
        """Override refresh to handle matrix attributes."""
        super().refresh(*args, **kwargs)

        # For backward compatibility, set matrix attributes for single questions
        if not self.is_multipart():
            self.matrix_a = self.operand_a
            self.matrix_b = self.operand_b

    def get_explanation(self, **kwargs) -> ca.Section:
        explanation = ca.Section()

        if self.is_multipart():
            # For multipart questions, provide simpler explanations
            explanation.add_element(
                ca.Paragraph([
                    "Matrix multiplication: Each element in the result is the dot product of "
                    "the corresponding row from the first matrix and column from the second matrix."
                ])
            )

            for i, data in enumerate(self.subquestion_data):
                letter = chr(ord('a') + i)
                matrix_a = data.get('matrix_a', data['operand_a'])
                matrix_b = data.get('matrix_b', data['operand_b'])
                result = data['result']

                explanation.add_element(ca.Paragraph([f"Part ({letter}): Matrices multiplied successfully."]))

        elif hasattr(self, 'multiplication_possible') and self.multiplication_possible:
            # Single question with successful multiplication
            explanation.add_element(ca.Paragraph(["Given matrices:"]))
            matrix_a_latex = ca.Matrix.to_latex(self.matrix_a, "b")
            matrix_b_latex = ca.Matrix.to_latex(self.matrix_b, "b")
            explanation.add_element(ca.Equation(f"A = {matrix_a_latex}, \\quad B = {matrix_b_latex}"))

            explanation.add_element(
                ca.Paragraph([
                    f"Matrix multiplication is possible because the number of columns in Matrix A ({self.cols_a}) "
                    f"equals the number of rows in Matrix B ({self.rows_b}). "
                    f"The result is a {self.result_rows}×{self.result_cols} matrix."
                ])
            )

            # Comprehensive matrix multiplication walkthrough
            explanation.add_element(ca.Paragraph(["Step-by-step calculation:"]))

            # Show detailed multiplication process using row×column visualization
            explanation.add_element(ca.Paragraph(["Each element is calculated as the dot product of a row from Matrix A and a column from Matrix B:"]))

            # Show calculation for first few elements with row×column visualization
            for i in range(min(2, self.result_rows)):
                for j in range(min(2, self.result_cols)):
                    # Get the row from matrix A and column from matrix B
                    row_a = [str(self.matrix_a[i][k]) for k in range(self.cols_a)]
                    col_b = [str(self.matrix_b[k][j]) for k in range(self.cols_a)]

                    # Create row and column vectors in LaTeX
                    row_latex = f"\\begin{{bmatrix}} {' & '.join(row_a)} \\end{{bmatrix}}"
                    col_latex = f"\\begin{{bmatrix}} {' \\\\\\\\ '.join(col_b)} \\end{{bmatrix}}"

                    # Show the calculation
                    element_calc = " + ".join([f"{self.matrix_a[i][k]} \\cdot {self.matrix_b[k][j]}" for k in range(self.cols_a)])

                    explanation.add_element(
                        ca.Equation(f"({i+1},{j+1}): {row_latex} \\cdot {col_latex} = {element_calc} = {self.result[i][j]}")
                    )

            explanation.add_element(ca.Paragraph(["Final result:"]))
            explanation.add_element(ca.Matrix(data=self.result, bracket_type="b"))
        else:
            # Single question with failed multiplication
            explanation.add_element(
                ca.Paragraph([
                    f"Matrix multiplication is not possible because the number of columns in Matrix A ({getattr(self, 'cols_a', 'unknown')}) "
                    f"does not equal the number of rows in Matrix B ({getattr(self, 'rows_b', 'unknown')})."
                ])
            )

        return explanation
