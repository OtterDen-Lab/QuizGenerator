#!env python
"""
Mixin classes to reduce boilerplate in question generation.
These mixins provide reusable patterns for common question structures.
"""

import abc
from typing import Dict, List, Any, Union
import QuizGenerator.contentast as ca


class TableQuestionMixin:
  """
  Mixin providing common table generation patterns for questions.

  This mixin identifies and abstracts the most common table patterns used
  across question types, reducing repetitive ca.Table creation code.
  """
  
  def create_info_table(self, info_dict: Dict[str, Any], transpose: bool = False) -> ca.Table:
    """
    Creates a vertical info table (key-value pairs).

    Common pattern: Display parameters/givens in a clean table format.
    Used by: HardDriveAccessTime, BaseAndBounds, etc.

    Args:
        info_dict: Dictionary of {label: value} pairs
        transpose: Whether to transpose the table (default: False)

    Returns:
        ca.Table with the information formatted
    """
    # Don't convert content AST elements to strings - let them render properly
    table_data = []
    for key, value in info_dict.items():
      # Keep content AST elements as-is, convert others to strings
      if isinstance(value, ca.Element):
        table_data.append([key, value])
      else:
        table_data.append([key, str(value)])

    return ca.Table(
      data=table_data,
      transpose=transpose
    )
  
  def create_answer_table(
      self,
      headers: List[str],
      data_rows: List[Dict[str, Any]],
      answer_columns: List[str] = None
  ) -> ca.Table:
    """
    Creates a table where some cells are answer blanks.

    Common pattern: Mix of given data and answer blanks in a structured table.
    Used by: VirtualAddressParts, SchedulingQuestion, CachingQuestion, etc.

    Args:
        headers: Column headers for the table
        data_rows: List of dictionaries, each representing a row
        answer_columns: List of column names that should be treated as answers

    Returns:
        ca.Table with answers embedded in appropriate cells
    """
    answer_columns = answer_columns or []
    
    def format_cell(row_data: Dict, column: str) -> Union[str, ca.Answer]:
      """Format a cell based on whether it should be an answer or plain data"""
      value = row_data.get(column, "")

      # If this column should contain answers and the value is an Answer object
      # Answer extends ca.Leaf, so it can be used directly
      if column in answer_columns and isinstance(value, ca.Answer):
        return value
      # If this column should contain answers but we have the answer key
      elif column in answer_columns and isinstance(value, str) and hasattr(self, 'answers'):
        answer_obj = self.answers.get(value)
        if answer_obj:
          return answer_obj

      # Otherwise return as plain data
      return str(value)
    
    table_data = [
      [format_cell(row, header) for header in headers]
      for row in data_rows
    ]
    
    return ca.Table(
      headers=headers,
      data=table_data
    )
  
  def create_parameter_answer_table(
      self,
      parameter_info: Dict[str, Any],
      answer_label: str,
      answer_key: str,
      transpose: bool = True
  ) -> ca.Table:
    """
    Creates a table combining parameters with a single answer.

    Common pattern: Show parameters/context, then ask for one calculated result.
    Used by: BaseAndBounds, many memory questions, etc.

    Args:
        parameter_info: Dictionary of {parameter_name: value}
        answer_label: Label for the answer row
        answer_key: Key to look up the answer in self.answers
        transpose: Whether to show as vertical table (default: True)

    Returns:
        ca.Table with parameters and answer
    """
    # Build data with parameters plus answer row
    data = [[key, str(value)] for key, value in parameter_info.items()]
    
    # Add answer row - Answer extends ca.Leaf so it can be used directly
    if hasattr(self, 'answers') and answer_key in self.answers:
      data.append([answer_label, self.answers[answer_key]])
    else:
      data.append([answer_label, f"[{answer_key}]"])  # Fallback
    
    return ca.Table(
      data=data,
      transpose=transpose
    )
  
  def create_fill_in_table(
      self,
      headers: List[str],
      template_rows: List[Dict[str, Any]]
  ) -> ca.Table:
    """
    Creates a table where multiple cells are answer blanks to fill in.

    Common pattern: Show a partially completed table where students fill blanks.
    Used by: CachingQuestion, SchedulingQuestion, etc.

    Args:
        headers: Column headers
        template_rows: Rows where values can be data or answer keys

    Returns:
        ca.Table with multiple answer blanks
    """
    
    def process_cell_value(value: Any) -> Union[str, ca.Answer]:
      """Convert cell values to appropriate display format"""
      # If it's already an Answer object, use it directly
      # Answer extends ca.Leaf so it can be used in the AST
      if isinstance(value, ca.Answer):
        return value
      # If it's a string that looks like an answer key, try to resolve it
      elif isinstance(value, str) and value.startswith("answer__") and hasattr(self, 'answers'):
        answer_obj = self.answers.get(value)
        if answer_obj:
          return answer_obj
      # Otherwise return as-is
      return str(value)
    
    table_data = [
      [process_cell_value(row.get(header, "")) for header in headers]
      for row in template_rows
    ]
    
    return ca.Table(
      headers=headers,
      data=table_data
    )


class BodyTemplatesMixin:
  """
  Mixin providing common body structure patterns.

  These methods create complete ca.Section objects following
  common question layout patterns.
  """
  
  def create_calculation_with_info_body(
      self,
      intro_text: str,
      info_table: ca.Table,
      answer_block: ca.AnswerBlock
  ) -> ca.Section:
    """
    Standard pattern: intro text + info table + answer block.

    Used by: HardDriveAccessTime, AverageMemoryAccessTime, etc.
    """
    body = ca.Section()
    body.add_element(ca.Paragraph([intro_text]))
    body.add_element(info_table)
    body.add_element(answer_block)
    return body
  
  def create_fill_in_table_body(
      self,
      intro_text: str,
      instructions: str,
      table: ca.Table
  ) -> ca.Section:
    """
    Standard pattern: intro + instructions + table with blanks.

    Used by: VirtualAddressParts, CachingQuestion, etc.
    """
    body = ca.Section()
    if intro_text:
      body.add_element(ca.Paragraph([intro_text]))
    if instructions:
      body.add_element(ca.Paragraph([instructions]))
    body.add_element(table)
    return body
  
  def create_parameter_calculation_body(
      self,
      intro_text: str,
      parameter_table: ca.Table,
      answer_table: ca.Table = None,
      additional_instructions: str = None
  ) -> ca.Section:
    """
    Standard pattern: intro + parameter table + optional answer table.

    Used by: BaseAndBounds, Paging, etc.
    """
    body = ca.Section()
    body.add_element(ca.Paragraph([intro_text]))
    body.add_element(parameter_table)
    
    if additional_instructions:
      body.add_element(ca.Paragraph([additional_instructions]))
    
    if answer_table:
      body.add_element(answer_table)
    
    return body


class MultiPartQuestionMixin:
  """
  Mixin providing multi-part question generation with labeled subparts (a), (b), (c), etc.

  This mixin enables questions to be split into multiple subparts when num_subquestions > 1.
  Each subpart gets its own calculation with proper (a), (b), (c) labeling and alignment.
  Primarily designed for vector math questions but extensible to other question types.

  Usage:
      class VectorDotProduct(VectorMathQuestion, MultiPartQuestionMixin):
          def get_body(self):
              if self.is_multipart():
                  return self.create_multipart_body()
              else:
                  return self.create_single_part_body()

  Methods provided:
      - is_multipart(): Check if this question should generate multiple subparts
      - create_repeated_problem_part(): Create the ca.RepeatedProblemPart element
      - generate_subquestion_data(): Abstract method for subclasses to implement
  """
  
  def is_multipart(self):
    """
    Check if this question should generate multiple subparts.

    Returns:
        bool: True if num_subquestions > 1, False otherwise
    """
    return getattr(self, 'num_subquestions', 1) > 1
  
  def create_repeated_problem_part(self, subpart_data_list):
    """
    Create a ca.RepeatedProblemPart element from subpart data.

    Args:
        subpart_data_list: List of data for each subpart. Each item can be:
            - A string (LaTeX equation content)
            - A ca.Element
            - A tuple/list of elements to be joined

    Returns:
        ca.RepeatedProblemPart: The formatted multi-part element

    Example:
        # For vector dot products
        subparts = [
            (matrix_a1, "\\cdot", matrix_b1),
            (matrix_a2, "\\cdot", matrix_b2)
        ]
        return self.create_repeated_problem_part(subparts)
    """
    import QuizGenerator.contentast as ca
    return ca.RepeatedProblemPart(subpart_data_list)
  
  def generate_subquestion_data(self):
    """
    Generate data for each subpart of the question.

    This is an abstract method that subclasses must implement.
    It should generate and return the data needed for each subpart.

    Returns:
        list: List of data for each subpart. The format depends on the
              specific question type but should be compatible with
              ca.RepeatedProblemPart.

    Example implementation:
        def generate_subquestion_data(self):
            subparts = []
            for i in range(self.num_subquestions):
                vector_a = self._generate_vector(self.dimension)
                vector_b = self._generate_vector(self.dimension)
                matrix_a = ca.Matrix.to_latex(
                    [[v] for v in vector_a], "b"
                )
                matrix_b = ca.Matrix.to_latex(
                    [[v] for v in vector_b], "b"
                )
                subparts.append((matrix_a, "\\cdot", matrix_b))
            return subparts
    """
    raise NotImplementedError(
      "Subclasses using MultiPartQuestionMixin must implement generate_subquestion_data()"
    )
  
  def create_multipart_body(self, intro_text="Calculate the following:"):
    """
    Create a standard multipart question body using the repeated problem part format.

    Args:
        intro_text: Introduction text for the question

    Returns:
        ca.Section: Complete question body with intro and subparts

    Example:
        def get_body(self):
            if self.is_multipart():
                return self.create_multipart_body("Calculate the dot products:")
            else:
                return self.create_single_part_body()
    """
    import QuizGenerator.contentast as ca
    body = ca.Section()
    body.add_element(ca.Paragraph([intro_text]))
    
    # Generate subpart data and create the repeated problem part
    subpart_data = self.generate_subquestion_data()
    repeated_part = self.create_repeated_problem_part(subpart_data)
    body.add_element(repeated_part)
    
    return body
  
  def get_subpart_answers(self):
    """
    Retrieve answers organized by subpart for multipart questions.

    Returns:
        dict: Dictionary mapping subpart letters ('a', 'b', 'c') to their answers.
              Returns empty dict if not a multipart question.

    Example:
        # For a 3-part question
        {
            'a': ca.Answer.integer('a', 5),
            'b': ca.Answer.integer('b', 12),
            'c': ca.Answer.integer('c', -3)
        }
    """
    if not self.is_multipart():
      return {}
    
    subpart_answers = {}
    for i in range(self.num_subquestions):
      letter = chr(ord('a') + i)
      # Look for answers with subpart keys
      answer_key = f"subpart_{letter}"
      if hasattr(self, 'answers') and answer_key in self.answers:
        subpart_answers[letter] = self.answers[answer_key]
    
    return subpart_answers


class MathOperationQuestion(MultiPartQuestionMixin, abc.ABC):
  """
  Abstract base class for mathematical operation questions (vectors, matrices, etc.).

  This class provides common infrastructure for questions that:
  - Perform operations on mathematical objects (vectors, matrices)
  - Support both single and multipart questions
  - Use LaTeX formatting for equations
  - Generate step-by-step explanations

  Subclasses must implement abstract methods for:
  - Generating operands (vectors, matrices, etc.)
  - Performing the mathematical operation
  - Formatting results for LaTeX display
  - Creating answer objects
  """
  
  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", "MATH")  # Default to MATH topic
    super().__init__(*args, **kwargs)
  
  # Abstract methods that subclasses must implement
  
  @abc.abstractmethod
  def get_operator(self):
    """Return the LaTeX operator for this operation (e.g., '+', '\\cdot', '\\times')."""
    pass
  
  @abc.abstractmethod
  def calculate_single_result(self, operand_a, operand_b):
    """Calculate the result for a single question with two operands."""
    pass
  
  @abc.abstractmethod
  def create_subquestion_answers(self, subpart_index, result):
    """Create answer objects for a subquestion result."""
    pass
  
  @abc.abstractmethod
  def generate_operands(self):
    """Generate two operands for the operation. Returns (operand_a, operand_b)."""
    pass
  
  @abc.abstractmethod
  def format_operand_latex(self, operand):
    """Format an operand for LaTeX display."""
    pass
  
  @abc.abstractmethod
  def format_single_equation(self, operand_a, operand_b):
    """Format the equation for single questions. Returns LaTeX string."""
    pass
  
  # Common implementation methods
  
  def get_intro_text(self):
    """Default intro text - subclasses can override."""
    return "Calculate the following:"
  
  def create_single_answers(self, result):
    """Create answers for single questions - just delegate to subquestion method."""
    return self.create_subquestion_answers(0, result)
  
  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)
    
    # Clear any existing data
    self.answers = {}
    
    if self.is_multipart():
      # Generate multiple subquestions
      self.subquestion_data = []
      for i in range(self.num_subquestions):
        # Generate unique operands for each subquestion
        operand_a, operand_b = self.generate_operands()
        result = self.calculate_single_result(operand_a, operand_b)
        
        self.subquestion_data.append(
          {
            'operand_a': operand_a,
            'operand_b': operand_b,
            'vector_a': operand_a,  # For vector compatibility
            'vector_b': operand_b,  # For vector compatibility
            'result': result
          }
        )
        
        # Create answers for this subpart
        self.create_subquestion_answers(i, result)
    else:
      # Single question (original behavior)
      self.operand_a, self.operand_b = self.generate_operands()
      self.result = self.calculate_single_result(self.operand_a, self.operand_b)
      
      # Create answers
      self.create_single_answers(self.result)
  
  def generate_subquestion_data(self):
    """Generate LaTeX content for each subpart of the question."""
    subparts = []
    for data in self.subquestion_data:
      operand_a_latex = self.format_operand_latex(data['operand_a'])
      operand_b_latex = self.format_operand_latex(data['operand_b'])
      # Return as tuple of (operand_a, operator, operand_b)
      subparts.append((operand_a_latex, self.get_operator(), operand_b_latex))
    return subparts
  
  def _get_body(self):
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    body.add_element(ca.Paragraph([self.get_intro_text()]))

    if self.is_multipart():
      # Use multipart formatting with repeated problem parts
      subpart_data = self.generate_subquestion_data()
      repeated_part = self.create_repeated_problem_part(subpart_data)
      body.add_element(repeated_part)
      # Collect answers from self.answers dict
      answers = list(self.answers.values())
    else:
      # Single equation display
      equation_latex = self.format_single_equation(self.operand_a, self.operand_b)
      body.add_element(ca.Equation(f"{equation_latex} = ", inline=False))

      # Canvas-only answer fields (hidden from PDF)
      single_answers = self._add_single_question_answers(body)
      if single_answers:
        answers.extend(single_answers)

    return body, answers

  def get_body(self):
    """Build question body (backward compatible interface)."""
    body, _ = self._get_body()
    return body

  def _add_single_question_answers(self, body):
    """Add Canvas-only answer fields for single questions. Subclasses can override.

    Returns:
        List of Answer objects that were added to the body.
    """
    # Default implementation - subclasses should override for specific answer formats
    return []

  def _get_explanation(self):
    """Default explanation structure. Subclasses should override for specific explanations."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph([self.get_explanation_intro()]))

    if self.is_multipart():
      # Handle multipart explanations
      for i, data in enumerate(self.subquestion_data):
        letter = chr(ord('a') + i)
        explanation.add_element(self.create_explanation_for_subpart(data, letter))
    else:
      # Single part explanation
      explanation.add_element(self.create_single_explanation())

    return explanation, []

  def get_explanation(self):
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation()
    return explanation
  
  def get_explanation_intro(self):
    """Get the intro text for explanations. Subclasses should override."""
    return "The calculation is performed as follows:"
  
  def create_explanation_for_subpart(self, subpart_data, letter):
    """Create explanation for a single subpart. Subclasses should override."""
    return ca.Paragraph([f"Part ({letter}): Calculation details would go here."])
  
  def create_single_explanation(self):
    """Create explanation for single questions. Subclasses should override."""
    return ca.Paragraph(["Single question explanation would go here."])
