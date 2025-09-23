#!env python
"""
Mixin classes to reduce boilerplate in question generation.
These mixins provide reusable patterns for common question structures.
"""

from typing import Dict, List, Any, Union
from QuizGenerator.misc import Answer
from QuizGenerator.contentast import ContentAST


class TableQuestionMixin:
    """
    Mixin providing common table generation patterns for questions.

    This mixin identifies and abstracts the most common table patterns used
    across question types, reducing repetitive ContentAST.Table creation code.
    """

    def create_info_table(self, info_dict: Dict[str, Any], transpose: bool = False) -> ContentAST.Table:
        """
        Creates a vertical info table (key-value pairs).

        Common pattern: Display parameters/givens in a clean table format.
        Used by: HardDriveAccessTime, BaseAndBounds, etc.

        Args:
            info_dict: Dictionary of {label: value} pairs
            transpose: Whether to transpose the table (default: False)

        Returns:
            ContentAST.Table with the information formatted
        """
        return ContentAST.Table(
            data=[[key, str(value)] for key, value in info_dict.items()],
            transpose=transpose
        )

    def create_answer_table(
        self,
        headers: List[str],
        data_rows: List[Dict[str, Any]],
        answer_columns: List[str] = None
    ) -> ContentAST.Table:
        """
        Creates a table where some cells are answer blanks.

        Common pattern: Mix of given data and answer blanks in a structured table.
        Used by: VirtualAddressParts, SchedulingQuestion, CachingQuestion, etc.

        Args:
            headers: Column headers for the table
            data_rows: List of dictionaries, each representing a row
            answer_columns: List of column names that should be treated as answers

        Returns:
            ContentAST.Table with answers embedded in appropriate cells
        """
        answer_columns = answer_columns or []

        def format_cell(row_data: Dict, column: str) -> Union[str, ContentAST.Answer]:
            """Format a cell based on whether it should be an answer or plain data"""
            value = row_data.get(column, "")

            # If this column should contain answers and the value is an Answer object
            if column in answer_columns and isinstance(value, Answer):
                return ContentAST.Answer(value)
            # If this column should contain answers but we have the answer key
            elif column in answer_columns and isinstance(value, str) and hasattr(self, 'answers'):
                answer_obj = self.answers.get(value)
                if answer_obj:
                    return ContentAST.Answer(answer_obj)

            # Otherwise return as plain data
            return str(value)

        table_data = [
            [format_cell(row, header) for header in headers]
            for row in data_rows
        ]

        return ContentAST.Table(
            headers=headers,
            data=table_data
        )

    def create_parameter_answer_table(
        self,
        parameter_info: Dict[str, Any],
        answer_label: str,
        answer_key: str,
        transpose: bool = True
    ) -> ContentAST.Table:
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
            ContentAST.Table with parameters and answer
        """
        # Build data with parameters plus answer row
        data = [[key, str(value)] for key, value in parameter_info.items()]

        # Add answer row
        if hasattr(self, 'answers') and answer_key in self.answers:
            data.append([answer_label, ContentAST.Answer(self.answers[answer_key])])
        else:
            data.append([answer_label, f"[{answer_key}]"])  # Fallback

        return ContentAST.Table(
            data=data,
            transpose=transpose
        )

    def create_fill_in_table(
        self,
        headers: List[str],
        template_rows: List[Dict[str, Any]]
    ) -> ContentAST.Table:
        """
        Creates a table where multiple cells are answer blanks to fill in.

        Common pattern: Show a partially completed table where students fill blanks.
        Used by: CachingQuestion, SchedulingQuestion, etc.

        Args:
            headers: Column headers
            template_rows: Rows where values can be data or answer keys

        Returns:
            ContentAST.Table with multiple answer blanks
        """
        def process_cell_value(value: Any) -> Union[str, ContentAST.Answer]:
            """Convert cell values to appropriate display format"""
            # If it's already an Answer object, wrap it
            if isinstance(value, Answer):
                return ContentAST.Answer(value)
            # If it's a string that looks like an answer key, try to resolve it
            elif isinstance(value, str) and value.startswith("answer__") and hasattr(self, 'answers'):
                answer_obj = self.answers.get(value)
                if answer_obj:
                    return ContentAST.Answer(answer_obj)
            # Otherwise return as-is
            return str(value)

        table_data = [
            [process_cell_value(row.get(header, "")) for header in headers]
            for row in template_rows
        ]

        return ContentAST.Table(
            headers=headers,
            data=table_data
        )


class BodyTemplatesMixin:
    """
    Mixin providing common body structure patterns.

    These methods create complete ContentAST.Section objects following
    common question layout patterns.
    """

    def create_calculation_with_info_body(
        self,
        intro_text: str,
        info_table: ContentAST.Table,
        answer_block: ContentAST.AnswerBlock
    ) -> ContentAST.Section:
        """
        Standard pattern: intro text + info table + answer block.

        Used by: HardDriveAccessTime, AverageMemoryAccessTime, etc.
        """
        body = ContentAST.Section()
        body.add_element(ContentAST.Paragraph([intro_text]))
        body.add_element(info_table)
        body.add_element(answer_block)
        return body

    def create_fill_in_table_body(
        self,
        intro_text: str,
        instructions: str,
        table: ContentAST.Table
    ) -> ContentAST.Section:
        """
        Standard pattern: intro + instructions + table with blanks.

        Used by: VirtualAddressParts, CachingQuestion, etc.
        """
        body = ContentAST.Section()
        body.add_element(ContentAST.Paragraph([intro_text]))
        if instructions:
            body.add_element(ContentAST.Paragraph([instructions]))
        body.add_element(table)
        return body

    def create_parameter_calculation_body(
        self,
        intro_text: str,
        parameter_table: ContentAST.Table,
        answer_table: ContentAST.Table = None,
        additional_instructions: str = None
    ) -> ContentAST.Section:
        """
        Standard pattern: intro + parameter table + optional answer table.

        Used by: BaseAndBounds, Paging, etc.
        """
        body = ContentAST.Section()
        body.add_element(ContentAST.Paragraph([intro_text]))
        body.add_element(parameter_table)

        if additional_instructions:
            body.add_element(ContentAST.Paragraph([additional_instructions]))

        if answer_table:
            body.add_element(answer_table)

        return body