#!env python
from __future__ import annotations

import collections
import itertools
import logging
import os.path
import random
import shutil
import subprocess
import tempfile
from datetime import datetime
from typing import List, Dict, Optional

import yaml

from QuizGenerator.misc import OutputFormat
from QuizGenerator.contentast import ContentAST
from QuizGenerator.question import Question, QuestionRegistry, QuestionGroup

log = logging.getLogger(__name__)


class Quiz:
  """
  A quiz object that will build up questions and output them in a range of formats (hopefully)
  It should be that a single quiz object can contain multiples -- essentially it builds up from the questions and then can generate a variety of questions.
  """
  
  INTEREST_THRESHOLD = 1.0
  
  def __init__(self, name, questions: List[dict|Question], practice, *args, **kwargs):
    self.name = name
    self.questions = questions
    self.instructions = kwargs.get("instructions", "")

    # Parse description with ContentAST if provided
    raw_description = kwargs.get("description", None)
    if raw_description:
      # Create a ContentAST document from the description text
      desc_doc = ContentAST.Document()
      desc_doc.add_element(ContentAST.Paragraph([raw_description]))
      self.description = desc_doc.render("html")
    else:
      self.description = None

    self.question_sort_order = None
    self.practice = practice
    self.preserve_order_point_values = set()  # Point values that should preserve question order

    # Plan: right now we just take in questions and then assume they have a score and a "generate" button
  
  def __iter__(self):
    def sort_func(q):
      if self.question_sort_order is not None:
        try:
          return (-q.points_value, self.question_sort_order.index(q.topic))
        except ValueError:
          return (-q.points_value, float('inf'))
      return -q.points_value
    return iter(sorted(self.questions, key=sort_func))
  
  @classmethod
  def from_yaml(cls, path_to_yaml) -> List[Quiz]:
    
    quizes_loaded : List[Quiz] = []
    
    with open(path_to_yaml) as fid:
      list_of_exam_dicts = list(yaml.safe_load_all(fid))
    
    for exam_dict in list_of_exam_dicts:
      # Get general quiz information from the dictionary
      name = exam_dict.get("name", f"Unnamed Exam ({datetime.now().strftime('%a %b %d %I:%M %p')})")
      practice = exam_dict.get("practice", False)
      description = exam_dict.get("description", None)
      sort_order = list(map(lambda t: Question.Topic.from_string(t), exam_dict.get("sort order", [])))
      sort_order = sort_order + list(filter(lambda t: t not in sort_order, Question.Topic))
      
      # Load questions from the quiz dictionary
      questions_for_exam = []
      # Track point values where order should be preserved (for layout optimization)
      preserve_order_point_values = set()

      for question_value, question_definitions in exam_dict["questions"].items():
        # todo: I can also add in "extra credit" and "mix-ins" as other keys to indicate extra credit or questions that can go anywhere
        log.info(f"Parsing {question_value} point questions")

        # Check for point-value-level config
        point_config = question_definitions.pop("_config", {})
        if point_config.get("preserve_order", False):
          preserve_order_point_values.add(question_value)
          log.info(f"  Point value {question_value} will preserve question order")

        def make_question(q_name, q_data, **kwargs):
          # Build up the kwargs that we're going to pass in
          # todo: this is currently a mess due to legacy things, so before I tell others to use this make it cleaner
          kwargs= {
            "name": q_name,
            "points_value": question_value,
            **q_data.get("kwargs", {}),
            **q_data,
            **kwargs,
          }
          
          # If we are passed in a topic then use it, otherwise don't set it which will have it set to a default
          if "topic" in q_data:
            kwargs["topic"] = Question.Topic.from_string(q_data.get("topic", "Misc"))
          
          # Add in a default, where if it isn't specified we're going to simply assume it is a text
          question_class = q_data.get("class", "FromText")
          
          new_question = QuestionRegistry.create(
            question_class,
            **kwargs
          )
          return new_question
        
        for q_name, q_data in question_definitions.items():
          # Set defaults for config
          question_config = {
            "group" : False,
            "num_to_pick" : 1,
            "random_per_student" : False,
            "repeat": 1,
            "topic": "MISC"
          }
          
          # Update config if it exists
          question_config.update(
            q_data.get("_config", {})
          )
          q_data.pop("_config", None)
          q_data.pop("pick", None) # todo: don't use this anymore
          q_data.pop("repeat", None) # todo: don't use this anymore
          
          # Check if it is a question group
          if question_config["group"]:
            
            # todo: Find a way to allow for "num_to_pick" to ensure lack of duplicates when using duplicates.
            #    It's probably going to be somewhere in the instantiate and get_attr fields, with "_current_questions"
            #    But will require changing how we add concrete questions (but that'll just be everything returns a list
            questions_for_exam.append(
              QuestionGroup(
                questions_in_group=[
                  make_question(name, data | {"topic" : question_config["topic"]}) for name, data in q_data.items()
                ],
                pick_once=(not question_config["random_per_student"])
              )
            )
          
          else: # Then this is just a single question
            questions_for_exam.extend([
              make_question(
                q_name,
                q_data,
                rng_seed_offset=repeat_number
              )
              for repeat_number in range(question_config["repeat"])
            ])
      log.debug(f"len(questions_for_exam): {len(questions_for_exam)}")
      quiz_from_yaml = cls(name, questions_for_exam, practice, description=description)
      quiz_from_yaml.set_sort_order(sort_order)
      quiz_from_yaml.preserve_order_point_values = preserve_order_point_values
      quizes_loaded.append(quiz_from_yaml)
    return quizes_loaded
  
  def _estimate_question_height(self, question, use_typst_measurement=False, **kwargs) -> float:
    """
    Estimate the rendered height of a question for layout optimization.
    Returns height in centimeters.

    Args:
        question: Question object to measure
        use_typst_measurement: If True, use Typst's layout engine for exact measurement
        **kwargs: Additional arguments passed to question rendering

    Returns:
        Height in centimeters
    """
    # Try Typst measurement if requested and available
    if use_typst_measurement:
      from QuizGenerator.typst_utils import measure_typst_content, check_typst_available

      if check_typst_available():
        try:
          # Render question to Typst
          question_ast = question.get_question(**kwargs)

          # Get just the content body (without the #question wrapper which adds spacing)
          typst_body = question_ast.body.render("typst", **kwargs)

          # Measure the content
          measured_height = measure_typst_content(typst_body, page_width_cm=18.0)

          if measured_height is not None:
            # Add base height for question formatting (header, line, etc.) ~1.5cm
            # Plus the spacing parameter
            total_height = 1.5 + measured_height + question.spacing
            log.debug(f"Typst measurement: {question.name} = {total_height:.2f}cm (content: {measured_height:.2f}cm, spacing: {question.spacing}cm)")
            return total_height
          else:
            log.debug(f"Typst measurement failed for {question.name}, falling back to heuristics")
        except Exception as e:
          log.warning(f"Error during Typst measurement: {e}, falling back to heuristics")
      else:
        log.debug("Typst not available, using heuristic estimation")

    # Fallback: Use heuristic estimation (original implementation)
    # Base height for question header, borders, and minimal content
    # Each question has: horizontal rule, question number line, and minipage wrapper
    base_height = 1.5  # cm

    # The spacing parameter directly controls \vspace{} in cm
    spacing_height = question.spacing  # cm

    # Estimate content height by rendering to LaTeX and analyzing structure
    question_ast = question.get_question(**kwargs)
    latex_content = question_ast.render("latex")

    # Count content that adds height (rough estimates in cm)
    content_height = 0.0

    # Tables add significant height (~0.5cm per row as rough estimate)
    table_count = latex_content.count('\\begin{tabular}')
    content_height += table_count * 3.0  # Assume ~3cm per table on average

    # Matrices add height
    matrix_count = latex_content.count('\\begin{') - table_count  # Rough matrix count
    content_height += matrix_count * 2.0  # ~2cm per matrix

    # Code blocks (verbatim) add significant height
    verbatim_count = latex_content.count('\\begin{verbatim}')
    content_height += verbatim_count * 4.0  # ~4cm per code block

    # Count paragraphs and text blocks (very rough estimate)
    # Each ~500 characters of text ≈ 1cm of height
    char_count = len(latex_content)
    content_height += (char_count / 500.0) * 0.5

    # Total estimated height
    total_height = base_height + spacing_height + content_height

    return total_height

  def _optimize_question_order(self, questions, **kwargs) -> List[Question]:
    """
    Optimize question ordering to minimize PDF length while respecting point-value tiers.
    Uses bin-packing heuristics to reorder questions within each point-value group.
    """
    # Group questions by point value
    from collections import defaultdict
    point_groups = defaultdict(list)

    for question in questions:
      point_groups[question.points_value].append(question)

    # Track which point values should preserve order (from config)
    preserve_order_for = kwargs.pop('preserve_order_for', set())

    # For each point group, estimate heights and apply bin-packing optimization
    optimized_questions = []
    is_first_page = True  # Track if we're packing the first page

    log.debug("Optimizing question order for PDF layout...")

    for points in sorted(point_groups.keys(), reverse=True):
      group = point_groups[points]

      # Check if this point tier should preserve order
      if points in preserve_order_for:
        # Sort by topic only (preserve original order)
        group.sort(key=lambda q: self.question_sort_order.index(q.topic))
        optimized_questions.extend(group)
        log.debug(f"  {points}pt questions: {len(group)} questions (order preserved by config)")
        # After adding preserved-order questions, we're likely past the first page
        is_first_page = False
        continue

      # If only 1-2 questions, no optimization needed
      if len(group) <= 2:
        # Still sort by topic for consistency
        group.sort(key=lambda q: self.question_sort_order.index(q.topic))
        optimized_questions.extend(group)
        log.debug(f"  {points}pt questions: {len(group)} questions (no optimization needed)")
        is_first_page = False
        continue

      # Estimate height for each question
      question_heights = [(q, self._estimate_question_height(q, **kwargs)) for q in group]

      # Sort by height descending to identify large and small questions
      question_heights.sort(key=lambda x: x[1], reverse=True)

      log.debug(f"  Question heights for {points}pt questions:")
      for q, h in question_heights:
        log.debug(f"    {q.name}: {h:.1f}cm (spacing={q.spacing}cm)")

      # Calculate page capacity in centimeters
      # A typical A4 page with margins has ~25cm of usable height
      # After accounting for headers and separators, estimate ~22cm per page
      base_page_capacity = 22.0  # cm

      # First page has header (title + name line) which takes ~3cm
      first_page_capacity = base_page_capacity - 3.0 if is_first_page else base_page_capacity

      # Better bin-packing strategy: interleave large and small questions
      # Strategy: Start each page with the largest unplaced question, then fill with smaller ones
      bins = []
      placed = [False] * len(question_heights)

      while not all(placed):
        # Determine capacity for this page
        page_capacity = first_page_capacity if len(bins) == 0 and is_first_page else base_page_capacity

        # Find the largest unplaced question to start a new page
        new_page = []
        page_height = 0

        for i, (question, height) in enumerate(question_heights):
          if not placed[i]:
            new_page.append(question)
            page_height = height
            placed[i] = True
            break

        # Now try to fill the remaining space with smaller questions
        for i, (question, height) in enumerate(question_heights):
          if not placed[i] and page_height + height <= page_capacity:
            new_page.append(question)
            page_height += height
            placed[i] = True

        bins.append((new_page, page_height))

      log.debug(f"  {points}pt questions: {len(group)} questions packed into {len(bins)} pages")
      for i, (page_questions, height) in enumerate(bins):
        log.debug(f"    Page {i+1}: {height:.1f}cm with {len(page_questions)} questions: {[q.name for q in page_questions]}")

      # Flatten bins back to ordered list
      for bin_contents, _ in bins:
        optimized_questions.extend(bin_contents)

      # After packing questions, we're no longer on the first page
      is_first_page = False

    return optimized_questions

  def get_quiz(self, **kwargs) -> ContentAST.Document:
    quiz = ContentAST.Document(title=self.name)

    # Check if optimization is requested (default: True)
    optimize_layout = kwargs.pop('optimize_layout', True)

    if optimize_layout:
      # Use optimized ordering, passing preserve_order config
      ordered_questions = self._optimize_question_order(
        self.questions,
        preserve_order_for=self.preserve_order_point_values,
        **kwargs
      )
    else:
      # Use simple ordering by point value and topic
      ordered_questions = sorted(
        self.questions,
        key=lambda q: (-q.points_value, self.question_sort_order.index(q.topic))
      )

    # Generate questions with sequential numbering for QR codes
    for question_number, question in enumerate(ordered_questions, start=1):
      question_ast = question.get_question(**kwargs)
      # Add question number to the AST for QR code generation
      question_ast.question_number = question_number
      quiz.add_element(question_ast)

    return quiz
  
  def describe(self):
    
    # Print out title
    print(f"Title: {self.name}")
    total_points = sum(map(lambda q: q.points_value, self.questions))
    total_questions = len(self.questions)
    
    # Print out overall information
    print(f"{total_points} points total, {total_questions} questions")
    
    # Print out the per-value information
    points_counter = collections.Counter([q.points_value for q in self.questions])
    for points in sorted(points_counter.keys(), reverse=True):
      print(f"{points_counter.get(points)} x {points}points")
    
    # Either get the sort order or default to the order in the enum class
    sort_order = self.question_sort_order
    if sort_order is None:
      sort_order = Question.Topic
      
    # Build per-topic information
    
    topic_information = {}
    topic_strings = {}
    for topic in sort_order:
      topic_strings = {"name": topic.name}
      
      question_count = len(list(map(lambda q: q.points_value, filter(lambda q: q.topic == topic, self.questions))))
      topic_points = sum(map(lambda q: q.points_value, filter(lambda q: q.topic == topic, self.questions)))
      
      # If we have questions add in some states, otherwise mark them as empty
      if question_count != 0:
        topic_strings["count_str"] = f"{question_count} questions ({ 100 * question_count / total_questions:0.1f}%)"
        topic_strings["points_str"] = f"{topic_points:2} points ({ 100 * topic_points / total_points:0.1f}%)"
      else:
        topic_strings["count_str"] = "--"
        topic_strings["points_str"] = "--"
      
      topic_information[topic] = topic_strings
    
    
    # Get padding string lengths
    paddings = collections.defaultdict(lambda: 0)
    for field in topic_strings.keys():
      paddings[field] = max(len(information[field]) for information in topic_information.values())
    
    # Print out topics information using the padding
    for topic in sort_order:
      topic_strings = topic_information[topic]
      print(f"{topic_strings['name']:{paddings['name']}} : {topic_strings['count_str']:{paddings['count_str']}} : {topic_strings['points_str']:{paddings['points_str']}}")
    
  def set_sort_order(self, sort_order):
    self.question_sort_order = sort_order

def main():
  pass
  

if __name__ == "__main__":
  main()
  