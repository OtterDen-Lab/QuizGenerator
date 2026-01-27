from __future__ import annotations

import abc
import logging
import math
from typing import List, Tuple, Callable, Union, Any
import sympy as sp

import QuizGenerator.contentast as ca
from QuizGenerator.question import Question, QuestionRegistry
from QuizGenerator.mixins import TableQuestionMixin, BodyTemplatesMixin

from .misc import generate_function, format_vector

log = logging.getLogger(__name__)


# Note: This file does not use ca.Answer wrappers - it uses TableQuestionMixin
# which handles answer display through create_answer_table(). The answers are created
# with labels embedded at creation time in refresh().


class GradientDescentQuestion(Question, abc.ABC):
  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.ML_OPTIMIZATION)
    super().__init__(*args, **kwargs)


@QuestionRegistry.register("GradientDescentWalkthrough")
class GradientDescentWalkthrough(GradientDescentQuestion, TableQuestionMixin, BodyTemplatesMixin):
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.num_steps = kwargs.get("num_steps", 4)
    self.num_variables = kwargs.get("num_variables", 2)
    self.max_degree = kwargs.get("max_degree", 2)
    self.single_variable = kwargs.get("single_variable", False)
    self.minimize = kwargs.get("minimize", True)  # Default to minimization
    
    if self.single_variable:
      self.num_variables = 1
      
  def _perform_gradient_descent(self) -> List[dict]:
    """
    Perform gradient descent and return step-by-step results.
    """
    results = []
    
    x = list(map(float, self.starting_point))  # current location as floats
    
    for step in range(self.num_steps):
      subs_map = dict(zip(self.variables, x))
      
      # gradient as floats
      g_syms = self.gradient_function.subs(subs_map)
      g = [float(val) for val in g_syms]
      
      # function value
      f_val = float(self.function.subs(subs_map))
      
      update = [self.learning_rate * gi for gi in g]
      
      results.append(
        {
          "step": step + 1,
          "location": x[:],
          "gradient": g[:],
          "update": update[:],
          "function_value": f_val,
        }
      )
      
      x = [xi - ui for xi, ui in zip(x, update)] if self.minimize else \
        [xi + ui for xi, ui in zip(x, update)]

    return results

  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)
    
    # Generate function and its properties
    self.variables, self.function, self.gradient_function, self.equation = generate_function(self.rng, self.num_variables, self.max_degree)
    
    # Generate learning rate (expanded range)
    self.learning_rate = self.rng.choice([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    
    self.starting_point = [self.rng.randint(-3, 3) for _ in range(self.num_variables)]
    
    # Perform gradient descent
    self.gradient_descent_results = self._perform_gradient_descent()
    
    # Set up answers
    self.answers = {}

    # Answers for each step
    for i, result in enumerate(self.gradient_descent_results):
      step = result['step']

      # Location answer
      location_key = f"answer__location_{step}"
      self.answers[location_key] = ca.AnswerTypes.Vector(list(result['location']), label=f"Location at step {step}")

      # Gradient answer
      gradient_key = f"answer__gradient_{step}"
      self.answers[gradient_key] = ca.AnswerTypes.Vector(list(result['gradient']), label=f"Gradient at step {step}")

      # Update answer
      update_key = f"answer__update_{step}"
      self.answers[update_key] = ca.AnswerTypes.Vector(list(result['update']), label=f"Update at step {step}")
  
  def _get_body(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    # Introduction
    objective = "minimize" if self.minimize else "maximize"
    sign = "-" if self.minimize else "+"

    body.add_element(
      ca.Paragraph(
        [
          f"Use gradient descent to {objective} the function ",
          ca.Equation(sp.latex(self.function), inline=True),
          " with learning rate ",
          ca.Equation(f"\\alpha = {self.learning_rate}", inline=True),
          f" and starting point {self.starting_point[0] if self.num_variables == 1 else tuple(self.starting_point)}. "
          "Fill in the table below with your calculations."
        ]
      )
    )

    # Create table data - use ca.Equation for proper LaTeX rendering in headers
    headers = [
      "n",
      "location",
      ca.Equation("\\nabla f", inline=True),
      ca.Equation("\\alpha \\cdot \\nabla f", inline=True)
    ]
    table_rows = []

    for i in range(self.num_steps):
      step = i + 1
      row = {"n": str(step)}

      if step == 1:

        # Fill in starting location for first row with default formatting
        row["location"] = f"{format_vector(self.starting_point)}"
        row[headers[2]] = f"answer__gradient_{step}"  # gradient column
        row[headers[3]] = f"answer__update_{step}"  # update column
        # Collect answers for this step (no location answer for step 1)
        answers.append(self.answers[f"answer__gradient_{step}"])
        answers.append(self.answers[f"answer__update_{step}"])
      else:
        # Subsequent rows - all answer fields
        row["location"] = f"answer__location_{step}"
        row[headers[2]] = f"answer__gradient_{step}"  # gradient column
        row[headers[3]] = f"answer__update_{step}"  # update column
        # Collect all answers for this step
        answers.append(self.answers[f"answer__location_{step}"])
        answers.append(self.answers[f"answer__gradient_{step}"])
        answers.append(self.answers[f"answer__update_{step}"])
      table_rows.append(row)

    # Create the table using mixin
    gradient_table = self.create_answer_table(
      headers=headers,
      data_rows=table_rows,
      answer_columns=["location", headers[2], headers[3]]  # Use actual header objects
    )

    body.add_element(gradient_table)

    return body, answers

  def get_body(self, **kwargs) -> ca.Section:
    """Build question body (backward compatible interface)."""
    body, _ = self._get_body(**kwargs)
    return body
  
  def _get_explanation(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(
      ca.Paragraph(
        [
          "Gradient descent is an optimization algorithm that iteratively moves towards "
          "the minimum of a function by taking steps proportional to the negative of the gradient."
        ]
      )
    )

    objective = "minimize" if self.minimize else "maximize"
    sign = "-" if self.minimize else "+"

    explanation.add_element(
      ca.Paragraph(
        [
          f"We want to {objective} the function ",
          ca.Equation(sp.latex(self.function), inline=True),
          ". First, we calculate the analytical gradient:"
        ]
      )
    )

    # Add analytical gradient calculation as a display equation (vertical vector)
    explanation.add_element(
      ca.Equation(f"\\nabla f = {sp.latex(self.gradient_function)}", inline=False)
    )

    explanation.add_element(
      ca.Paragraph(
        [
          f"Since we want to {objective}, we use the update rule: ",
          ca.Equation(f"x_{{new}} = x_{{old}} {sign} \\alpha \\nabla f", inline=True),
          f". We start at {tuple(self.starting_point)} with learning rate ",
          ca.Equation(f"\\alpha = {self.learning_rate}", inline=True),
          "."
        ]
      )
    )

    # Add completed table showing all solutions
    explanation.add_element(
      ca.Paragraph(
        [
          "**Solution Table:**"
        ]
      )
    )

    # Create filled solution table
    solution_headers = [
      "n",
      "location",
      ca.Equation("\\nabla f", inline=True),
      ca.Equation("\\alpha \\cdot \\nabla f", inline=True)
    ]

    solution_rows = []
    for i, result in enumerate(self.gradient_descent_results):
      step = result['step']
      row = {"n": str(step)}

      row["location"] = f"{format_vector(result['location'])}"
      row[solution_headers[2]] = f"{format_vector(result['gradient'])}"
      row[solution_headers[3]] = f"{format_vector(result['update'])}"

      solution_rows.append(row)

    # Create solution table (non-answer table, just display)
    solution_table = self.create_answer_table(
      headers=solution_headers,
      data_rows=solution_rows,
      answer_columns=[]  # No answer columns since this is just for display
    )

    explanation.add_element(solution_table)

    # Step-by-step explanation
    for i, result in enumerate(self.gradient_descent_results):
      step = result['step']

      explanation.add_element(
        ca.Paragraph(
          [
            f"**Step {step}:**"
          ]
        )
      )

      explanation.add_element(
        ca.Paragraph(
          [
            f"Location: {format_vector(result['location'])}"
          ]
        )
      )

      explanation.add_element(
        ca.Paragraph(
          [
            f"Gradient: {format_vector(result['gradient'])}"
          ]
        )
      )

      explanation.add_element(
        ca.Paragraph(
          [
            "Update: ",
            ca.Equation(
              f"\\alpha \\cdot \\nabla f = {self.learning_rate} \\cdot {format_vector(result['gradient'])} = {format_vector(result['update'])}",
              inline=True
            )
          ]
        )
      )

      if step < len(self.gradient_descent_results):
        # Calculate next location for display
        current_loc = result['location']
        update = result['update']
        next_loc = [current_loc[j] - update[j] for j in range(len(current_loc))]

        explanation.add_element(
          ca.Paragraph(
            [
              f"Next location: {format_vector(current_loc)} - {format_vector(result['update'])} = {format_vector(next_loc)}"
            ]
          )
        )

    function_values = [r['function_value'] for r in self.gradient_descent_results]
    explanation.add_element(
      ca.Paragraph(
        [
          f"Function values: {[f'{v:.4f}' for v in function_values]}"
        ]
      )
    )

    return explanation, []

  def get_explanation(self, **kwargs) -> ca.Section:
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation(**kwargs)
    return explanation
