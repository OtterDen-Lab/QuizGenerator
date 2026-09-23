from __future__ import annotations

import abc
import logging

import sympy
import sympy as sp

import QuizGenerator.generation.contentast as ca
from QuizGenerator.generation.mixins import BodyTemplatesMixin, TableQuestionMixin
from QuizGenerator.generation.question import Question, QuestionRegistry

from .misc import format_vector, generate_function

log = logging.getLogger(__name__)


# Note: This file does not use ca.Answer wrappers - it uses TableQuestionMixin
# which handles answer display through create_answer_table(). The answers are created
# with labels embedded at creation time in _build_context().


class GradientDescentQuestion(Question, abc.ABC):
  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.ML_OPTIMIZATION)
    super().__init__(*args, **kwargs)


@QuestionRegistry.register("GradientDescentWalkthrough")
class GradientDescentWalkthrough(GradientDescentQuestion, TableQuestionMixin, BodyTemplatesMixin):
  DEFAULT_NUM_STEPS = 4
  DEFAULT_NUM_VARIABLES = 2
  DEFAULT_SINGLE_VARIABLE = False
  STEP_DIGITS = 4
  # With the generated quadratics, alpha < 0.5 guarantees a strict decrease
  # away from the optimum even for the largest quadratic coefficient (2).
  LEARNING_RATES = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.num_steps = kwargs.get("num_steps", self.DEFAULT_NUM_STEPS)
    self.num_variables = kwargs.get("num_variables", self.DEFAULT_NUM_VARIABLES)
    self.single_variable = kwargs.get("single_variable", self.DEFAULT_SINGLE_VARIABLE)
    
    if self.single_variable:
      self.num_variables = 1
  
  @classmethod
  def _perform_gradient_descent(
    cls,
    function: sympy.Function,
    gradient_function,
    starting_point,
    num_steps,
    variables,
    learning_rate,
  ) -> list[dict]:
    """
    Perform gradient descent and return step-by-step results.
    """
    results = []
    
    # Each table entry is rounded before it is used in the next row, matching
    # the arithmetic students can reproduce from their submitted work.
    x = [round(float(value), cls.STEP_DIGITS) for value in starting_point]
    
    for step in range(num_steps):
      subs_map = dict(zip(variables, x))
      
      # gradient as floats
      g_syms = gradient_function.subs(subs_map)
      g = [round(float(val), cls.STEP_DIGITS) for val in g_syms]
      
      # function value
      f_val = float(function.subs(subs_map))
      
      update = [round(learning_rate * gi, cls.STEP_DIGITS) for gi in g]
      next_x = [round(xi - ui, cls.STEP_DIGITS) for xi, ui in zip(x, update)]
      
      results.append(
        {
          "step": step + 1,
          "location": x[:],
          "gradient": g[:],
          "update": update[:],
          "function_value": f_val,
          "next_location": next_x[:],
        }
      )

      x = next_x

    return results

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    context.num_steps = kwargs.get("num_steps", cls.DEFAULT_NUM_STEPS)
    if context.num_steps < 1:
      raise ValueError("num_steps must be at least 1")
    context.num_variables = kwargs.get("num_variables", cls.DEFAULT_NUM_VARIABLES)
    context.single_variable = kwargs.get("single_variable", cls.DEFAULT_SINGLE_VARIABLE)
    if context.single_variable:
      context.num_variables = 1
    context.minimize = True

    # Generate function and its properties
    context.use_coupled_quadratic = context.num_variables == 2
    context.variables, context.function, context.gradient_function, context.equation = generate_function(
      context.rng,
      context.num_variables,
      max_degree=2,
      use_quadratic=True,
      use_coupled_quadratic=context.use_coupled_quadratic,
    )

    context.learning_rate = context.rng.choice(cls.LEARNING_RATES)

    context.starting_point = [context.rng.randint(-3, 3) for _ in range(context.num_variables)]

    # Perform gradient descent
    context.gradient_descent_results = cls._perform_gradient_descent(
      context.function,
      context.gradient_function,
      context.starting_point,
      context.num_steps,
      context.variables,
      context.learning_rate,
    )
    context.final_location = context.gradient_descent_results[-1]['next_location']
    context.final_function_value = float(context.function.subs(
      dict(zip(context.variables, context.final_location))
    ))

    # Build answers for each step
    context.step_answers = {}
    for i, result in enumerate(context.gradient_descent_results):
      step = result['step']

      # Location answer
      location_key = f"answer__location_{step}"
      context.step_answers[location_key] = ca.AnswerTypes.Vector(list(result['location']), label=f"Location at step {step}")

      # Gradient answer
      gradient_key = f"answer__gradient_{step}"
      context.step_answers[gradient_key] = ca.AnswerTypes.Vector(list(result['gradient']), label=f"Gradient at step {step}")

      # Update answer
      update_key = f"answer__update_{step}"
      context.step_answers[update_key] = ca.AnswerTypes.Vector(list(result['update']), label=f"Update at step {step}")

    context.step_answers["answer__final_location"] = ca.AnswerTypes.Vector(
      list(context.final_location), label=f"Location after step {context.num_steps}"
    )
    return context

  @classmethod
  def is_interesting_ctx(cls, context) -> bool:
    """Reject zero-gradient starts and any non-decreasing rounded step."""
    results = context.gradient_descent_results
    if not any(abs(value) > 1e-10 for value in results[0]['gradient']):
      return False

    function_values = [result['function_value'] for result in results]
    function_values.append(context.final_function_value)
    return all(
      function_values[index + 1] < function_values[index] - 1e-10
      for index in range(len(function_values) - 1)
    )
  
  @classmethod
  def _build_body(cls, context) -> tuple[ca.Section, list[ca.Answer]]:
    """Build question body and collect answers."""
    self = context
    body = ca.Section()
    answers = []

    body.add_element(
      ca.Paragraph(
        [
          "Use gradient descent to minimize the function ",
          ca.Equation(sp.latex(self.function), inline=True),
          " with learning rate ",
          ca.Equation(f"\\alpha = {self.learning_rate}", inline=True),
          f" and starting point {self.starting_point[0] if self.num_variables == 1 else tuple(self.starting_point)}. "
          "Round each table entry to four decimal places before using it in the next row."
        ]
      )
    )

    # Create table data - use ca.Equation for proper LaTeX rendering in headers
    headers = [
      "t",
      ca.Equation("x^{(t)}", inline=True),
      ca.Equation("\\nabla f", inline=True),
      ca.Equation("\\alpha \\cdot \\nabla f", inline=True)
    ]
    table_rows = []

    for i, result in enumerate(self.gradient_descent_results):
      step = result['step']
      row = {"t": str(i)}

      if i == 0:

        # Fill in starting location for first row with default formatting
        row[headers[1]] = f"{format_vector(self.starting_point)}"
        row[headers[2]] = self.step_answers[f"answer__gradient_{step}"]  # gradient column
        row[headers[3]] = self.step_answers[f"answer__update_{step}"]  # update column
        # Collect answers for this step (no location answer for step 1)
        answers.append(self.step_answers[f"answer__gradient_{step}"])
        answers.append(self.step_answers[f"answer__update_{step}"])
      else:
        # Subsequent rows - all answer fields
        row[headers[1]] = self.step_answers[f"answer__location_{step}"]
        row[headers[2]] = self.step_answers[f"answer__gradient_{step}"]
        row[headers[3]] = self.step_answers[f"answer__update_{step}"]
        # Collect all answers for this step
        answers.append(self.step_answers[f"answer__location_{step}"])
        answers.append(self.step_answers[f"answer__gradient_{step}"])
        answers.append(self.step_answers[f"answer__update_{step}"])
      table_rows.append(row)

    # The final row makes the destination of the final requested update visible.
    table_rows.append({
      "t": str(self.num_steps),
      headers[1]: self.step_answers["answer__final_location"],
      headers[2]: "—",
      headers[3]: "—",
    })
    answers.append(self.step_answers["answer__final_location"])

    # Create the table using mixin
    gradient_table = cls.create_answer_table(
      headers=headers,
      data_rows=table_rows,
      answer_columns=[headers[1], headers[2], headers[3]]  # Use actual header objects
    )

    body.add_element(gradient_table)

    return body, answers

  @classmethod
  def _build_explanation(cls, context) -> tuple[ca.Section, list[ca.Answer]]:
    """Build question explanation."""
    self = context
    explanation = ca.Section()

    explanation.add_element(
      ca.Paragraph(
        [
          "Gradient descent is an optimization algorithm that iteratively moves towards "
          "the minimum of a function by taking steps proportional to the negative of the gradient."
        ]
      )
    )

    explanation.add_element(
      ca.Paragraph(
        [
          "We want to minimize the function ",
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
          "Since we want to minimize, we use the update rule: ",
          ca.Equation(r"x^{(t+1)} = x^{(t)} - \alpha \nabla f(x^{(t)})", inline=True),
          f". We start at {tuple(self.starting_point)} with learning rate ",
          ca.Equation(f"\\alpha = {self.learning_rate}", inline=True),
          ". Round each table entry to four decimal places before continuing."
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
      "t",
      ca.Equation("x^{(t)}", inline=True),
      ca.Equation("\\nabla f", inline=True),
      ca.Equation("\\alpha \\cdot \\nabla f", inline=True)
    ]

    solution_rows = []
    for i, result in enumerate(self.gradient_descent_results):
      row = {"t": str(i)}

      row[solution_headers[1]] = f"{format_vector(result['location'])}"
      row[solution_headers[2]] = f"{format_vector(result['gradient'])}"
      row[solution_headers[3]] = f"{format_vector(result['update'])}"

      solution_rows.append(row)

    solution_rows.append({
      "t": str(self.num_steps),
      solution_headers[1]: format_vector(self.final_location),
      solution_headers[2]: "—",
      solution_headers[3]: "—",
    })

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
        current_loc = result['location']
        update = result['update']
        next_loc = result['next_location']

        explanation.add_element(
          ca.Paragraph(
            [
              f"Next location: {format_vector(current_loc)} - {format_vector(result['update'])} = {format_vector(next_loc)}"
            ]
          )
        )

    function_values = [r['function_value'] for r in self.gradient_descent_results]
    function_values.append(self.final_function_value)
    explanation.add_element(
      ca.Paragraph(
        [
          f"Function values: {[f'{v:.4f}' for v in function_values]}"
        ]
      )
    )

    return explanation, []
