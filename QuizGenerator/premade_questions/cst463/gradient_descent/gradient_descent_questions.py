from __future__ import annotations

import abc
import logging
import math
from typing import List, Tuple, Callable, Union
import numpy as np
import sympy as sp
# from sympy import symbols, latex, diff, lambdify


from QuizGenerator.misc import ContentAST
from QuizGenerator.question import Question, Answer, QuestionRegistry
from QuizGenerator.mixins import TableQuestionMixin, BodyTemplatesMixin

log = logging.getLogger(__name__)


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
  
  def _generate_function(self) -> Tuple[Callable, Callable, str, List[float]]:
    """
    Generate a function, its gradient, LaTeX representation, and optimal point using SymPy.
    Supports 1-5 variables.
    Returns: (function, gradient_function, latex_string, optimal_point)
    """
    # Create variable symbols
    
    # variables: tuple even for a single variable
    var_names = [f'x_{i}' for i in range(self.num_variables)]
    self.variables = sp.symbols(var_names)  # returns a tuple; robust when n==1
    
    # monomials up to max_degree; drop constant 1
    terms = [m for m in sp.polys.itermonomials(self.variables, self.max_degree) if m != 1]
    
    # random nonzero integer coefficients in [-10,-1] ∪ [1,9]
    coeff_pool = [*range(-10, 0), *range(1, 10)]
    
    # polynomial; if no terms (e.g., max_degree==0), fall back to 0
    poly = sp.Add(*(self.rng.choice(coeff_pool) * t for t in terms)) if terms else sp.Integer(0)
    
    # f(x_1, ..., x_n) = poly
    f = sp.Function('f')
    self.function = poly
    self.gradient_function = sp.Matrix([poly.diff(v) for v in self.variables])
    self.equation = sp.Eq(f(*self.variables), poly)
    
    return
    
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
  
  def _format_vector(self, vec: List[float], decimal_places: int = 4) -> str:
    """Format a vector for display, handling single vs multi-variable cases."""
    if len(vec) == 1:
      return f"{vec[0]:.{decimal_places}f}"
    else:
      formatted_elements = [f"{x:.{decimal_places}f}" for x in vec]
      return f"[{', '.join(formatted_elements)}]"
  
  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)
    log.debug("Refreshing...")
    
    # Generate function and its properties
    self._generate_function()
    # self.function, self.gradient_function, self.function_latex, self.optimal_point = self._generate_function()
    
    log.debug(self.function)
    log.debug(sp.latex(self.equation))
    
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
      self.answers[location_key] = Answer.point_value(location_key, tuple(result['location']))
      
      # Gradient answer
      gradient_key = f"answer__gradient_{step}"
      self.answers[gradient_key] = Answer.point_value(gradient_key, tuple(result['gradient']))
      
      # Update answer
      update_key = f"answer__update_{step}"
      self.answers[update_key] = Answer.point_value(update_key, tuple(result['update']))
  
  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()
    
    # Introduction
    objective = "minimize" if self.minimize else "maximize"
    sign = "-" if self.minimize else "+"
    
    body.add_element(
      ContentAST.Paragraph(
        [
          f"Use gradient descent to {objective} the function ",
          ContentAST.Equation(sp.latex(self.function), inline=True),
          " with learning rate ",
          ContentAST.Equation(f"\\alpha = {self.learning_rate}", inline=True),
          f" and starting point {self.starting_point[0] if self.num_variables == 1 else tuple(self.starting_point)}. "
          "Fill in the table below with your calculations."
        ]
      )
    )
    
    # Create table data - use ContentAST.Equation for proper LaTeX rendering in headers
    headers = [
      "n",
      "location",
      ContentAST.Equation("\\nabla f", inline=True),
      ContentAST.Equation("\\alpha \\cdot \\nabla f", inline=True)
    ]
    table_rows = []
    
    for i in range(self.num_steps):
      step = i + 1
      row = {"n": str(step)}
      
      if step == 1:
        # Fill in starting location for first row with default formatting
        row["location"] = f"{self.starting_point[0]:0.{Answer.DEFAULT_ROUNDING_DIGITS}f}"
        row[headers[2]] = f"answer__gradient_{step}"  # gradient column
        row[headers[3]] = f"answer__update_{step}"  # update column
      else:
        # Subsequent rows - all answer fields
        row["location"] = f"answer__location_{step}"
        row[headers[2]] = f"answer__gradient_{step}"  # gradient column
        row[headers[3]] = f"answer__update_{step}"  # update column
      table_rows.append(row)
    
    # Create the table using mixin
    gradient_table = self.create_answer_table(
      headers=headers,
      data_rows=table_rows,
      answer_columns=["location", headers[2], headers[3]]  # Use actual header objects
    )
    
    body.add_element(gradient_table)
    
    return body
  
  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    explanation.add_element(
      ContentAST.Paragraph(
        [
          "Gradient descent is an optimization algorithm that iteratively moves towards "
          "the minimum of a function by taking steps proportional to the negative of the gradient."
        ]
      )
    )
    
    objective = "minimize" if self.minimize else "maximize"
    sign = "-" if self.minimize else "+"
    
    explanation.add_element(
      ContentAST.Paragraph(
        [
          f"Since we want to {objective} the function ",
          ContentAST.Equation(sp.latex(self.function), inline=True),
          f", we should use the update rule: ",
          ContentAST.Equation(f"x_{{new}} = x_{{old}} {sign} \\alpha \\nabla f", inline=True),
          f". We start at {tuple(self.starting_point)} with learning rate ",
          ContentAST.Equation(f"\\alpha = {self.learning_rate}", inline=True),
          "."
        ]
      )
    )
    
    # Add completed table showing all solutions
    explanation.add_element(
      ContentAST.Paragraph(
        [
          "**Solution Table:**"
        ]
      )
    )
    
    # Create filled solution table
    solution_headers = [
      "n",
      "location",
      ContentAST.Equation("\\nabla f", inline=True),
      ContentAST.Equation("\\alpha \\cdot \\nabla f", inline=True)
    ]
    
    solution_rows = []
    for i, result in enumerate(self.gradient_descent_results):
      step = result['step']
      row = {"n": str(step)}
      
      row["location"] = f"{tuple(result['location'])}"
      row[solution_headers[2]] = f"{tuple(result['gradient'])}"
      row[solution_headers[3]] = f"{tuple(result['update'])}"
    
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
        ContentAST.Paragraph(
          [
            f"**Step {step}:**"
          ]
        )
      )
      
      explanation.add_element(
        ContentAST.Paragraph(
          [
            f"Location: {self._format_vector(result['location'])}"
          ]
        )
      )
      
      explanation.add_element(
        ContentAST.Paragraph(
          [
            f"Gradient: {self._format_vector(result['gradient'])}"
          ]
        )
      )
      
      explanation.add_element(
        ContentAST.Paragraph(
          [
            "Update: ",
            ContentAST.Equation(
              f"\\alpha \\cdot \\nabla f = {self.learning_rate} \\cdot {self._format_vector(result['gradient'])} = {self._format_vector(result['update'])}",
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
          ContentAST.Paragraph(
            [
              f"Next location: {self._format_vector(current_loc)} - {self._format_vector(result['update'])} = {self._format_vector(next_loc)}"
            ]
          )
        )
    
    function_values = [r['function_value'] for r in self.gradient_descent_results]
    explanation.add_element(
      ContentAST.Paragraph(
        [
          f"Function values: {[f'{v:.4f}' for v in function_values]}"
        ]
      )
    )
    
    return explanation
