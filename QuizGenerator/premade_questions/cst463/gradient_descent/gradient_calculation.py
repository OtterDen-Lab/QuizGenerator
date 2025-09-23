from __future__ import annotations

import abc
import logging
from typing import List
import sympy as sp

from QuizGenerator.misc import ContentAST
from QuizGenerator.question import Question, Answer, QuestionRegistry
from .misc import generate_function

log = logging.getLogger(__name__)


class DerivativeQuestion(Question, abc.ABC):
  """Base class for derivative calculation questions."""

  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.ML_OPTIMIZATION)
    super().__init__(*args, **kwargs)
    self.num_variables = kwargs.get("num_variables", 2)
    self.max_degree = kwargs.get("max_degree", 2)

  def _generate_evaluation_point(self) -> List[float]:
    """Generate a random point for gradient evaluation."""
    return [self.rng.randint(-3, 3) for _ in range(self.num_variables)]

  def _format_partial_derivative(self, var_index: int) -> str:
    """Format partial derivative symbol for display."""
    if self.num_variables == 1:
      return "\\frac{df}{dx_0}"
    else:
      return f"\\frac{{\\partial f}}{{\\partial x_{var_index}}}"

  def _create_derivative_answers(self, evaluation_point: List[float]) -> None:
    """Create answer fields for each partial derivative at the evaluation point."""
    self.answers = {}

    # Evaluate gradient at the specified point
    subs_map = dict(zip(self.variables, evaluation_point))

    # Create answer for each partial derivative
    for i in range(self.num_variables):
      answer_key = f"partial_derivative_{i}"
      # Evaluate the partial derivative and convert to float
      partial_value = self.gradient_function[i].subs(subs_map)
      try:
        gradient_value = float(partial_value)
      except TypeError:
        # If direct float conversion fails, try numerical evaluation
        gradient_value = float(partial_value.evalf())

      self.answers[answer_key] = Answer.float_value(answer_key, gradient_value)

  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()

    # Display the function
    body.add_element(
      ContentAST.Paragraph([
        "Given the function ",
        ContentAST.Equation(sp.latex(self.equation), inline=True),
        ", calculate the gradient at the point ",
        ContentAST.Equation(f"({', '.join(map(str, self.evaluation_point))})", inline=True),
        "."
      ])
    )

    # Create answer fields for each partial derivative
    for i in range(self.num_variables):
      body.add_element(
        ContentAST.Paragraph([
          ContentAST.Equation(self._format_partial_derivative(i), inline=True),
          f" evaluated at ({', '.join(map(str, self.evaluation_point))}) = ",
          ContentAST.Answer(self.answers[f"partial_derivative_{i}"])
        ])
      )

    return body

  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()

    # Show the function and its gradient
    explanation.add_element(
      ContentAST.Paragraph([
        "To find the gradient, we calculate the partial derivatives of ",
        ContentAST.Equation(sp.latex(self.equation), inline=True),
        ":"
      ])
    )

    # Show analytical gradient
    explanation.add_element(
      ContentAST.Equation(f"\\nabla f = {sp.latex(self.gradient_function)}", inline=False)
    )

    # Show evaluation at the specific point
    explanation.add_element(
      ContentAST.Paragraph([
        f"Evaluating at the point ({', '.join(map(str, self.evaluation_point))}):"
      ])
    )

    # Show each partial derivative calculation
    subs_map = dict(zip(self.variables, self.evaluation_point))
    for i in range(self.num_variables):
      partial_expr = self.gradient_function[i]
      partial_value = float(partial_expr.subs(subs_map))

      explanation.add_element(
        ContentAST.Paragraph([
          ContentAST.Equation(
            f"{self._format_partial_derivative(i)} = {sp.latex(partial_expr)} = {partial_value}",
            inline=False
          )
        ])
      )

    return explanation


@QuestionRegistry.register("DerivativeBasic")
class DerivativeBasic(DerivativeQuestion):
  """Basic derivative calculation using polynomial functions."""

  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)

    # Generate a basic polynomial function
    self.variables, self.function, self.gradient_function, self.equation = generate_function(
      self.rng, self.num_variables, self.max_degree
    )

    # Generate evaluation point
    self.evaluation_point = self._generate_evaluation_point()

    # Create answers
    self._create_derivative_answers(self.evaluation_point)


@QuestionRegistry.register("DerivativeChain")
class DerivativeChain(DerivativeQuestion):
  """Chain rule derivative calculation using function composition."""

  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)

    # Generate inner and outer functions for composition
    self._generate_composed_function()

    # Generate evaluation point
    self.evaluation_point = self._generate_evaluation_point()

    # Create answers
    self._create_derivative_answers(self.evaluation_point)

  def _generate_composed_function(self) -> None:
    """Generate a composed function f(g(x)) for chain rule practice."""
    # Create variable symbols
    var_names = [f'x_{i}' for i in range(self.num_variables)]
    self.variables = sp.symbols(var_names)

    # Generate inner function g(x) - simpler polynomial
    inner_terms = [m for m in sp.polys.itermonomials(self.variables, max(1, self.max_degree-1)) if m != 1]
    coeff_pool = [*range(-5, 0), *range(1, 6)]  # Smaller coefficients for inner function

    if inner_terms:
      inner_poly = sp.Add(*(self.rng.choice(coeff_pool) * t for t in inner_terms))
    else:
      inner_poly = sp.Add(*[self.rng.choice(coeff_pool) * v for v in self.variables])

    # Generate outer function - simple function that takes the inner result
    # For simplicity, use powers and basic operations
    u = sp.Symbol('u')  # Intermediate variable
    outer_functions = [
      u**2,
      u**3,
      sp.sin(u),
      sp.cos(u),
      sp.exp(u),
      sp.log(sp.Abs(u) + 1)  # Add 1 to avoid log(0)
    ]

    outer_func = self.rng.choice(outer_functions)

    # Compose the functions: f(g(x))
    self.inner_function = inner_poly
    self.outer_function = outer_func
    self.function = outer_func.subs(u, inner_poly)

    # Calculate gradient using chain rule
    self.gradient_function = sp.Matrix([self.function.diff(v) for v in self.variables])

    # Create equation for display
    f = sp.Function('f')
    self.equation = sp.Eq(f(*self.variables), self.function)

  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()

    # Show the composed function structure
    explanation.add_element(
      ContentAST.Paragraph([
        "This is a composition of functions requiring the chain rule. The function ",
        ContentAST.Equation(sp.latex(self.equation), inline=True),
        " can be written as ",
        ContentAST.Equation(f"f(g(x)) \\text{{ where }} g(x) = {sp.latex(self.inner_function)}", inline=True),
        "."
      ])
    )

    # Show chain rule application
    explanation.add_element(
      ContentAST.Paragraph([
        "Using the chain rule, we calculate:"
      ])
    )

    # Show analytical gradient
    explanation.add_element(
      ContentAST.Equation(f"\\nabla f = {sp.latex(self.gradient_function)}", inline=False)
    )

    # Show evaluation at the specific point
    explanation.add_element(
      ContentAST.Paragraph([
        f"Evaluating at the point ({', '.join(map(str, self.evaluation_point))}):"
      ])
    )

    # Show each partial derivative calculation
    subs_map = dict(zip(self.variables, self.evaluation_point))
    for i in range(self.num_variables):
      partial_expr = self.gradient_function[i]
      partial_value = float(partial_expr.subs(subs_map))

      explanation.add_element(
        ContentAST.Paragraph([
          ContentAST.Equation(
            f"{self._format_partial_derivative(i)} = {sp.latex(partial_expr)} = {partial_value}",
            inline=False
          )
        ])
      )

    return explanation