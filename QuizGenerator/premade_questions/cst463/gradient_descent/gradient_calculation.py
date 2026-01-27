from __future__ import annotations

import abc
import logging
from typing import List, Tuple
import sympy as sp

import QuizGenerator.contentast as ca
from QuizGenerator.question import Question, QuestionRegistry
from .misc import generate_function, format_vector

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

    # Format evaluation point for label
    eval_point_str = ", ".join([f"x_{i} = {evaluation_point[i]}" for i in range(self.num_variables)])

    # Create answer for each partial derivative
    for i in range(self.num_variables):
      answer_key = f"partial_derivative_{i}"
      # Evaluate the partial derivative and convert to float
      partial_value = self.gradient_function[i].subs(subs_map)
      try:
        gradient_value = float(partial_value)
      except (TypeError, ValueError):
        # If we get a complex number or other conversion error,
        # this likely means log hit a negative value - regenerate
        raise ValueError("Complex number encountered - need to regenerate")

      # Use auto_float for Canvas compatibility with integers and decimals
      # Label includes the partial derivative notation
      label = f"∂f/∂x_{i} at ({eval_point_str})"
      self.answers[answer_key] = ca.AnswerTypes.Float(gradient_value, label=label)

  def _create_gradient_vector_answer(self) -> None:
    """Create a single gradient vector answer for PDF format."""
    # Format gradient as vector notation
    subs_map = dict(zip(self.variables, self.evaluation_point))
    gradient_values = []

    for i in range(self.num_variables):
      partial_value = self.gradient_function[i].subs(subs_map)
      try:
        gradient_value = float(partial_value)
      except TypeError:
        gradient_value = float(partial_value.evalf())
      gradient_values.append(gradient_value)

    # Format as vector for display using consistent formatting
    vector_str = format_vector(gradient_values)
    self.answers["gradient_vector"] = ca.AnswerTypes.String(vector_str, pdf_only=True)

  def _get_body(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    # Display the function
    body.add_element(
      ca.Paragraph([
        "Given the function ",
        ca.Equation(sp.latex(self.equation), inline=True),
        ", calculate the gradient at the point ",
        ca.Equation(format_vector(self.evaluation_point), inline=True),
        "."
      ])
    )

    # Format evaluation point for LaTeX
    eval_point_str = ", ".join([f"x_{i} = {self.evaluation_point[i]}" for i in range(self.num_variables)])

    # For PDF: Use OnlyLatex to show gradient vector format (no answer blank)
    body.add_element(
      ca.OnlyLatex([
        ca.Paragraph([
          ca.Equation(
            f"\\left. \\nabla f \\right|_{{{eval_point_str}}} = ",
            inline=True
          )
        ])
      ])
    )

    # For Canvas: Use OnlyHtml to show individual partial derivatives
    for i in range(self.num_variables):
      answer = self.answers[f"partial_derivative_{i}"]
      answers.append(answer)
      body.add_element(
        ca.OnlyHtml([
          ca.Paragraph([
            ca.Equation(
              f"\\left. {self._format_partial_derivative(i)} \\right|_{{{eval_point_str}}} = ",
              inline=True
            ),
            answer
          ])
        ])
      )

    return body, answers

  def get_body(self, **kwargs) -> ca.Section:
    """Build question body (backward compatible interface)."""
    body, _ = self._get_body(**kwargs)
    return body

  def _get_explanation(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question explanation."""
    explanation = ca.Section()

    # Show the function and its gradient
    explanation.add_element(
      ca.Paragraph([
        "To find the gradient, we calculate the partial derivatives of ",
        ca.Equation(sp.latex(self.equation), inline=True),
        ":"
      ])
    )

    # Show analytical gradient
    explanation.add_element(
      ca.Equation(f"\\nabla f = {sp.latex(self.gradient_function)}", inline=False)
    )

    # Show evaluation at the specific point
    explanation.add_element(
      ca.Paragraph([
        f"Evaluating at the point {format_vector(self.evaluation_point)}:"
      ])
    )

    # Show each partial derivative calculation
    subs_map = dict(zip(self.variables, self.evaluation_point))
    for i in range(self.num_variables):
      partial_expr = self.gradient_function[i]
      partial_value = partial_expr.subs(subs_map)

      # Use ca.Answer.accepted_strings for clean numerical formatting
      try:
        numerical_value = float(partial_value)
      except (TypeError, ValueError):
        numerical_value = float(partial_value.evalf())

      # Get clean string representation
      clean_value = sorted(ca.Answer.accepted_strings(numerical_value), key=lambda s: len(s))[0]

      explanation.add_element(
        ca.Paragraph([
          ca.Equation(
            f"{self._format_partial_derivative(i)} = {sp.latex(partial_expr)} = {clean_value}",
            inline=False
          )
        ])
      )

    return explanation, []

  def get_explanation(self, **kwargs) -> ca.Section:
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation(**kwargs)
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

    # For PDF: Create single gradient vector answer
    self._create_gradient_vector_answer()


@QuestionRegistry.register("DerivativeChain")
class DerivativeChain(DerivativeQuestion):
  """Chain rule derivative calculation using function composition."""

  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)

    # Try to generate a valid function/point combination, regenerating if we hit complex numbers
    max_attempts = 10
    for attempt in range(max_attempts):
      try:
        # Generate inner and outer functions for composition
        self._generate_composed_function()

        # Generate evaluation point
        self.evaluation_point = self._generate_evaluation_point()

        # Create answers - this will raise ValueError if we get complex numbers
        self._create_derivative_answers(self.evaluation_point)

        # For PDF: Create single gradient vector answer
        self._create_gradient_vector_answer()

        # If we get here, everything worked
        break

      except ValueError as e:
        if "Complex number encountered" in str(e) and attempt < max_attempts - 1:
          # Advance RNG state by making a dummy call
          _ = self.rng.random()
          continue
        else:
          # If we've exhausted attempts or different error, re-raise
          raise

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

    # Generate outer function - use polynomials, exp, and ln for reliable evaluation
    u = sp.Symbol('u')  # Intermediate variable
    outer_functions = [
      u**2,
      u**3,
      u**4,
      sp.exp(u),
      sp.log(u + 2)  # Add 2 to ensure positive argument for evaluation points
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

  def _get_explanation(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question explanation."""
    explanation = ca.Section()

    # Show the composed function structure
    explanation.add_element(
      ca.Paragraph([
        "This is a composition of functions requiring the chain rule. The function ",
        ca.Equation(sp.latex(self.equation), inline=True),
        " can be written as ",
        ca.Equation(f"f(g(x)) \\text{{ where }} g(x) = {sp.latex(self.inner_function)}", inline=True),
        "."
      ])
    )

    # Explain chain rule with Leibniz notation
    explanation.add_element(
      ca.Paragraph([
        "The chain rule states that for a composite function ",
        ca.Equation("f(g(x))", inline=True),
        ", the derivative with respect to each variable is found by multiplying the derivative of the outer function with respect to the inner function by the derivative of the inner function with respect to the variable:"
      ])
    )

    # Show chain rule formula for each variable
    for i in range(self.num_variables):
      var_name = f"x_{i}"
      explanation.add_element(
        ca.Equation(
          f"\\frac{{\\partial f}}{{\\partial {var_name}}} = \\frac{{\\partial f}}{{\\partial g}} \\cdot \\frac{{\\partial g}}{{\\partial {var_name}}}",
          inline=False
        )
      )

    explanation.add_element(
      ca.Paragraph([
        "Applying this to our specific function:"
      ])
    )

    # Show the specific derivatives step by step
    for i in range(self.num_variables):
      var_name = f"x_{i}"

      # Get outer function derivative with respect to inner function
      outer_deriv = self.outer_function.diff(sp.Symbol('u'))
      inner_deriv = self.inner_function.diff(self.variables[i])

      explanation.add_element(
        ca.Paragraph([
          f"For {var_name}:"
        ])
      )

      explanation.add_element(
        ca.Equation(
          f"\\frac{{\\partial f}}{{\\partial {var_name}}} = \\left({sp.latex(outer_deriv)}\\right) \\cdot \\left({sp.latex(inner_deriv)}\\right)",
          inline=False
        )
      )

    # Show analytical gradient
    explanation.add_element(
      ca.Paragraph([
        "This gives us the complete gradient:"
      ])
    )

    explanation.add_element(
      ca.Equation(f"\\nabla f = {sp.latex(self.gradient_function)}", inline=False)
    )

    # Show evaluation at the specific point
    explanation.add_element(
      ca.Paragraph([
        f"Evaluating at the point {format_vector(self.evaluation_point)}:"
      ])
    )

    # Show each partial derivative calculation
    subs_map = dict(zip(self.variables, self.evaluation_point))
    for i in range(self.num_variables):
      partial_expr = self.gradient_function[i]
      partial_value = partial_expr.subs(subs_map)

      # Use ca.Answer.accepted_strings for clean numerical formatting
      try:
        numerical_value = float(partial_value)
      except (TypeError, ValueError):
        numerical_value = float(partial_value.evalf())

      # Get clean string representation
      clean_value = sorted(ca.Answer.accepted_strings(numerical_value), key=lambda s: len(s))[0]

      explanation.add_element(
        ca.Paragraph([
          ca.Equation(
            f"{self._format_partial_derivative(i)} = {sp.latex(partial_expr)} = {clean_value}",
            inline=False
          )
        ])
      )

    return explanation, []

  def get_explanation(self, **kwargs) -> ca.Section:
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation(**kwargs)
    return explanation
