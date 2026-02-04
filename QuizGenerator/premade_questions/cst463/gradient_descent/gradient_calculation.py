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

  DEFAULT_NUM_VARIABLES = 2
  DEFAULT_MAX_DEGREE = 2

  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.ML_OPTIMIZATION)
    super().__init__(*args, **kwargs)
    self.num_variables = kwargs.get("num_variables", self.DEFAULT_NUM_VARIABLES)
    self.max_degree = kwargs.get("max_degree", self.DEFAULT_MAX_DEGREE)

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    context["num_variables"] = kwargs.get("num_variables", cls.DEFAULT_NUM_VARIABLES)
    context["max_degree"] = kwargs.get("max_degree", cls.DEFAULT_MAX_DEGREE)
    return context

  @classmethod
  def _generate_evaluation_point(cls, context) -> List[float]:
    """Generate a random point for gradient evaluation."""
    return [context.rng.randint(-3, 3) for _ in range(context.num_variables)]

  @staticmethod
  def _format_partial_derivative(var_index: int, num_variables: int) -> str:
    """Format partial derivative symbol for display."""
    if num_variables == 1:
      return "\\frac{df}{dx_0}"
    return f"\\frac{{\\partial f}}{{\\partial x_{var_index}}}"

  @staticmethod
  def _create_derivative_answers(context) -> List[ca.Answer]:
    """Create answer fields for each partial derivative at the evaluation point."""
    answers: List[ca.Answer] = []

    # Evaluate gradient at the specified point
    subs_map = dict(zip(context.variables, context.evaluation_point))

    # Format evaluation point for label
    eval_point_str = ", ".join(
      [f"x_{i} = {context.evaluation_point[i]}" for i in range(context.num_variables)]
    )

    # Create answer for each partial derivative
    for i in range(context.num_variables):
      answer_key = f"partial_derivative_{i}"
      # Evaluate the partial derivative and convert to float
      partial_value = context.gradient_function[i].subs(subs_map)
      try:
        gradient_value = float(partial_value)
      except (TypeError, ValueError):
        # If we get a complex number or other conversion error,
        # this likely means log hit a negative value - regenerate
        raise ValueError("Complex number encountered - need to regenerate")

      # Use auto_float for Canvas compatibility with integers and decimals
      # Label includes the partial derivative notation
      label = f"∂f/∂x_{i} at ({eval_point_str})"
      answers.append(ca.AnswerTypes.Float(gradient_value, label=label))

    return answers

  @staticmethod
  def _create_gradient_vector_answer(context) -> ca.Answer:
    """Create a single gradient vector answer for PDF format."""
    # Format gradient as vector notation
    subs_map = dict(zip(context.variables, context.evaluation_point))
    gradient_values = []

    for i in range(context.num_variables):
      partial_value = context.gradient_function[i].subs(subs_map)
      try:
        gradient_value = float(partial_value)
      except TypeError:
        gradient_value = float(partial_value.evalf())
      gradient_values.append(gradient_value)

    # Format as vector for display using consistent formatting
    vector_str = format_vector(gradient_values)
    return ca.AnswerTypes.String(vector_str, pdf_only=True)

  @classmethod
  def _build_body(cls, context) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    # Display the function
    body.add_element(
      ca.Paragraph([
        "Given the function ",
        ca.Equation(sp.latex(context.equation), inline=True),
        ", calculate the gradient at the point ",
        ca.Equation(format_vector(context.evaluation_point), inline=True),
        "."
      ])
    )

    # Format evaluation point for LaTeX
    eval_point_str = ", ".join(
      [f"x_{i} = {context.evaluation_point[i]}" for i in range(context.num_variables)]
    )

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
    derivative_answers = cls._create_derivative_answers(context)
    for i, answer in enumerate(derivative_answers):
      answers.append(answer)
      body.add_element(
        ca.OnlyHtml([
          ca.Paragraph([
            ca.Equation(
              f"\\left. {cls._format_partial_derivative(i, context.num_variables)} \\right|_{{{eval_point_str}}} = ",
              inline=True
            ),
            answer
          ])
        ])
      )

    return body, answers

  @classmethod
  def _build_explanation(cls, context) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question explanation."""
    explanation = ca.Section()

    # Show the function and its gradient
    explanation.add_element(
      ca.Paragraph([
        "To find the gradient, we calculate the partial derivatives of ",
        ca.Equation(sp.latex(context.equation), inline=True),
        ":"
      ])
    )

    # Show analytical gradient
    explanation.add_element(
      ca.Equation(f"\\nabla f = {sp.latex(context.gradient_function)}", inline=False)
    )

    # Show evaluation at the specific point
    explanation.add_element(
      ca.Paragraph([
        f"Evaluating at the point {format_vector(context.evaluation_point)}:"
      ])
    )

    # Show each partial derivative calculation
    subs_map = dict(zip(context.variables, context.evaluation_point))
    for i in range(context.num_variables):
      partial_expr = context.gradient_function[i]
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
            f"{cls._format_partial_derivative(i, context.num_variables)} = {sp.latex(partial_expr)} = {clean_value}",
            inline=False
          )
        ])
      )

    return explanation, []


@QuestionRegistry.register("DerivativeBasic")
class DerivativeBasic(DerivativeQuestion):
  """Basic derivative calculation using polynomial functions."""

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)

    # Generate a basic polynomial function
    context.variables, context.function, context.gradient_function, context.equation = generate_function(
      context.rng, context.num_variables, context.max_degree
    )

    # Generate evaluation point
    context.evaluation_point = cls._generate_evaluation_point(context)

    # Create answers for evaluation point (used in _build_body)
    cls._create_derivative_answers(context)

    return context


@QuestionRegistry.register("DerivativeChain")
class DerivativeChain(DerivativeQuestion):
  """Chain rule derivative calculation using function composition."""

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)

    # Try to generate a valid function/point combination, regenerating if we hit complex numbers
    max_attempts = 10
    for attempt in range(max_attempts):
      try:
        # Generate inner and outer functions for composition
        cls._generate_composed_function(context)

        # Generate evaluation point
        context.evaluation_point = cls._generate_evaluation_point(context)

        # Create answers - this will raise ValueError if we get complex numbers
        cls._create_derivative_answers(context)

        # If we get here, everything worked
        break

      except ValueError as e:
        if "Complex number encountered" in str(e) and attempt < max_attempts - 1:
          # Advance RNG state by making a dummy call
          _ = context.rng.random()
          continue
        else:
          # If we've exhausted attempts or different error, re-raise
          raise
    return context

  @staticmethod
  def _generate_composed_function(context) -> None:
    """Generate a composed function f(g(x)) for chain rule practice."""
    # Create variable symbols
    var_names = [f'x_{i}' for i in range(context.num_variables)]
    context.variables = sp.symbols(var_names)

    # Generate inner function g(x) - simpler polynomial
    inner_terms = [m for m in sp.polys.itermonomials(context.variables, max(1, context.max_degree-1)) if m != 1]
    coeff_pool = [*range(-5, 0), *range(1, 6)]  # Smaller coefficients for inner function

    if inner_terms:
      inner_poly = sp.Add(*(context.rng.choice(coeff_pool) * t for t in inner_terms))
    else:
      inner_poly = sp.Add(*[context.rng.choice(coeff_pool) * v for v in context.variables])

    # Generate outer function - use polynomials, exp, and ln for reliable evaluation
    u = sp.Symbol('u')  # Intermediate variable
    outer_functions = [
      u**2,
      u**3,
      u**4,
      sp.exp(u),
      sp.log(u + 2)  # Add 2 to ensure positive argument for evaluation points
    ]

    outer_func = context.rng.choice(outer_functions)

    # Compose the functions: f(g(x))
    context.inner_function = inner_poly
    context.outer_function = outer_func
    context.function = outer_func.subs(u, inner_poly)

    # Calculate gradient using chain rule
    context.gradient_function = sp.Matrix([context.function.diff(v) for v in context.variables])

    # Create equation for display
    f = sp.Function('f')
    context.equation = sp.Eq(f(*context.variables), context.function)
    
    return context

  @classmethod
  def _build_explanation(cls, context) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question explanation."""
    explanation = ca.Section()

    # Show the composed function structure
    explanation.add_element(
      ca.Paragraph([
        "This is a composition of functions requiring the chain rule. The function ",
        ca.Equation(sp.latex(context.equation), inline=True),
        " can be written as ",
        ca.Equation(f"f(g(x)) \\text{{ where }} g(x) = {sp.latex(context.inner_function)}", inline=True),
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
    for i in range(context.num_variables):
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
    for i in range(context.num_variables):
      var_name = f"x_{i}"

      # Get outer function derivative with respect to inner function
      outer_deriv = context.outer_function.diff(sp.Symbol('u'))
      inner_deriv = context.inner_function.diff(context.variables[i])

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
      ca.Equation(f"\\nabla f = {sp.latex(context.gradient_function)}", inline=False)
    )

    # Show evaluation at the specific point
    explanation.add_element(
      ca.Paragraph([
        f"Evaluating at the point {format_vector(context.evaluation_point)}:"
      ])
    )

    # Show each partial derivative calculation
    subs_map = dict(zip(context.variables, context.evaluation_point))
    for i in range(context.num_variables):
      partial_expr = context.gradient_function[i]
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
            f"{cls._format_partial_derivative(i, context.num_variables)} = {sp.latex(partial_expr)} = {clean_value}",
            inline=False
          )
        ])
      )

    return explanation, []
