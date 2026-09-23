
from typing import Any

import sympy as sp

import QuizGenerator.generation.contentast as ca


def generate_function(
  rng,
  num_variables: int,
  max_degree: int,
  use_quadratic: bool = True,
  use_coupled_quadratic: bool = False,
) -> tuple[Any, sp.Expr, sp.MutableDenseMatrix, sp.Equality]:
  """
  Generate a function, its gradient, LaTeX representation, and optimal point using SymPy.

  When ``use_coupled_quadratic`` is enabled for exactly two variables, add a
  small cross term while retaining strict convexity.
  Returns: (variables, function, gradient_function, equation)

  Args:
    rng: Random number generator
    num_variables: Number of variables in the function
    max_degree: Maximum degree of polynomial terms (only used if use_quadratic=False)
    use_quadratic: If True, generates well-conditioned quadratic functions that converge nicely.
                   If False, uses the original random polynomial generation.
    use_coupled_quadratic: For a two-variable quadratic, include a small
                           nonzero cross term so each partial derivative
                           depends on both variables.
  """
  # Create variable symbols
  var_names = [f'x_{i}' for i in range(num_variables)]
  variables = sp.symbols(var_names)  # returns a tuple; robust when n==1

  if use_quadratic:
    # Generate well-conditioned quadratic function: f = sum of (x_i - center_i)^2 terms
    # This creates a paraboloid with a clear minimum at (center_0, center_1, ...)

    # Random center point (small integers for clean calculations)
    centers = [rng.choice([-2, -1, 0, 1, 2]) for _ in range(num_variables)]

    # Random positive coefficients for each squared term (keeps function convex).
    # Coupled walkthroughs use a narrower range so their stable learning-rate
    # bound remains comfortably above the configured maximum of 0.4.
    coefficient_choices = [0.5, 1, 1.5] if use_coupled_quadratic else [0.5, 1, 1.5, 2]
    coeffs = [rng.choice(coefficient_choices) for _ in range(num_variables)]

    # Build quadratic: sum of coeff_i * (x_i - center_i)^2
    poly = sp.Add(*[
      coeffs[i] * (variables[i] - centers[i])**2
      for i in range(num_variables)
    ])

    if use_coupled_quadratic and num_variables == 2:
      # Add a small interaction term.  With a, b >= 0.5 and |d| = 0.5,
      # d^2 < 4ab, so the quadratic remains strictly convex.
      coupling = rng.choice([-0.5, 0.5])
      poly += coupling * (variables[0] - centers[0]) * (variables[1] - centers[1])

  else:
    # Original random polynomial generation (may not converge well)
    # monomials up to max_degree; drop constant 1
    terms = [m for m in sp.polys.itermonomials(variables, max_degree) if m != 1]

    # random nonzero integer coefficients in [-10,-1] ∪ [1,9]
    coeff_pool = [*range(-10, 0), *range(1, 10)]

    # polynomial; if no terms (e.g., max_degree==0), fall back to 0
    poly = sp.Add(*(rng.choice(coeff_pool) * t for t in terms)) if terms else sp.Integer(0)

  # Expand before presenting the function so students can differentiate the
  # polynomial directly rather than first distributing its shifted squares.
  poly = sp.expand(poly)

  # f(x_1, ..., x_n) = poly
  f = sp.Function('f')
  function = poly
  gradient_function = sp.Matrix([poly.diff(v) for v in variables])
  equation = sp.Eq(f(*variables), poly)

  return variables, function, gradient_function, equation
  
  
def format_vector(vec: list[float]) -> str:
  
  vector_string = ', '.join(
    [
      sorted(ca.Answer.accepted_strings(v), key=lambda s: len(s))[0]
      for v in vec
    ]
  )
  
  if len(vec) == 1:
    return vector_string
  else:
    return f"({vector_string})"
