
from typing import List, Tuple, Callable, Union, Any
import sympy as sp


def generate_function(rng, num_variables: int, max_degree: int) -> tuple[Any, sp.Expr, sp.MutableDenseMatrix, sp.Equality]:
  """
  Generate a function, its gradient, LaTeX representation, and optimal point using SymPy.
  Returns: (function, gradient_function, latex_string, optimal_point)
  """
  # Create variable symbols
  
  # variables: tuple even for a single variable
  var_names = [f'x_{i}' for i in range(num_variables)]
  variables = sp.symbols(var_names)  # returns a tuple; robust when n==1
  
  # monomials up to max_degree; drop constant 1
  terms = [m for m in sp.polys.itermonomials(variables, max_degree) if m != 1]
  
  # random nonzero integer coefficients in [-10,-1] ∪ [1,9]
  coeff_pool = [*range(-10, 0), *range(1, 10)]
  
  # polynomial; if no terms (e.g., max_degree==0), fall back to 0
  poly = sp.Add(*(rng.choice(coeff_pool) * t for t in terms)) if terms else sp.Integer(0)
  
  # f(x_1, ..., x_n) = poly
  f = sp.Function('f')
  function = poly
  gradient_function = sp.Matrix([poly.diff(v) for v in variables])
  equation = sp.Eq(f(*variables), poly)
  
  return variables, function, gradient_function, equation