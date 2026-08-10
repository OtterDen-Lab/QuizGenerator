"""MATH270 questions connecting mathematics and computing."""

from QuizGenerator.generation.premade_questions.math_common import make_question


def _linear_system(rng, **kwargs):
  x, y = rng.randint(1, 5), rng.randint(1, 5)
  return {"question": "Solve the system and enter x.", "equation": f"x+y={x+y},\\qquad x-y={x-y}", "answer": x, "answer_kind": "int", "label": "x", "explanation": ["Add the equations so that y and -y cancel. This leaves 2x equal to the sum of the right sides.", f"The sum is {x+y} + ({x-y}) = {2*x}, so x = {x}."], "explanation_equations": [f"2x={2*x}\\quad\\Rightarrow\\quad x={x}"]}


def _matrix_product(rng, **kwargs):
  a, b, c, d = [rng.randint(1, 5) for _ in range(4)]
  return {"question": "Find the top-left entry of the matrix product.", "equation": f"\\begin{{bmatrix}}{a}&{b}\\\\0&1\\end{{bmatrix}}\\begin{{bmatrix}}{c}&0\\\\{d}&1\\end{{bmatrix}}", "answer": a*c+b*d, "answer_kind": "int", "label": "Top-left entry", "explanation": ["The top-left entry is the dot product of the first row of the left matrix and the first column of the right matrix.", f"Multiply {a} by {c}, multiply {b} by {d}, and add the results."], "explanation_equations": [f"{a}({c})+{b}({d})={a*c}+{b*d}={a*c+b*d}"]}


def _vector_dot(rng, **kwargs):
  a, b, c, d = [rng.randint(-4, 5) for _ in range(4)]
  return {"question": "Find the dot product of the vectors.", "equation": f"({a},{b})\\cdot({c},{d})", "answer": a*c+b*d, "answer_kind": "int", "label": "Dot product", "explanation": ["Multiply corresponding components, then add those products.", f"The products are {a} times {c} and {b} times {d}."], "explanation_equations": [f"{a}({c})+{b}({d})={a*c}+{b*d}={a*c+b*d}"]}


def _determinant(rng, **kwargs):
  a, b, c, d = [rng.randint(1, 6) for _ in range(4)]
  return {"question": "Find the determinant of the matrix.", "equation": f"\\begin{{vmatrix}}{a}&{b}\\\\{c}&{d}\\end{{vmatrix}}", "answer": a*d-b*c, "answer_kind": "int", "label": "Determinant", "explanation": ["For a 2 by 2 matrix, multiply the main diagonal and subtract the product of the other diagonal.", f"Compute {a} times {d}, then subtract {b} times {c}."], "explanation_equations": [f"{a}({d})-{b}({c})={a*d}-{b*c}={a*d-b*c}"]}


def _eigenvalue(rng, **kwargs):
  a, d = rng.randint(1, 7), rng.randint(1, 7)
  return {"question": "Enter one eigenvalue of this diagonal matrix.", "equation": f"\\begin{{bmatrix}}{a}&0\\\\0&{d}\\end{{bmatrix}}", "answer": a, "answer_kind": "int", "label": "Eigenvalue", "explanation": ["For a diagonal matrix, the entries on the diagonal are its eigenvalues.", f"The diagonal entries are {a} and {d}, so {a} is one valid eigenvalue."]}


def _trapezoid(rng, **kwargs):
  left, right, fl, fr = 0, rng.randint(2, 6), rng.randint(1, 5), rng.randint(6, 12)
  value = (right-left)*(fl+fr)/2
  return {"question": "Use one trapezoid to approximate the integral from 0 to the stated upper bound.", "equation": f"f(0)={fl},\\quad f({right})={fr}", "answer": value, "label": "Approximation", "explanation": ["The trapezoid rule with one interval is width times the average of the two endpoint heights.", f"The width is {right}, and the endpoint heights are {fl} and {fr}."], "explanation_equations": [f"{right}\\cdot\\frac{{{fl}+{fr}}}{{2}}={value}"]}


def _expected_value(rng, **kwargs):
  return {"question": "A game pays $0 with probability 0.7 and $10 with probability 0.3. Find its expected value.", "equation": "E[X]=\\sum xP(X=x)", "answer": 3, "answer_kind": "int", "label": "Expected value", "explanation": ["Expected value is a weighted average: multiply each possible outcome by its probability, then add.", "The zero-dollar outcome contributes 0, and the ten-dollar outcome contributes 10 times 0.3."], "explanation_equations": ["E[X]=0(0.7)+10(0.3)=3"]}


def _binomial(rng, **kwargs):
  n, p = 4, 0.5
  return {"question": "For four independent fair coin flips, find the probability of exactly two heads.", "equation": "X\\sim\\operatorname{Binomial}(4,0.5)", "answer": 6/16, "label": "Probability", "explanation": ["Choose which two of the four flips are heads, then multiply by the probability of each particular sequence.", "There are 6 choices of two positions, and each four-flip sequence has probability (1/2) to the fourth."], "explanation_equations": ["P(X=2)=\\binom{4}{2}(0.5)^2(0.5)^2=\\frac{6}{16}=0.375"]}


def _random_code(rng, **kwargs):
  return {"question": "What is the expected value of the Python expression below?", "code": "sum([1, 2, 3]) / 3", "answer": 2, "answer_kind": "int", "label": "Expected value", "explanation": ["This code adds the three equally likely values and divides by how many values there are. That is the same calculation as a discrete expected value with equal probabilities.", "The total is 1 + 2 + 3 = 6, and 6 divided by 3 is 2."], "explanation_equations": ["E[X]=\\frac{1+2+3}{3}=2"]}


def _regex(rng, **kwargs):
  return {"question": "Does the string abbb match the regular expression below? Enter YES or NO.", "equation": "ab^*", "answer": "YES", "answer_kind": "string", "label": "Matches?", "explanation": ["The expression starts with one a. The star after b means zero or more b characters may follow.", "The string abbb has one a followed by three b characters, so it matches."]}


for _name, _builder in {"LinearSystem": _linear_system, "MatrixMultiplication": _matrix_product, "VectorDotProduct": _vector_dot, "Determinant": _determinant, "DiagonalEigenvalue": _eigenvalue, "TrapezoidRule": _trapezoid, "ExpectedValue": _expected_value, "BinomialProbability": _binomial, "PythonExpectedValue": _random_code, "RegularExpressionMatch": _regex}.items():
  globals()[_name] = make_question("math270", _name, _builder)
