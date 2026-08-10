"""Foundational MATH151 Calculus II questions with numerical walkthroughs."""

import math

from QuizGenerator.generation.premade_questions.math_common import make_question


def _exp(rng, **kwargs):
  a, x = rng.randint(2, 5), rng.randint(1, 3)
  return {"question": "Evaluate the exponential function.", "equation": f"e^{{{x}}}", "answer": math.exp(x), "label": "Value", "explanation": [f"The exponent tells how many factors of e are represented. Using e approximately 2.71828, e to the {x} power is {math.exp(x):.4f}.", f"So the value is approximately {math.exp(x):.4f}."]}


def _log_derivative(rng, **kwargs):
  coefficient = rng.randint(2, 7)
  return {"question": "Differentiate and enter the numerator of the derivative.", "equation": f"f(x) = {coefficient}\\ln(x)", "answer": coefficient, "answer_kind": "int", "label": "Numerator", "explanation": ["The derivative of ln(x) is 1/x. A constant multiplier stays in front.", f"The derivative is {coefficient}/x, so its numerator is {coefficient}."], "explanation_equations": [f"f'(x) = {coefficient}\\cdot\\frac{{1}}{{x}} = \\frac{{{coefficient}}}{{x}}"]}


def _inverse_trig(rng, **kwargs):
  return {"question": "Evaluate the inverse trigonometric expression in radians.", "equation": "\\arcsin\\left(\\frac{1}{2}\\right)", "answer": math.pi / 6, "label": "Angle", "explanation": ["Arcsine asks for the angle whose sine is the input. The special angle with sine one-half is 30 degrees.", "Convert 30 degrees to radians: 30 degrees is pi divided by 6."], "explanation_equations": [r"\sin\left(\frac{\pi}{6}\right)=\frac{1}{2}"]}


def _parts(rng, **kwargs):
  upper = rng.randint(2, 5)
  value = upper * math.log(upper) - upper + 1
  return {"question": "Evaluate this integral using integration by parts.", "equation": f"\\int_1^{{{upper}}} \\ln(x)\\,dx", "answer": value, "label": "Integral", "explanation": ["For the integral of ln(x), choose u = ln(x) and dv = dx. Then du = 1/x dx and v = x.", f"The antiderivative is x ln(x) - x. Evaluate it at {upper} and at 1."], "explanation_equations": [f"\\left[x\\ln(x)-x\\right]_1^{{{upper}}} = {upper}\\ln({upper})-{upper}+1 = {value:.4f}"]}


def _partial_fraction(rng, **kwargs):
  a, b = rng.randint(1, 4), rng.randint(5, 9)
  return {"question": "Identify the coefficient A in the partial-fraction form.", "equation": f"\\frac{{1}}{{(x+{a})(x+{b})}} = \\frac{{A}}{{x+{a}}} + \\frac{{B}}{{x+{b}}}", "answer": 1 / (b - a), "label": "A", "explanation": [f"Multiply by both denominators. Then set x equal to -{a}; this makes the B term disappear.", f"That leaves 1 = A({b - a}), so A = 1/{b - a}."], "explanation_equations": [f"1 = A(-{a}+{b}) = {b-a}A \\quad \\Rightarrow \\quad A=\\frac{{1}}{{{b-a}}}"]}


def _separable(rng, **kwargs):
  rate, initial, time = rng.randint(1, 3), rng.randint(2, 5), rng.randint(1, 3)
  value = initial * math.exp(rate * time)
  return {"question": f"Solve the separable differential equation at t = {time}.", "equation": f"\\frac{{dy}}{{dt}} = {rate}y, \\qquad y(0)={initial}", "answer": value, "label": "y(t)", "explanation": ["For y' = ky, the solution has the form y(t) = Ce to the kt. The initial value gives C.", f"Here C = {initial}, so substitute t = {time}."], "explanation_equations": [f"y({time}) = {initial}e^{{{rate}({time})}} = {value:.4f}"]}


def _taylor(rng, **kwargs):
  x = rng.choice([0.1, 0.2, 0.3])
  value = 1 + x + x**2 / 2
  return {"question": "Use the second-degree Taylor polynomial for e to the x at x = 0.", "equation": f"x={x}", "answer": value, "label": "Approximation", "explanation": ["The second-degree Taylor polynomial for e to the x is 1 + x + x squared divided by 2. Replace x with the given value.", f"This gives an approximation of {value:.4f}."], "explanation_equations": [f"1+{x}+\\frac{{({x})^2}}{{2}}={value}"]}


def _lhopital(rng, **kwargs):
  point = rng.randint(1, 5)
  return {"question": "Evaluate the indeterminate limit using L'Hopital's rule.", "equation": f"\\lim_{{x\\to {point}}}\\frac{{x^2-{point**2}}}{{x-{point}}}", "answer": 2 * point, "answer_kind": "int", "label": "Limit", "explanation": ["Direct substitution gives 0/0, so differentiate the numerator and denominator once. The derivatives are 2x and 1.", f"Now substitute x = {point} into 2x."], "explanation_equations": [f"\\lim_{{x\\to {point}}}\\frac{{2x}}{{1}} = 2({point}) = {2*point}"]}


def _improper(rng, **kwargs):
  lower = rng.randint(1, 4)
  return {"question": "Evaluate the improper integral.", "equation": f"\\int_{{{lower}}}^{{\\infty}} \\frac{{1}}{{x^2}}\\,dx", "answer": 1 / lower, "label": "Integral", "explanation": ["Replace infinity with a limit. An antiderivative of 1/x squared is -1/x.", f"As the upper bound grows without limit, -1/x approaches 0. Subtracting the lower-bound value gives 1/{lower}."], "explanation_equations": [f"\\lim_{{b\\to\\infty}}\\left[-\\frac{{1}}{{x}}\\right]_{{{lower}}}^b = 0-\\left(-\\frac{{1}}{{{lower}}}\\right) = \\frac{{1}}{{{lower}}}"]}


def _geometric_series(rng, **kwargs):
  ratio = rng.choice([0.2, 0.25, 0.5])
  return {"question": "Find the sum of this infinite geometric series.", "equation": f"1+{ratio}+{ratio}^2+\\cdots", "answer": 1 / (1-ratio), "label": "Sum", "explanation": ["An infinite geometric series with first term a and ratio r has sum a/(1-r) when the absolute value of r is less than 1.", f"Here a = 1 and r = {ratio}, which is less than 1 in absolute value."], "explanation_equations": [f"S=\\frac{{1}}{{1-{ratio}}}={1/(1-ratio)}"]}


for _name, _builder in {"ExponentialEvaluation": _exp, "LogDerivative": _log_derivative, "InverseTrigValue": _inverse_trig, "IntegrationByParts": _parts, "PartialFractions": _partial_fraction, "SeparableDifferentialEquation": _separable, "TaylorApproximation": _taylor, "LHopitalLimit": _lhopital, "ImproperIntegral": _improper, "GeometricSeries": _geometric_series}.items():
  globals()[_name] = make_question("math151", _name, _builder)
