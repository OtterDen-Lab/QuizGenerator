"""Foundational MATH150 Calculus I questions with worked explanations."""

import math

from QuizGenerator.generation.premade_questions.math_common import make_question


def _limit(rng, **kwargs):
  a, b, x = rng.randint(2, 6), rng.randint(-5, 5), rng.randint(-3, 3)
  value = a * x + b
  return {
    "question": "Evaluate the limit by direct substitution.",
    "equation": f"\\lim_{{x \\to {x}}} ({a}x {b:+})",
    "answer": value, "answer_kind": "int", "label": "Limit",
    "explanation": [f"This is a polynomial, so it is continuous. We may replace x directly with {x}.", f"The limit is {value}."],
    "explanation_equations": [f"{a}({x}) {b:+} = {value}"],
  }


def _continuity(rng, **kwargs):
  point = rng.randint(-4, 4)
  return {
    "question": "Is this rational function continuous at the stated x-value? Enter YES or NO.",
    "equation": f"f(x) = \\frac{{x+1}}{{x-({point})}}, \\qquad x={point}",
    "answer": "NO", "answer_kind": "string", "label": "Continuous?",
    "explanation": [f"A rational function is not continuous where its denominator is zero. At x = {point}, the denominator becomes {point} - ({point}) = 0.", f"Because division by zero is undefined, the function is not continuous at x = {point}."],
  }


def _power_derivative(rng, **kwargs):
  coefficient, exponent = rng.randint(2, 6), rng.randint(2, 5)
  derivative_coefficient = coefficient * exponent
  return {
    "question": "Differentiate the function. Enter the coefficient of the derivative term.",
    "equation": f"f(x) = {coefficient}x^{{{exponent}}}",
    "answer": derivative_coefficient, "answer_kind": "int", "label": "Coefficient",
    "explanation": [f"The power rule says to multiply the coefficient by the exponent, then reduce the exponent by 1.", f"The coefficient of f'(x) is {derivative_coefficient}."],
    "explanation_equations": [f"f'(x) = {coefficient}({exponent})x^{{{exponent - 1}}} = {derivative_coefficient}x^{{{exponent - 1}}}"],
  }


def _trig_derivative(rng, **kwargs):
  coefficient = rng.randint(2, 6)
  return {
    "question": "Differentiate the function. Enter the coefficient multiplying cos(x).",
    "equation": f"f(x) = {coefficient}\\sin(x)",
    "answer": coefficient, "answer_kind": "int", "label": "Coefficient",
    "explanation": ["The derivative of sin(x) is cos(x). The constant in front stays in front when differentiating.", f"Therefore the coefficient of cos(x) is {coefficient}."],
    "explanation_equations": [f"f'(x) = {coefficient}\\cos(x)"],
  }


def _chain_rule(rng, **kwargs):
  outer, inner = rng.randint(2, 5), rng.randint(2, 6)
  return {
    "question": "Use the chain rule. Enter the coefficient of x in the derivative.",
    "equation": f"f(x) = ({inner}x+1)^{{{outer}}}",
    "answer": outer * inner, "answer_kind": "int", "label": "Coefficient",
    "explanation": [f"Differentiate the outside power first, keeping ({inner}x+1) in place. Then multiply by the derivative of the inside, which is {inner}.", f"Multiplying {outer} by {inner} gives {outer * inner}."],
    "explanation_equations": [f"f'(x) = {outer}({inner}x+1)^{{{outer - 1}}}({inner}) = {outer * inner}({inner}x+1)^{{{outer - 1}}}"],
  }


def _critical_point(rng, **kwargs):
  point = rng.randint(-5, 5)
  return {
    "question": "Find the x-coordinate of the critical point.",
    "equation": f"f'(x) = 2(x-({point}))",
    "answer": point, "answer_kind": "int", "label": "Critical x-value",
    "explanation": ["Critical points can occur where the derivative is zero. Set the displayed derivative equal to zero.", f"The derivative is zero when x = {point}, so that is the critical x-value."],
    "explanation_equations": [f"2(x-({point})) = 0 \\quad \\Rightarrow \\quad x = {point}"],
  }


def _implicit(rng, **kwargs):
  x, y = rng.randint(1, 5), rng.randint(1, 5)
  value = -x / y
  return {
    "question": f"For the circle below, find dy/dx at the point ({x}, {y}).",
    "equation": "x^2+y^2=25",
    "answer": value, "label": "dy/dx",
    "explanation": ["Differentiate both sides with respect to x. Remember that y depends on x, so differentiating y squared requires dy/dx.", f"Now substitute x = {x} and y = {y} into the derivative formula."],
    "explanation_equations": [f"2x + 2y\\frac{{dy}}{{dx}} = 0", f"\\frac{{dy}}{{dx}} = -\\frac{{x}}{{y}} = -\\frac{{{x}}}{{{y}}} = {value}"],
  }


def _related_rates(rng, **kwargs):
  radius, rate = rng.randint(2, 8), rng.randint(1, 4)
  value = 2 * math.pi * radius * rate
  return {
    "question": f"A circle's radius is {radius} cm and growing at {rate} cm/s. Find how fast its area is growing.",
    "equation": "A = \\pi r^2",
    "answer": value, "label": "dA/dt",
    "explanation": ["Differentiate the area formula with respect to time. The radius changes with time, so multiply by dr/dt.", f"Use r = {radius} and dr/dt = {rate}."],
    "explanation_equations": [f"\\frac{{dA}}{{dt}} = 2\\pi r\\frac{{dr}}{{dt}} = 2\\pi({radius})({rate}) = {value}"],
  }


def _mvt(rng, **kwargs):
  left, right = sorted(rng.sample(range(-3, 6), 2))
  value = left + right
  return {
    "question": "For f(x) = x squared, find the value c guaranteed by the Mean Value Theorem on the interval shown.",
    "equation": f"[{left}, {right}]",
    "answer": value / 2, "label": "c",
    "explanation": ["The Mean Value Theorem matches the instantaneous slope f'(c) to the average slope over the interval.", f"For x squared, the average slope from {left} to {right} is {value}, and f'(x) = 2x."],
    "explanation_equations": [f"2c = {value} \\quad \\Rightarrow \\quad c = {value / 2}"],
  }


def _definite_integral(rng, **kwargs):
  upper = rng.randint(2, 6)
  value = upper ** 2 / 2
  return {
    "question": "Evaluate the definite integral using an antiderivative.",
    "equation": f"\\int_0^{{{upper}}} x\\,dx",
    "answer": value, "label": "Integral",
    "explanation": ["An antiderivative of x is x squared divided by 2. Evaluate that antiderivative at the upper limit and subtract its value at the lower limit.", f"At x = {upper}, the antiderivative is {upper} squared divided by 2, which is {value}."],
    "explanation_equations": [f"\\left.\\frac{{x^2}}{{2}}\\right|_0^{{{upper}}} = \\frac{{{upper}^2}}{{2}} - 0 = {value}"],
  }


for _name, _builder in {
  "LimitEvaluation": _limit, "ContinuityCheck": _continuity,
  "PowerRuleDerivative": _power_derivative, "TrigDerivative": _trig_derivative,
  "ChainRuleDerivative": _chain_rule, "CriticalPoint": _critical_point,
  "ImplicitDerivative": _implicit, "RelatedRatesCircle": _related_rates,
  "MeanValueTheorem": _mvt, "DefiniteIntegral": _definite_integral,
}.items():
  globals()[_name] = make_question("math150", _name, _builder)
