"""Short, low-college-level precalculus questions for MATH130.

The questions intentionally use small integers and familiar contexts.  They
are intended as a placement/readiness check rather than as difficult exam
problems.
"""

import math
import random

import QuizGenerator.generation.contentast as ca
from QuizGenerator.generation.question import Question, QuestionRegistry


class Math130Question(Question):
  """Base class that gives every MATH130 question the MATH topic."""

  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.MATH)
    super().__init__(*args, **kwargs)


def _section(prompt, answer, explanation):
  prompt_element = prompt if isinstance(prompt, ca.Element) else ca.Paragraph([prompt])
  body = ca.Section([prompt_element, ca.AnswerBlock(answer)])
  return body, ca.Section([ca.Paragraph([explanation])])


def _signed(value):
  """Format an integer with its sign, avoiding expressions such as x - (-3)."""
  return f"+ {value}" if value >= 0 else f"- {abs(value)}"


def _explanation(*elements):
  """Build a worked explanation from paragraphs and equations."""
  return ca.Section(list(elements))


@QuestionRegistry.register()
class FunctionEvaluation(Math130Question):
  """Evaluate a function at a specified input."""

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)
    a, b, x = rng.randint(-5, 5), rng.randint(-5, 5), rng.randint(-3, 4)
    return {"a": a, "b": b, "x": x, "value": a * x + b}

  @classmethod
  def _build_body(cls, context):
    answer = ca.AnswerTypes.Int(context["value"], label="f(x)")
    prompt = ca.Paragraph([
      "Let ", ca.Equation(f"f(x) = {context['a']}x {_signed(context['b'])}", inline=True),
      f". Find f({context['x']})."
    ])
    return _section(
      prompt,
      answer,
      f"Substitute x = {context['x']}: f({context['x']}) = {context['a']}({context['x']}) + ({context['b']}) = {context['value']}.",
    )[0]

  @classmethod
  def _build_explanation(cls, context):
    a, b, x, value = context["a"], context["b"], context["x"], context["value"]
    return _explanation(
      ca.Paragraph([
        "The notation f(", ca.Equation(str(x), inline=True), ") means that we use ",
        ca.Equation(str(x), inline=True), " in place of every ",
        ca.Equation("x", inline=True), " in the rule for the function."
      ]),
      ca.Equation(f"f({x}) = {a}({x}) {_signed(b)} = {value}"),
      ca.Paragraph([f"So the value of the function at x = {x} is {value}."])
    )


@QuestionRegistry.register()
class QuadraticVertex(Math130Question):
  """Find the vertex of a quadratic in vertex-friendly form."""

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)
    h, k = rng.randint(-5, 5), rng.randint(-5, 5)
    coefficient = rng.choice([-1, 1])
    return {"h": h, "k": k, "coefficient": coefficient}

  @classmethod
  def _build_body(cls, context):
    h, k = context["h"], context["k"]
    answer = ca.AnswerTypes.String(f"({h}, {k})", label="Turning point")
    coefficient = "" if context["coefficient"] == 1 else "-"
    prompt = ca.Equation(f"y = {coefficient}(x {_signed(-h)})^2 {_signed(k)}")
    return _section(
      ca.Section([prompt, ca.Paragraph(["Find the turning point (also called the vertex) and enter it as (h, k)."])]),
      answer,
      f"The vertex form has vertex (h, k), so the vertex is ({h}, {k}). The parabola has a {'minimum' if context['coefficient'] == 1 else 'maximum'} there.",
    )[0]

  @classmethod
  def _build_explanation(cls, context):
    kind = "minimum" if context["coefficient"] == 1 else "maximum"
    h, k = context["h"], context["k"]
    return _explanation(
      ca.Paragraph([
        "A quadratic written in vertex form has the pattern ",
        ca.Equation("y = a(x - h)^2 + k", inline=True),
        ". Its turning point is ", ca.Equation("(h, k)", inline=True), "."
      ]),
      ca.Paragraph([
        f"Here, h = {h} and k = {k}, so the turning point is ({h}, {k}). "
        f"Because the squared term has a {'positive' if context['coefficient'] == 1 else 'negative'} coefficient, "
        f"the parabola has a {kind} at that point."
      ])
    )


@QuestionRegistry.register()
class PolynomialFactoring(Math130Question):
  """Find the real zeros of a simple factored polynomial."""

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)
    r1 = rng.randint(-5, 1)
    r2 = rng.randint(r1 + 1, 6)
    return {"r1": r1, "r2": r2}

  @classmethod
  def _build_body(cls, context):
    r1, r2 = context["r1"], context["r2"]
    answer = ca.AnswerTypes.List(False, [str(r1), str(r2)], label="Zeros")
    prompt = ca.Paragraph([
      "Find the real zeros of ",
      ca.Equation(f"p(x) = (x {_signed(-r1)})(x {_signed(-r2)})", inline=True),
      ". Enter them separated by commas."
    ])
    return _section(
      prompt,
      answer,
      f"Set each factor equal to zero: x = {r1} or x = {r2}.",
    )[0]

  @classmethod
  def _build_explanation(cls, context):
    r1, r2 = context["r1"], context["r2"]
    return _explanation(
      ca.Paragraph([
        "A product equals zero when at least one factor equals zero. "
        "Set each factor equal to zero and solve for x."
      ]),
      ca.Equation(f"x {_signed(-r1)} = 0 \\quad \\text{{or}} \\quad x {_signed(-r2)} = 0"),
      ca.Paragraph([f"This gives x = {r1} or x = {r2}, so those are the two real zeros."])
    )


@QuestionRegistry.register()
class RationalDomain(Math130Question):
  """Identify the excluded value in a rational function's domain."""

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)
    excluded = rng.randint(-6, 6)
    return {"excluded": excluded}

  @classmethod
  def _build_body(cls, context):
    excluded = context["excluded"]
    answer = ca.AnswerTypes.Int(excluded, label="Excluded x-value")
    prompt = ca.Paragraph([
      "For ", ca.Equation(f"R(x) = \\frac{{1}}{{x {_signed(-excluded)}}}", inline=True),
      ", which x-value is not in the domain?"
    ])
    return _section(
      prompt,
      answer,
      f"The denominator is zero when x - ({excluded}) = 0, so x = {excluded} is excluded.",
    )[0]

  @classmethod
  def _build_explanation(cls, context):
    excluded = context["excluded"]
    return _explanation(
      ca.Paragraph([
        "A fraction is undefined when its denominator is zero, so we only need to find "
        "the value of x that makes the denominator zero."
      ]),
      ca.Equation(f"x {_signed(-excluded)} = 0 \\quad \\Rightarrow \\quad x = {excluded}"),
      ca.Paragraph([f"Therefore x = {excluded} is not allowed in the domain."])
    )


@QuestionRegistry.register()
class ExponentialGrowth(Math130Question):
  """Evaluate a basic exponential growth model."""

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)
    initial = rng.choice([100, 200, 500])
    rate = rng.choice([0.1, 0.2, 0.5])
    years = rng.randint(1, 3)
    value = initial * (1 + rate) ** years
    return {"initial": initial, "rate": rate, "years": years, "value": value}

  @classmethod
  def _build_body(cls, context):
    answer = ca.AnswerTypes.Float(context["value"], label="Amount", tolerance=0.01)
    pct = int(context["rate"] * 100)
    prompt = ca.Paragraph([
      "A population follows ",
      ca.Equation(f"P(t) = {context['initial']}({1 + context['rate']})^t", inline=True),
      f". Find P({context['years']})."
    ])
    return _section(
      prompt,
      answer,
      f"Substitute t = {context['years']}. The model gives {context['value']:.2f}; the growth rate is {pct}% per time unit.",
    )[0]

  @classmethod
  def _build_explanation(cls, context):
    initial, rate, years, value = (
      context["initial"], context["rate"], context["years"], context["value"]
    )
    growth_factor = 1 + rate
    powered_factor = growth_factor ** years
    return _explanation(
      ca.Paragraph([
        f"The input t tells us how many time periods have passed. Since the question asks for {years} "
        f"time periods, replace t with {years} in the model."
      ]),
      ca.Equation(
        f"P({years}) = {initial}({growth_factor})^{{{years}}} "
        f"= {initial}({powered_factor}) = {value}"
      ),
      ca.Paragraph([f"The population after {years} time periods is {value:.2f}."])
    )


@QuestionRegistry.register()
class LogarithmConversion(Math130Question):
  """Convert an exponential statement to logarithmic form."""

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)
    base = rng.choice([2, 3, 5, 10])
    exponent = rng.randint(1, 4)
    return {"base": base, "exponent": exponent, "power": base ** exponent}

  @classmethod
  def _build_body(cls, context):
    b, e, p = context["base"], context["exponent"], context["power"]
    answer = ca.AnswerTypes.Int(e, label="Logarithm value")
    prompt = ca.Paragraph([
      "Evaluate ", ca.Equation(f"\\log_{{{b}}}({p})", inline=True), "."
    ])
    return _section(
      prompt,
      answer,
      f"Because {b}^{e} = {p}, log_{b}({p}) = {e}.",
    )[0]

  @classmethod
  def _build_explanation(cls, context):
    base, exponent, power = context["base"], context["exponent"], context["power"]
    return _explanation(
      ca.Paragraph([
        "A logarithm asks: what exponent must be placed on the base to make the given number? "
        f"Look for the exponent that makes {base} become {power}."
      ]),
      ca.Equation(f"{base}^{{{exponent}}} = {power}"),
      ca.Paragraph([
        "Because this equation is true, ",
        ca.Equation(f"\\log_{{{base}}}({power}) = {exponent}", inline=True), "."
      ])
    )


@QuestionRegistry.register()
class TrigSpecialAngle(Math130Question):
  """Use a special-angle sine value."""

  ANGLES = [(0, 0.0), (30, 0.5), (90, 1.0), (180, 0.0), (270, -1.0)]

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    angle, value = random.Random(rng_seed).choice(cls.ANGLES)
    return {"angle": angle, "value": value}

  @classmethod
  def _build_body(cls, context):
    answer = ca.AnswerTypes.Float(context["value"], label="sin value")
    prompt = ca.Paragraph([
      "Evaluate ", ca.Equation(f"\\sin({context['angle']}^\\circ)", inline=True), "."
    ])
    return _section(
      prompt,
      answer,
      f"On the unit circle, sin({context['angle']} degrees) = {context['value']}.",
    )[0]

  @classmethod
  def _build_explanation(cls, context):
    angle, value = context["angle"], context["value"]
    return _explanation(
      ca.Paragraph([
        "On the unit circle, sine is the vertical coordinate (the y-coordinate) of the point at the angle. "
        f"At {angle} degrees, that vertical coordinate is {value}."
      ]),
      ca.Equation(f"\\sin({angle}^\\circ) = {value}"),
      ca.Paragraph([f"So the requested sine value is {value}."])
    )


@QuestionRegistry.register()
class TrigEquation(Math130Question):
  """Solve a simple sine equation on 0 to 360 degrees."""

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    angle = random.Random(rng_seed).choice([30, 90, 150, 210, 270, 330])
    value = round(math.sin(math.radians(angle)), 4)
    second = (180 - angle) % 360
    solutions = sorted(set([angle, second]))
    return {"angle": angle, "value": value, "solutions": solutions}

  @classmethod
  def _build_body(cls, context):
    answer = ca.AnswerTypes.List(False, [str(x) for x in context["solutions"]], label="Angles (degrees)")
    prompt = ca.Paragraph([
      "Solve ", ca.Equation(f"\\sin(\\theta) = {context['value']}", inline=True),
      " for 0 <= theta <= 360 degrees. Enter all angles separated by commas."
    ])
    return _section(
      prompt,
      answer,
      f"The sine value occurs at the listed unit-circle angles: {', '.join(map(str, context['solutions']))} degrees.",
    )[0]

  @classmethod
  def _build_explanation(cls, context):
    value = context["value"]
    solutions = context["solutions"]
    solution_text = ", ".join(str(solution) for solution in solutions)
    return _explanation(
      ca.Paragraph([
        "First find the unit-circle angles whose vertical coordinate (sine) is ",
        ca.Equation(str(value), inline=True), ". Then keep only the angles from 0 through 360 degrees."
      ]),
      ca.Equation(
        "\\sin(\\theta) = " + str(value) + " \\quad \\Rightarrow \\quad "
        + ", \\; ".join(f"\\theta = {solution}^\\circ" for solution in solutions)
      ),
      ca.Paragraph([f"Therefore the solutions are {solution_text} degrees."])
    )


@QuestionRegistry.register()
class LinearModelPrediction(Math130Question):
  """Use a linear model to make a small real-world prediction."""

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)
    intercept = rng.randint(5, 20)
    slope = rng.randint(2, 8)
    x = rng.randint(3, 10)
    return {"intercept": intercept, "slope": slope, "x": x, "value": intercept + slope * x}

  @classmethod
  def _build_body(cls, context):
    answer = ca.AnswerTypes.Int(context["value"], label="Prediction")
    prompt = ca.Paragraph([
      "A taxi fare is modeled by ",
      ca.Equation(f"C(m) = {context['intercept']} + {context['slope']}m", inline=True),
      f", where m is miles. Predict the fare for {context['x']} miles."
    ])
    return _section(
      prompt,
      answer,
      f"C({context['x']}) = {context['intercept']} + {context['slope']}({context['x']}) = {context['value']}.",
    )[0]

  @classmethod
  def _build_explanation(cls, context):
    intercept, slope, miles, value = (
      context["intercept"], context["slope"], context["x"], context["value"]
    )
    mileage_cost = slope * miles
    return _explanation(
      ca.Paragraph([
        f"The model uses m for the number of miles traveled. This trip is {miles} miles, "
        f"so replace m with {miles}."
      ]),
      ca.Equation(
        f"C({miles}) = {intercept} + {slope}({miles}) = {intercept} + {mileage_cost} = {value}"
      ),
      ca.Paragraph([
        f"The {intercept} is the starting fare, and {slope} times {miles} is the mileage charge. "
        f"Together, the predicted fare is {value}."
      ])
    )


@QuestionRegistry.register()
class DataSummary(Math130Question):
  """Compute and interpret the mean of a short data set."""

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)
    data = [rng.randint(1, 10) for _ in range(5)]
    return {"data": data, "mean": sum(data) / len(data)}

  @classmethod
  def _build_body(cls, context):
    answer = ca.AnswerTypes.Float(context["mean"], label="Mean", tolerance=0.01)
    values = ", ".join(map(str, context["data"]))
    prompt = ca.Paragraph([f"Find the mean of this data set: {values}."])
    return _section(
      prompt,
      answer,
      f"Add the five values ({sum(context['data'])}) and divide by 5: mean = {context['mean']:.2f}.",
    )[0]

  @classmethod
  def _build_explanation(cls, context):
    data, mean = context["data"], context["mean"]
    total = sum(data)
    values = " + ".join(str(value) for value in data)
    return _explanation(
      ca.Paragraph([
        "The mean is the average. To find it, add every data value and then divide by the number of values."
      ]),
      ca.Equation(f"\\frac{{{values}}}{{{len(data)}}} = \\frac{{{total}}}{{{len(data)}}} = {mean}"),
      ca.Paragraph([f"There are {len(data)} values, so the mean is {mean:.2f}."])
    )
