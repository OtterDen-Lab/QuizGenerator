from __future__ import annotations

import abc
import io
import logging

import matplotlib.pyplot as plt
import numpy as np
import sympy
import sympy as sp

import QuizGenerator.generation.contentast as ca
from QuizGenerator.generation.mixins import BodyTemplatesMixin, TableQuestionMixin
from QuizGenerator.generation.question import Question, QuestionRegistry

from .misc import format_vector, generate_function

log = logging.getLogger(__name__)


# Note: This file does not use ca.Answer wrappers - it uses TableQuestionMixin
# which handles answer display through create_answer_table(). The answers are created
# with labels embedded at creation time in _build_context().


class GradientDescentQuestion(Question, abc.ABC):
  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.ML_OPTIMIZATION)
    super().__init__(*args, **kwargs)


@QuestionRegistry.register("GradientDescentWalkthrough")
class GradientDescentWalkthrough(GradientDescentQuestion, TableQuestionMixin, BodyTemplatesMixin):
  DEFAULT_NUM_STEPS = 4
  DEFAULT_NUM_VARIABLES = 2
  DEFAULT_SINGLE_VARIABLE = False
  STEP_DIGITS = 4
  # With the generated quadratics, alpha < 0.5 guarantees a strict decrease
  # away from the optimum even for the largest quadratic coefficient (2).
  LEARNING_RATES = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.num_steps = kwargs.get("num_steps", self.DEFAULT_NUM_STEPS)
    self.num_variables = kwargs.get("num_variables", self.DEFAULT_NUM_VARIABLES)
    self.single_variable = kwargs.get("single_variable", self.DEFAULT_SINGLE_VARIABLE)
    
    if self.single_variable:
      self.num_variables = 1
  
  @classmethod
  def _perform_gradient_descent(
    cls,
    function: sympy.Function,
    gradient_function,
    starting_point,
    num_steps,
    variables,
    learning_rate,
  ) -> list[dict]:
    """
    Perform gradient descent and return step-by-step results.
    """
    results = []
    
    # Each table entry is rounded before it is used in the next row, matching
    # the arithmetic students can reproduce from their submitted work.
    x = [round(float(value), cls.STEP_DIGITS) for value in starting_point]
    
    for step in range(num_steps):
      subs_map = dict(zip(variables, x))
      
      # gradient as floats
      g_syms = gradient_function.subs(subs_map)
      g = [round(float(val), cls.STEP_DIGITS) for val in g_syms]
      
      # function value
      f_val = float(function.subs(subs_map))
      
      update = [round(learning_rate * gi, cls.STEP_DIGITS) for gi in g]
      next_x = [round(xi - ui, cls.STEP_DIGITS) for xi, ui in zip(x, update)]
      
      results.append(
        {
          "step": step + 1,
          "location": x[:],
          "gradient": g[:],
          "update": update[:],
          "function_value": f_val,
          "next_location": next_x[:],
        }
      )

      x = next_x

    return results

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    context.num_steps = kwargs.get("num_steps", cls.DEFAULT_NUM_STEPS)
    if context.num_steps < 1:
      raise ValueError("num_steps must be at least 1")
    context.num_variables = kwargs.get("num_variables", cls.DEFAULT_NUM_VARIABLES)
    context.single_variable = kwargs.get("single_variable", cls.DEFAULT_SINGLE_VARIABLE)
    if context.single_variable:
      context.num_variables = 1
    context.minimize = True

    # Generate function and its properties
    context.use_coupled_quadratic = context.num_variables == 2
    context.variables, context.function, context.gradient_function, context.equation = generate_function(
      context.rng,
      context.num_variables,
      max_degree=2,
      use_quadratic=True,
      use_coupled_quadratic=context.use_coupled_quadratic,
    )

    context.learning_rate = context.rng.choice(cls.LEARNING_RATES)

    context.starting_point = [context.rng.randint(-3, 3) for _ in range(context.num_variables)]

    # Perform gradient descent
    context.gradient_descent_results = cls._perform_gradient_descent(
      context.function,
      context.gradient_function,
      context.starting_point,
      context.num_steps,
      context.variables,
      context.learning_rate,
    )
    context.final_location = context.gradient_descent_results[-1]['next_location']
    context.final_function_value = float(context.function.subs(
      dict(zip(context.variables, context.final_location))
    ))

    # Build answers for each step
    context.step_answers = {}
    for i, result in enumerate(context.gradient_descent_results):
      step = result['step']

      # Location answer
      location_key = f"answer__location_{step}"
      context.step_answers[location_key] = ca.AnswerTypes.Vector(list(result['location']), label=f"Location at step {step}")

      # Gradient answer
      gradient_key = f"answer__gradient_{step}"
      context.step_answers[gradient_key] = ca.AnswerTypes.Vector(list(result['gradient']), label=f"Gradient at step {step}")

      # Update answer
      update_key = f"answer__update_{step}"
      context.step_answers[update_key] = ca.AnswerTypes.Vector(list(result['update']), label=f"Update at step {step}")

    context.step_answers["answer__final_location"] = ca.AnswerTypes.Vector(
      list(context.final_location), label=f"Location after step {context.num_steps}"
    )
    return context

  @classmethod
  def is_interesting_ctx(cls, context) -> bool:
    """Reject zero-gradient starts and any non-decreasing rounded step."""
    results = context.gradient_descent_results
    if not any(abs(value) > 1e-10 for value in results[0]['gradient']):
      return False

    function_values = [result['function_value'] for result in results]
    function_values.append(context.final_function_value)
    return all(
      function_values[index + 1] < function_values[index] - 1e-10
      for index in range(len(function_values) - 1)
    )

  @classmethod
  def _generate_trajectory_plot(cls, context) -> io.BytesIO:
    """Plot the rounded gradient-descent locations used in the solution table."""
    locations = [result['location'] for result in context.gradient_descent_results]
    locations.append(context.final_location)
    function = sp.lambdify(context.variables, context.function, "numpy")

    if context.num_variables == 1:
      x_points = np.array([location[0] for location in locations])
      margin = max(1.0, 0.35 * (x_points.max() - x_points.min()))
      x_values = np.linspace(x_points.min() - margin, x_points.max() + margin, 300)
      y_values = np.asarray(function(x_values), dtype=float)
      point_values = np.asarray(function(x_points), dtype=float)

      fig, ax = plt.subplots(figsize=(7, 3.8))
      ax.plot(x_values, y_values, color="#2673a8", linewidth=2, label="f(x)")
      ax.plot(x_points, point_values, "o", color="#c23b22", markersize=7,
              label="gradient-descent locations")
      for step, (x_value, y_value) in enumerate(zip(x_points, point_values)):
        ax.annotate(f"t={step}", (x_value, y_value), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9)
      ax.set_xlabel(r"$x$")
      ax.set_ylabel(r"$f(x)$")
      ax.set_title("Gradient-descent trajectory")
      ax.grid(alpha=0.25)
      ax.legend(loc="best")
    elif context.num_variables == 2:
      x_points = np.array([location[0] for location in locations])
      y_points = np.array([location[1] for location in locations])

      # Keep the surface focused on the part students actually traversed.
      # Integer endpoints make the close-up readable without hiding any
      # rounded iterate locations.
      def enclosing_integer_bounds(values):
        lower = float(np.floor(values.min()))
        upper = float(np.ceil(values.max()))
        if lower == upper:
          lower -= 0.5
          upper += 0.5
        return lower, upper

      x_lower, x_upper = enclosing_integer_bounds(x_points)
      y_lower, y_upper = enclosing_integer_bounds(y_points)
      x_values = np.linspace(x_lower, x_upper, 90)
      y_values = np.linspace(y_lower, y_upper, 90)
      x_grid, y_grid = np.meshgrid(x_values, y_values)
      z_grid = np.asarray(function(x_grid, y_grid), dtype=float)
      z_points = np.asarray(function(x_points, y_points), dtype=float)
      z_lower = min(0.0, float(np.floor(z_grid.min())))
      z_upper = float(np.ceil(z_grid.max()))
      if z_lower == z_upper:
        z_upper += 1.0
      # A small lift avoids z-fighting. The trajectory is deliberately drawn
      # above the surface so every update remains visible in the overview.
      path_lift = max(1e-6, 1e-5 * (z_grid.max() - z_grid.min()))
      label_lift = max(0.1, 0.04 * (z_grid.max() - z_grid.min()))
      label_dx = 0.09 * (x_values.max() - x_values.min())
      label_dy = 0.09 * (y_values.max() - y_values.min())
      label_offsets = ((0.6, 0.7), (0.7, -1.0), (-1.0, 0.7), (-1.0, -1.0))

      fig = plt.figure(figsize=(7, 4.8))
      ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
      ax.plot_surface(x_grid, y_grid, z_grid, cmap="Blues", alpha=0.8,
                      linewidth=0, antialiased=True, zorder=1)
      ax.plot(x_points, y_points, z_points + path_lift, "o-", color="#c23b22",
              linewidth=2, markersize=5, label="gradient-descent path", zorder=10)
      for step, (x_value, y_value, z_value) in enumerate(zip(x_points, y_points, z_points)):
        offset_x, offset_y = label_offsets[step % len(label_offsets)]
        ax.text(x_value + offset_x * label_dx, y_value + offset_y * label_dy,
                z_value + (2 + step % 2) * label_lift, f"t={step}",
                color="#7a2015", fontsize=8, ha="center", zorder=11)
      ax.set_xlabel(r"$x_0$", labelpad=8)
      ax.set_ylabel(r"$x_1$", labelpad=8)
      ax.set_zlabel(r"$f(x_0, x_1)$", labelpad=8)
      ax.set_xlim(x_lower, x_upper)
      ax.set_ylim(y_lower, y_upper)
      ax.set_zlim(z_lower, z_upper)
      ax.set_title("Gradient-descent trajectory on the loss surface")
      ax.view_init(elev=28, azim=-58)
      ax.legend(loc="upper left")
    else:
      raise ValueError("Trajectory plots support one- and two-variable functions only.")

    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buffer.seek(0)
    return buffer

  @classmethod
  def _generate_coordinate_projection_plot(cls, context) -> io.BytesIO:
    """Plot coordinate-loss projections with loss-surface cross-sections."""
    locations = [result['location'] for result in context.gradient_descent_results]
    locations.append(context.final_location)
    function_values = [result['function_value'] for result in context.gradient_descent_results]
    function_values.append(context.final_function_value)
    function = sp.lambdify(context.variables, context.function, "numpy")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    coordinate_names = (r"$x_0$", r"$x_1$")
    for coordinate_index, (axis, coordinate_name) in enumerate(zip(axes, coordinate_names)):
      coordinate_values = np.array([
        location[coordinate_index] for location in locations
      ])
      coordinate_margin = max(0.1, 0.15 * (coordinate_values.max() - coordinate_values.min()))
      curve_coordinates = np.linspace(
        coordinate_values.min() - coordinate_margin,
        coordinate_values.max() + coordinate_margin,
        200,
      )

      # Each blue curve is a slice of the surface through one iterate. For the
      # x_0 panel x_1 is held fixed, and vice versa, so its red point lies on it.
      for step, location in enumerate(locations):
        if coordinate_index == 0:
          curve_values = function(curve_coordinates, location[1])
        else:
          curve_values = function(location[0], curve_coordinates)
        axis.plot(
          curve_coordinates,
          curve_values,
          color="#2673a8",
          alpha=0.28,
          linewidth=1.25,
          label="loss-surface slices" if step == 0 else None,
        )
      axis.plot(coordinate_values, function_values, "o-", color="#c23b22",
                linewidth=2, markersize=6, label="gradient-descent path")
      for step, (coordinate_value, function_value) in enumerate(
          zip(coordinate_values, function_values)
      ):
        vertical_offset = 8 if step % 2 == 0 else -14
        axis.annotate(f"t={step}", (coordinate_value, function_value),
                      xytext=(0, vertical_offset), textcoords="offset points",
                      ha="center", fontsize=8)
      axis.set_xlabel(coordinate_name)
      axis.set_title(f"Projection onto the {coordinate_name}-loss plane")
      axis.grid(alpha=0.25)
      axis.margins(x=0.08, y=0.14)
      axis.legend(loc="best", fontsize=8)
    axes[0].set_ylabel(r"$f(x_0, x_1)$")

    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buffer.seek(0)
    return buffer
  
  @classmethod
  def _build_body(cls, context) -> tuple[ca.Section, list[ca.Answer]]:
    """Build question body and collect answers."""
    self = context
    body = ca.Section()
    answers = []

    body.add_element(
      ca.Paragraph(
        [
          "Use gradient descent to minimize the function ",
          ca.Equation(sp.latex(self.function), inline=True),
          " with learning rate ",
          ca.Equation(f"\\alpha = {self.learning_rate}", inline=True),
          f" and starting point {self.starting_point[0] if self.num_variables == 1 else tuple(self.starting_point)}. "
          "Round each table entry to four decimal places before using it in the next row."
        ]
      )
    )

    # Create table data - use ca.Equation for proper LaTeX rendering in headers
    headers = [
      "t",
      ca.Equation("x^{(t)}", inline=True),
      ca.Equation("\\nabla f", inline=True),
      ca.Equation("\\alpha \\cdot \\nabla f", inline=True)
    ]
    table_rows = []

    for i, result in enumerate(self.gradient_descent_results):
      step = result['step']
      row = {"t": str(i)}

      if i == 0:

        # Fill in starting location for first row with default formatting
        row[headers[1]] = f"{format_vector(self.starting_point)}"
        row[headers[2]] = self.step_answers[f"answer__gradient_{step}"]  # gradient column
        row[headers[3]] = self.step_answers[f"answer__update_{step}"]  # update column
        # Collect answers for this step (no location answer for step 1)
        answers.append(self.step_answers[f"answer__gradient_{step}"])
        answers.append(self.step_answers[f"answer__update_{step}"])
      else:
        # Subsequent rows - all answer fields
        row[headers[1]] = self.step_answers[f"answer__location_{step}"]
        row[headers[2]] = self.step_answers[f"answer__gradient_{step}"]
        row[headers[3]] = self.step_answers[f"answer__update_{step}"]
        # Collect all answers for this step
        answers.append(self.step_answers[f"answer__location_{step}"])
        answers.append(self.step_answers[f"answer__gradient_{step}"])
        answers.append(self.step_answers[f"answer__update_{step}"])
      table_rows.append(row)

    # The final row makes the destination of the final requested update visible.
    table_rows.append({
      "t": str(self.num_steps),
      headers[1]: self.step_answers["answer__final_location"],
      headers[2]: "—",
      headers[3]: "—",
    })
    answers.append(self.step_answers["answer__final_location"])

    # Create the table using mixin
    gradient_table = cls.create_answer_table(
      headers=headers,
      data_rows=table_rows,
      answer_columns=[headers[1], headers[2], headers[3]]  # Use actual header objects
    )

    body.add_element(gradient_table)

    return body, answers

  @classmethod
  def _build_explanation(cls, context) -> tuple[ca.Section, list[ca.Answer]]:
    """Build question explanation."""
    self = context
    explanation = ca.Section()

    explanation.add_element(
      ca.Paragraph(
        [
          "Gradient descent is an optimization algorithm that iteratively moves towards "
          "the minimum of a function by taking steps proportional to the negative of the gradient."
        ]
      )
    )

    explanation.add_element(
      ca.Paragraph(
        [
          "We want to minimize the function ",
          ca.Equation(sp.latex(self.function), inline=True),
          ". First, we calculate the analytical gradient:"
        ]
      )
    )

    # Add analytical gradient calculation as a display equation (vertical vector)
    explanation.add_element(
      ca.Equation(f"\\nabla f = {sp.latex(self.gradient_function)}", inline=False)
    )

    explanation.add_element(
      ca.Paragraph(
        [
          "Since we want to minimize, we use the update rule: ",
          ca.Equation(r"x^{(t+1)} = x^{(t)} - \alpha \nabla f(x^{(t)})", inline=True),
          f". We start at {tuple(self.starting_point)} with learning rate ",
          ca.Equation(f"\\alpha = {self.learning_rate}", inline=True),
          ". Round each table entry to four decimal places before continuing."
        ]
      )
    )

    # Add completed table showing all solutions
    explanation.add_element(
      ca.Paragraph(
        [
          "**Solution Table:**"
        ]
      )
    )

    # Create filled solution table
    solution_headers = [
      "t",
      ca.Equation("x^{(t)}", inline=True),
      ca.Equation("\\nabla f", inline=True),
      ca.Equation("\\alpha \\cdot \\nabla f", inline=True)
    ]

    solution_rows = []
    for i, result in enumerate(self.gradient_descent_results):
      row = {"t": str(i)}

      row[solution_headers[1]] = f"{format_vector(result['location'])}"
      row[solution_headers[2]] = f"{format_vector(result['gradient'])}"
      row[solution_headers[3]] = f"{format_vector(result['update'])}"

      solution_rows.append(row)

    solution_rows.append({
      "t": str(self.num_steps),
      solution_headers[1]: format_vector(self.final_location),
      solution_headers[2]: "—",
      solution_headers[3]: "—",
    })

    # Create solution table (non-answer table, just display)
    solution_table = self.create_answer_table(
      headers=solution_headers,
      data_rows=solution_rows,
      answer_columns=[]  # No answer columns since this is just for display
    )

    explanation.add_element(solution_table)

    if self.num_variables in (1, 2):
      explanation.add_element(
        ca.Paragraph(
          [
            "The plot marks each rounded location from the table. The path moves "
            "down the function surface from ",
            ca.Equation("t=0", inline=True),
            f" through {self.num_steps} updates."
          ]
        )
      )
      explanation.add_element(
        ca.Picture(
          img_data=cls._generate_trajectory_plot(self),
          caption="Gradient-descent path; labels identify the table step."
        )
      )
      if self.num_variables == 2:
        explanation.add_element(
          ca.Paragraph(
            [
              "These coordinate-loss projections show the same red locations. Each "
              "blue curve is a cross-section of the loss surface through one location."
            ]
          )
        )
        explanation.add_element(
          ca.Picture(
            img_data=cls._generate_coordinate_projection_plot(self),
            caption="Coordinate projections of the gradient-descent path."
          )
        )

    # Step-by-step explanation
    for i, result in enumerate(self.gradient_descent_results):
      step = result['step']

      explanation.add_element(
        ca.Paragraph(
          [
            f"**Step {step}:**"
          ]
        )
      )

      explanation.add_element(
        ca.Paragraph(
          [
            f"Location: {format_vector(result['location'])}"
          ]
        )
      )

      explanation.add_element(
        ca.Paragraph(
          [
            f"Gradient: {format_vector(result['gradient'])}"
          ]
        )
      )

      explanation.add_element(
        ca.Paragraph(
          [
            "Update: ",
            ca.Equation(
              f"\\alpha \\cdot \\nabla f = {self.learning_rate} \\cdot {format_vector(result['gradient'])} = {format_vector(result['update'])}",
              inline=True
            )
          ]
        )
      )

      if step < len(self.gradient_descent_results):
        current_loc = result['location']
        update = result['update']
        next_loc = result['next_location']

        explanation.add_element(
          ca.Paragraph(
            [
              f"Next location: {format_vector(current_loc)} - {format_vector(result['update'])} = {format_vector(next_loc)}"
            ]
          )
        )

    function_values = [r['function_value'] for r in self.gradient_descent_results]
    function_values.append(self.final_function_value)
    explanation.add_element(
      ca.Paragraph(
        [
          f"Function values: {[f'{v:.4f}' for v in function_values]}"
        ]
      )
    )

    return explanation, []
