from __future__ import annotations

import abc
import logging
import math
from typing import List, Tuple, Callable, Union
import numpy as np

from QuizGenerator.misc import ContentAST
from QuizGenerator.question import Question, Answer, QuestionRegistry
from QuizGenerator.mixins import TableQuestionMixin, BodyTemplatesMixin

log = logging.getLogger(__name__)


class GradientDescentQuestion(Question, abc.ABC):
    def __init__(self, *args, **kwargs):
        kwargs["topic"] = kwargs.get("topic", Question.Topic.ML_OPTIMIZATION)
        super().__init__(*args, **kwargs)


@QuestionRegistry.register("GradientDescentWalkthrough")
class GradientDescentWalkthrough(GradientDescentQuestion, TableQuestionMixin, BodyTemplatesMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_steps = kwargs.get("num_steps", 4)
        self.num_variables = kwargs.get("num_variables", 2)
        self.single_variable = kwargs.get("single_variable", False)

        if self.single_variable:
            self.num_variables = 1

    def _generate_function(self) -> Tuple[Callable, Callable, str, List[float]]:
        """
        Generate a function, its gradient, LaTeX representation, and optimal point.
        Returns: (function, gradient_function, latex_string, optimal_point)
        """
        if self.num_variables == 1:
            # Single variable: f(x) = a(x - h)^2 + k
            a = self.rng.choice([0.5, 1.0, 1.5, 2.0])
            h = self.rng.uniform(-3.0, 3.0)
            k = self.rng.uniform(0.0, 2.0)

            def f(x):
                if isinstance(x, (list, np.ndarray)):
                    x = x[0]
                return a * (x - h)**2 + k

            def grad_f(x):
                if isinstance(x, (list, np.ndarray)):
                    x = x[0]
                return [2 * a * (x - h)]

            latex = f"f(x) = {a:.1f}(x - {h:.1f})^2 + {k:.1f}"
            optimal = [h]

        else:
            # Two variables: f(x1, x2) = a(x1 - h1)^2 + b(x2 - h2)^2 + k
            a = self.rng.choice([0.5, 1.0, 1.5])
            b = self.rng.choice([0.5, 1.0, 1.5])
            h1 = self.rng.uniform(-2.0, 2.0)
            h2 = self.rng.uniform(-2.0, 2.0)
            k = self.rng.uniform(0.0, 2.0)

            def f(x):
                x1, x2 = x[0], x[1]
                return a * (x1 - h1)**2 + b * (x2 - h2)**2 + k

            def grad_f(x):
                x1, x2 = x[0], x[1]
                return [2 * a * (x1 - h1), 2 * b * (x2 - h2)]

            latex = f"f(x_1, x_2) = {a:.1f}(x_1 - {h1:.1f})^2 + {b:.1f}(x_2 - {h2:.1f})^2 + {k:.1f}"
            optimal = [h1, h2]

        return f, grad_f, latex, optimal

    def _format_vector(self, vec: List[float], decimal_places: int = 4) -> str:
        """Format a vector for display, handling single vs multi-variable cases."""
        if len(vec) == 1:
            return f"{vec[0]:.{decimal_places}f}"
        else:
            formatted_elements = [f"{x:.{decimal_places}f}" for x in vec]
            return f"[{', '.join(formatted_elements)}]"

    def _perform_gradient_descent(self) -> List[dict]:
        """
        Perform gradient descent and return step-by-step results.
        """
        results = []
        current_location = self.starting_point.copy()

        for step in range(self.num_steps):
            # Calculate gradient at current location
            gradient = self.gradient_function(current_location)

            # Calculate update (learning_rate * gradient)
            update = [self.learning_rate * g for g in gradient]

            # Calculate function value at current location
            function_value = self.function(current_location)

            results.append({
                'step': step + 1,
                'location': current_location.copy(),
                'gradient': gradient.copy(),
                'update': update.copy(),
                'function_value': function_value
            })

            # Update location for next iteration
            current_location = [current_location[i] - update[i] for i in range(len(current_location))]

        return results

    def _analyze_convergence(self, results: List[dict]) -> bool:
        """
        Analyze if the gradient descent appears to be converging.
        Look for both decreasing function values and decreasing gradient magnitudes.
        """
        if len(results) < 2:
            return True

        # Check if function values are decreasing
        function_values = [r['function_value'] for r in results]
        function_decreasing = all(function_values[i] >= function_values[i+1] for i in range(len(function_values)-1))

        # Check if gradient magnitudes are decreasing
        gradient_magnitudes = [math.sqrt(sum(g**2 for g in r['gradient'])) for r in results]
        gradient_decreasing = all(gradient_magnitudes[i] >= gradient_magnitudes[i+1] for i in range(len(gradient_magnitudes)-1))

        return function_decreasing and gradient_decreasing

    def refresh(self, rng_seed=None, *args, **kwargs):
        super().refresh(rng_seed=rng_seed, *args, **kwargs)
        log.debug("Refreshing...")

        # Generate function and its properties
        self.function, self.gradient_function, self.function_latex, self.optimal_point = self._generate_function()

        # Generate learning rate (small enough to ensure reasonable behavior)
        self.learning_rate = self.rng.choice([0.01, 0.05, 0.1, 0.15, 0.2])

        # Generate starting point (away from optimal)
        if self.num_variables == 1:
            optimal_x = self.optimal_point[0]
            # Start 2-4 units away from optimal
            offset = self.rng.uniform(2.0, 4.0) * self.rng.choice([-1, 1])
            self.starting_point = [optimal_x + offset]
        else:
            # Start 2-3 units away from optimal in each dimension
            self.starting_point = []
            for i in range(self.num_variables):
                offset = self.rng.uniform(2.0, 3.0) * self.rng.choice([-1, 1])
                self.starting_point.append(self.optimal_point[i] + offset)

        # Perform gradient descent
        self.gradient_descent_results = self._perform_gradient_descent()

        # Analyze convergence
        self.appears_to_converge = self._analyze_convergence(self.gradient_descent_results)

        # Set up answers
        self.answers = {}

        # Answers for each step
        for i, result in enumerate(self.gradient_descent_results):
            step = result['step']

            # Location answer
            location_key = f"answer__location_{step}"
            if self.num_variables == 1:
                self.answers[location_key] = Answer.auto_float(location_key, result['location'][0])
            else:
                # For multi-variable, we'll need separate answers for each component
                for j in range(self.num_variables):
                    comp_key = f"answer__location_{step}_x{j+1}"
                    self.answers[comp_key] = Answer.auto_float(comp_key, result['location'][j])

            # Gradient answer
            gradient_key = f"answer__gradient_{step}"
            if self.num_variables == 1:
                self.answers[gradient_key] = Answer.auto_float(gradient_key, result['gradient'][0])
            else:
                for j in range(self.num_variables):
                    comp_key = f"answer__gradient_{step}_x{j+1}"
                    self.answers[comp_key] = Answer.auto_float(comp_key, result['gradient'][j])

            # Update answer
            update_key = f"answer__update_{step}"
            if self.num_variables == 1:
                self.answers[update_key] = Answer.auto_float(update_key, result['update'][0])
            else:
                for j in range(self.num_variables):
                    comp_key = f"answer__update_{step}_x{j+1}"
                    self.answers[comp_key] = Answer.auto_float(comp_key, result['update'][j])

        # Convergence answer
        self.answers["answer__convergence"] = Answer.string(
            "answer__convergence",
            "Yes" if self.appears_to_converge else "No"
        )

    def get_body(self, **kwargs) -> ContentAST.Section:
        body = ContentAST.Section()

        # Introduction
        body.add_element(
            ContentAST.Paragraph([
                "Given the function ",
                ContentAST.Equation(self.function_latex, inline=True),
                ", learning rate ",
                ContentAST.Equation(f"\\alpha = {self.learning_rate}", inline=True),
                f", and starting point {self._format_vector(self.starting_point)}, "
                "perform gradient descent for the specified number of steps. "
                "Fill in the table below with your calculations."
            ])
        )

        # Create table data - use ContentAST.Equation for proper LaTeX rendering in headers
        headers = [
            "n",
            "location",
            ContentAST.Equation("\\nabla f", inline=True),
            ContentAST.Equation("\\alpha \\cdot \\nabla f", inline=True)
        ]
        log.debug("Here!")
        table_rows = []

        for i in range(self.num_steps):
            step = i + 1
            row = {"n": str(step)}

            if self.num_variables == 1:
                # Single variable - simple answer fields
                row["location"] = f"answer__location_{step}"
                row[headers[2]] = f"answer__gradient_{step}"  # gradient column
                row[headers[3]] = f"answer__update_{step}"   # update column
            else:
                # Multi-variable - need component-wise answers
                location_parts = []
                gradient_parts = []
                update_parts = []

                for j in range(self.num_variables):
                    location_parts.append(f"answer__location_{step}_x{j+1}")
                    gradient_parts.append(f"answer__gradient_{step}_x{j+1}")
                    update_parts.append(f"answer__update_{step}_x{j+1}")

                row["location"] = f"[{', '.join(location_parts)}]"
                row[headers[2]] = f"[{', '.join(gradient_parts)}]"  # gradient column
                row[headers[3]] = f"[{', '.join(update_parts)}]"   # update column

            table_rows.append(row)

        # Create the table using mixin
        gradient_table = self.create_answer_table(
            headers=headers,
            data_rows=table_rows,
            answer_columns=["location", headers[2], headers[3]]  # Use actual header objects
        )

        body.add_element(gradient_table)

        # Convergence question
        body.add_element(
            ContentAST.AnswerBlock(
                ContentAST.Answer(
                    answer=self.answers["answer__convergence"],
                    label="Does this gradient descent appear to converge? (Yes/No)"
                )
            )
        )

        return body

    def get_explanation(self, **kwargs) -> ContentAST.Section:
        explanation = ContentAST.Section()

        explanation.add_element(
            ContentAST.Paragraph([
                "Gradient descent is an optimization algorithm that iteratively moves towards "
                "the minimum of a function by taking steps proportional to the negative of the gradient."
            ])
        )

        explanation.add_element(
            ContentAST.Paragraph([
                "For the function ",
                ContentAST.Equation(self.function_latex, inline=True),
                f", we start at {self._format_vector(self.starting_point)} with learning rate ",
                ContentAST.Equation(f"\\alpha = {self.learning_rate}", inline=True),
                "."
            ])
        )

        # Step-by-step explanation
        for i, result in enumerate(self.gradient_descent_results):
            step = result['step']

            explanation.add_element(
                ContentAST.Paragraph([
                    f"**Step {step}:**"
                ])
            )

            explanation.add_element(
                ContentAST.Paragraph([
                    f"Location: {self._format_vector(result['location'])}"
                ])
            )

            explanation.add_element(
                ContentAST.Paragraph([
                    f"Gradient: {self._format_vector(result['gradient'])}"
                ])
            )

            explanation.add_element(
                ContentAST.Paragraph([
                    "Update: ",
                    ContentAST.Equation(f"\\alpha \\times \\nabla f = {self.learning_rate} \\times {self._format_vector(result['gradient'])} = {self._format_vector(result['update'])}", inline=True)
                ])
            )

            if step < len(self.gradient_descent_results):
                # Calculate next location for display
                current_loc = result['location']
                update = result['update']
                next_loc = [current_loc[j] - update[j] for j in range(len(current_loc))]

                explanation.add_element(
                    ContentAST.Paragraph([
                        f"Next location: {self._format_vector(current_loc)} - {self._format_vector(result['update'])} = {self._format_vector(next_loc)}"
                    ])
                )

        # Convergence analysis
        explanation.add_element(
            ContentAST.Paragraph([
                f"**Convergence Analysis:** The algorithm {'appears to converge' if self.appears_to_converge else 'does not appear to converge'} "
                f"based on the trend in function values and gradient magnitudes."
            ])
        )

        function_values = [r['function_value'] for r in self.gradient_descent_results]
        explanation.add_element(
            ContentAST.Paragraph([
                f"Function values: {[f'{v:.4f}' for v in function_values]}"
            ])
        )

        return explanation