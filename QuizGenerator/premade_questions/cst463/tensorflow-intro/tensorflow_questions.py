from __future__ import annotations

import abc
import io
import logging
import re
import numpy as np
import sympy as sp
from typing import List, Tuple, Dict, Any

from QuizGenerator.contentast import ContentAST
from QuizGenerator.question import Question, Answer, QuestionRegistry
from QuizGenerator.mixins import TableQuestionMixin, BodyTemplatesMixin

# Import gradient descent utilities
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gradient_descent'))
from misc import generate_function, format_vector

log = logging.getLogger(__name__)


@QuestionRegistry.register()
class ParameterCountingQuestion(Question):
  """
  Question asking students to count parameters in a neural network.

  Given a dense network architecture, students calculate:
  - Total number of weights
  - Total number of biases
  - Total trainable parameters
  """

  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.ML_OPTIMIZATION)
    super().__init__(*args, **kwargs)

    self.num_layers = kwargs.get("num_layers", None)
    self.include_biases = kwargs.get("include_biases", True)

  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)

    # Generate random architecture
    if self.num_layers is None:
      self.num_layers = self.rng.choice([3, 4])

    # Generate layer sizes
    # Input layer: common sizes for typical problems
    input_sizes = [28*28, 32*32, 784, 1024, 64, 128]
    self.layer_sizes = [self.rng.choice(input_sizes)]

    # Hidden layers: reasonable sizes
    for i in range(self.num_layers - 2):
      hidden_size = self.rng.choice([32, 64, 128, 256, 512])
      self.layer_sizes.append(hidden_size)

    # Output layer: typical classification sizes
    output_size = self.rng.choice([2, 10, 100, 1000])
    self.layer_sizes.append(output_size)

    # Calculate correct answers
    self.total_weights = 0
    self.total_biases = 0
    self.weights_per_layer = []
    self.biases_per_layer = []

    for i in range(len(self.layer_sizes) - 1):
      weights = self.layer_sizes[i] * self.layer_sizes[i+1]
      biases = self.layer_sizes[i+1] if self.include_biases else 0

      self.weights_per_layer.append(weights)
      self.biases_per_layer.append(biases)

      self.total_weights += weights
      self.total_biases += biases

    self.total_params = self.total_weights + self.total_biases

    # Create answers
    self._create_answers()

  def _create_answers(self):
    """Create answer fields."""
    self.answers = {}

    self.answers["total_weights"] = Answer.integer("total_weights", self.total_weights)

    if self.include_biases:
      self.answers["total_biases"] = Answer.integer("total_biases", self.total_biases)
      self.answers["total_params"] = Answer.integer("total_params", self.total_params)
    else:
      self.answers["total_params"] = Answer.integer("total_params", self.total_params)

  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()

    # Question description
    body.add_element(ContentAST.Paragraph([
      "Consider a fully-connected (dense) neural network with the following architecture:"
    ]))

    # Display architecture
    arch_parts = []
    for i, size in enumerate(self.layer_sizes):
      if i > 0:
        arch_parts.append(" → ")
      arch_parts.append(str(size))

    body.add_element(ContentAST.Paragraph([
      "Architecture: " + "".join(arch_parts)
    ]))

    if self.include_biases:
      body.add_element(ContentAST.Paragraph([
        "Each layer includes bias terms."
      ]))

    # Questions
    body.add_element(ContentAST.Paragraph([
      "Calculate the following:"
    ]))

    # Answer table
    table_data = []
    table_data.append(["Parameter Type", "Count"])

    table_data.append([
      "Total weights (connections between layers)",
      ContentAST.Answer(self.answers["total_weights"])
    ])

    if self.include_biases:
      table_data.append([
        "Total biases",
        ContentAST.Answer(self.answers["total_biases"])
      ])

    table_data.append([
      "Total trainable parameters",
      ContentAST.Answer(self.answers["total_params"])
    ])

    body.add_element(ContentAST.Table(data=table_data))

    return body

  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph([
      "To count parameters in a dense neural network, we calculate weights and biases for each layer."
    ]))

    explanation.add_element(ContentAST.Paragraph([
      "**Weights calculation:**"
    ]))

    for i in range(len(self.layer_sizes) - 1):
      input_size = self.layer_sizes[i]
      output_size = self.layer_sizes[i+1]
      weights = self.weights_per_layer[i]

      explanation.add_element(ContentAST.Paragraph([
        f"Layer {i+1} → {i+2}: ",
        ContentAST.Equation(f"{input_size} \\times {output_size} = {weights:,}", inline=True),
        " weights"
      ]))

    explanation.add_element(ContentAST.Paragraph([
      "Total weights: ",
      ContentAST.Equation(
        f"{' + '.join([f'{w:,}' for w in self.weights_per_layer])} = {self.total_weights:,}",
        inline=True
      )
    ]))

    if self.include_biases:
      explanation.add_element(ContentAST.Paragraph([
        "**Biases calculation:**"
      ]))

      for i in range(len(self.layer_sizes) - 1):
        output_size = self.layer_sizes[i+1]
        biases = self.biases_per_layer[i]

        explanation.add_element(ContentAST.Paragraph([
          f"Layer {i+2}: {biases:,} biases (one per neuron)"
        ]))

      explanation.add_element(ContentAST.Paragraph([
        "Total biases: ",
        ContentAST.Equation(
          f"{' + '.join([f'{b:,}' for b in self.biases_per_layer])} = {self.total_biases:,}",
          inline=True
        )
      ]))

    explanation.add_element(ContentAST.Paragraph([
      "**Total trainable parameters:**"
    ]))

    if self.include_biases:
      explanation.add_element(ContentAST.Equation(
        f"\\text{{Total}} = {self.total_weights:,} + {self.total_biases:,} = {self.total_params:,}",
        inline=False
      ))
    else:
      explanation.add_element(ContentAST.Equation(
        f"\\text{{Total}} = {self.total_weights:,}",
        inline=False
      ))

    return explanation


@QuestionRegistry.register()
class ActivationFunctionComputationQuestion(Question):
  """
  Question asking students to compute activation function outputs.

  Given a vector of inputs and an activation function, students calculate
  the output for each element (or entire vector for softmax).
  """

  ACTIVATION_RELU = "relu"
  ACTIVATION_SIGMOID = "sigmoid"
  ACTIVATION_TANH = "tanh"
  ACTIVATION_SOFTMAX = "softmax"
  ACTIVATION_LEAKY_RELU = "leaky_relu"

  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.ML_OPTIMIZATION)
    super().__init__(*args, **kwargs)

    self.vector_size = kwargs.get("vector_size", None)
    self.activation = kwargs.get("activation", None)

  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)

    # Generate random input vector
    if self.vector_size is None:
      self.vector_size = self.rng.choice([3, 4, 5])

    self.input_vector = [
      round(self.rng.uniform(-3, 3), 1)
      for _ in range(self.vector_size)
    ]

    # Select activation function
    if self.activation is None:
      activations = [
        self.ACTIVATION_RELU,
        self.ACTIVATION_SIGMOID,
        self.ACTIVATION_TANH,
        self.ACTIVATION_SOFTMAX,
        self.ACTIVATION_LEAKY_RELU
      ]
      self.activation = self.rng.choice(activations)

    # For leaky ReLU, set alpha
    self.leaky_alpha = 0.01

    # Compute outputs
    self.output_vector = self._compute_activation(self.input_vector)

    # Create answers
    self._create_answers()

  def _compute_activation(self, inputs):
    """Compute activation function output."""
    if self.activation == self.ACTIVATION_RELU:
      return [max(0, x) for x in inputs]

    elif self.activation == self.ACTIVATION_SIGMOID:
      return [1 / (1 + np.exp(-x)) for x in inputs]

    elif self.activation == self.ACTIVATION_TANH:
      return [np.tanh(x) for x in inputs]

    elif self.activation == self.ACTIVATION_SOFTMAX:
      # Subtract max for numerical stability
      exp_vals = [np.exp(x - max(inputs)) for x in inputs]
      sum_exp = sum(exp_vals)
      return [e / sum_exp for e in exp_vals]

    elif self.activation == self.ACTIVATION_LEAKY_RELU:
      return [x if x > 0 else self.leaky_alpha * x for x in inputs]

    else:
      raise ValueError(f"Unknown activation: {self.activation}")

  def _get_activation_name(self):
    """Get human-readable activation name."""
    names = {
      self.ACTIVATION_RELU: "ReLU",
      self.ACTIVATION_SIGMOID: "Sigmoid",
      self.ACTIVATION_TANH: "Tanh",
      self.ACTIVATION_SOFTMAX: "Softmax",
      self.ACTIVATION_LEAKY_RELU: "Leaky ReLU"
    }
    return names.get(self.activation, "Unknown")

  def _get_activation_formula(self):
    """Get LaTeX formula for activation function."""
    if self.activation == self.ACTIVATION_RELU:
      return r"\text{ReLU}(x) = \max(0, x)"

    elif self.activation == self.ACTIVATION_SIGMOID:
      return r"\sigma(x) = \frac{1}{1 + e^{-x}}"

    elif self.activation == self.ACTIVATION_TANH:
      return r"\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}"

    elif self.activation == self.ACTIVATION_SOFTMAX:
      return r"\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}"

    elif self.activation == self.ACTIVATION_LEAKY_RELU:
      return f"\\text{{LeakyReLU}}(x) = \\begin{{cases}} x & x > 0 \\\\ {self.leaky_alpha}x & x \\leq 0 \\end{{cases}}"

    return ""

  def _create_answers(self):
    """Create answer fields."""
    self.answers = {}

    if self.activation == self.ACTIVATION_SOFTMAX:
      # Softmax: single vector answer
      self.answers["output"] = Answer.vector_value("output", self.output_vector)
    else:
      # Element-wise: individual answers
      for i, output in enumerate(self.output_vector):
        key = f"output_{i}"
        self.answers[key] = Answer.float_value(key, float(output))

  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()

    # Question description
    body.add_element(ContentAST.Paragraph([
      f"Given the input vector below, compute the output after applying the {self._get_activation_name()} activation function."
    ]))

    # Display formula
    body.add_element(ContentAST.Paragraph([
      "Activation function: ",
      ContentAST.Equation(self._get_activation_formula(), inline=True)
    ]))

    # Input vector
    input_str = ", ".join([f"{x:.1f}" for x in self.input_vector])
    body.add_element(ContentAST.Paragraph([
      "Input: ",
      ContentAST.Equation(f"[{input_str}]", inline=True)
    ]))

    # Answer table
    if self.activation == self.ACTIVATION_SOFTMAX:
      body.add_element(ContentAST.Paragraph([
        "Compute the output vector:"
      ]))

      table_data = []
      table_data.append(["Output Vector"])
      table_data.append([ContentAST.Answer(self.answers["output"])])

      body.add_element(ContentAST.Table(data=table_data))

    else:
      body.add_element(ContentAST.Paragraph([
        "Compute the output for each element:"
      ]))

      table_data = []
      table_data.append(["Input", "Output"])

      for i, x in enumerate(self.input_vector):
        table_data.append([
          ContentAST.Equation(f"{x:.1f}", inline=True),
          ContentAST.Answer(self.answers[f"output_{i}"])
        ])

      body.add_element(ContentAST.Table(data=table_data))

    return body

  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph([
      f"To compute the {self._get_activation_name()} activation, we apply the formula to each input."
    ]))

    if self.activation == self.ACTIVATION_SOFTMAX:
      explanation.add_element(ContentAST.Paragraph([
        "**Softmax computation:**"
      ]))

      # Show exponentials
      exp_strs = [f"e^{{{x:.1f}}}" for x in self.input_vector]
      explanation.add_element(ContentAST.Paragraph([
        "First, compute exponentials: ",
        ContentAST.Equation(", ".join(exp_strs), inline=True)
      ]))

      # Numerical values
      exp_vals = [np.exp(x) for x in self.input_vector]
      exp_vals_str = ", ".join([f"{e:.4f}" for e in exp_vals])
      explanation.add_element(ContentAST.Paragraph([
        ContentAST.Equation(f"\\approx [{exp_vals_str}]", inline=True)
      ]))

      # Sum
      sum_exp = sum(exp_vals)
      explanation.add_element(ContentAST.Paragraph([
        "Sum: ",
        ContentAST.Equation(f"{sum_exp:.4f}", inline=True)
      ]))

      # Final outputs
      explanation.add_element(ContentAST.Paragraph([
        "Divide each by the sum:"
      ]))

      for i, (exp_val, output) in enumerate(zip(exp_vals, self.output_vector)):
        explanation.add_element(ContentAST.Equation(
          f"\\text{{softmax}}({self.input_vector[i]:.1f}) = \\frac{{{exp_val:.4f}}}{{{sum_exp:.4f}}} = {output:.4f}",
          inline=False
        ))

    else:
      explanation.add_element(ContentAST.Paragraph([
        "**Element-wise computation:**"
      ]))

      for i, (x, y) in enumerate(zip(self.input_vector, self.output_vector)):
        if self.activation == self.ACTIVATION_RELU:
          explanation.add_element(ContentAST.Equation(
            f"\\text{{ReLU}}({x:.1f}) = \\max(0, {x:.1f}) = {y:.4f}",
            inline=False
          ))

        elif self.activation == self.ACTIVATION_SIGMOID:
          explanation.add_element(ContentAST.Equation(
            f"\\sigma({x:.1f}) = \\frac{{1}}{{1 + e^{{-{x:.1f}}}}} = {y:.4f}",
            inline=False
          ))

        elif self.activation == self.ACTIVATION_TANH:
          explanation.add_element(ContentAST.Equation(
            f"\\tanh({x:.1f}) = {y:.4f}",
            inline=False
          ))

        elif self.activation == self.ACTIVATION_LEAKY_RELU:
          if x > 0:
            explanation.add_element(ContentAST.Equation(
              f"\\text{{LeakyReLU}}({x:.1f}) = {x:.1f} = {y:.4f}",
              inline=False
            ))
          else:
            explanation.add_element(ContentAST.Equation(
              f"\\text{{LeakyReLU}}({x:.1f}) = {self.leaky_alpha} \\times {x:.1f} = {y:.4f}",
              inline=False
            ))

    return explanation


@QuestionRegistry.register()
class RegularizationCalculationQuestion(Question):
  """
  Question asking students to calculate loss with L2 regularization.

  Given a small network (2-4 weights), students calculate:
  - Forward pass
  - Base MSE loss
  - L2 regularization penalty
  - Total loss
  - Gradient with regularization for one weight
  """

  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.ML_OPTIMIZATION)
    super().__init__(*args, **kwargs)

    self.num_weights = kwargs.get("num_weights", None)

  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)

    # Generate small network (2-4 weights for simplicity)
    if self.num_weights is None:
      self.num_weights = self.rng.choice([2, 3, 4])

    # Generate weights (small values)
    self.weights = [
      round(self.rng.uniform(-2, 2), 1)
      for _ in range(self.num_weights)
    ]

    # Generate input and target
    self.input_val = round(self.rng.uniform(-3, 3), 1)
    self.target = round(self.rng.uniform(-5, 5), 1)

    # Regularization coefficient
    self.lambda_reg = self.rng.choice([0.01, 0.05, 0.1, 0.5])

    # Forward pass (simple linear combination for simplicity)
    # prediction = sum(w_i * input^i) for i in 0..n
    # This gives us a polynomial: w0 + w1*x + w2*x^2 + ...
    self.prediction = sum(
      w * (self.input_val ** i)
      for i, w in enumerate(self.weights)
    )

    # Calculate losses
    self.base_loss = 0.5 * (self.target - self.prediction) ** 2
    self.l2_penalty = (self.lambda_reg / 2) * sum(w**2 for w in self.weights)
    self.total_loss = self.base_loss + self.l2_penalty

    # Calculate gradient for first weight (w0, the bias term)
    # dL_base/dw0 = -(target - prediction) * dPrediction/dw0
    # dPrediction/dw0 = input^0 = 1
    # dL_reg/dw0 = lambda * w0
    # dL_total/dw0 = dL_base/dw0 + dL_reg/dw0

    self.grad_base_w0 = -(self.target - self.prediction) * 1  # derivative of w0*x^0
    self.grad_reg_w0 = self.lambda_reg * self.weights[0]
    self.grad_total_w0 = self.grad_base_w0 + self.grad_reg_w0

    # Create answers
    self._create_answers()

  def _create_answers(self):
    """Create answer fields."""
    self.answers = {}

    self.answers["prediction"] = Answer.float_value("prediction", float(self.prediction))
    self.answers["base_loss"] = Answer.float_value("base_loss", float(self.base_loss))
    self.answers["l2_penalty"] = Answer.float_value("l2_penalty", float(self.l2_penalty))
    self.answers["total_loss"] = Answer.float_value("total_loss", float(self.total_loss))
    self.answers["grad_total_w0"] = Answer.auto_float("grad_total_w0", float(self.grad_total_w0))

  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()

    # Question description
    body.add_element(ContentAST.Paragraph([
      "Consider a simple model with the following parameters:"
    ]))

    # Display weights
    weight_strs = [f"w_{i} = {w:.1f}" for i, w in enumerate(self.weights)]
    body.add_element(ContentAST.Paragraph([
      "Weights: ",
      ContentAST.Equation(", ".join(weight_strs), inline=True)
    ]))

    # Model equation
    terms = []
    for i, w in enumerate(self.weights):
      if i == 0:
        terms.append(f"w_0")
      elif i == 1:
        terms.append(f"w_1 x")
      else:
        terms.append(f"w_{i} x^{i}")

    model_eq = " + ".join(terms)
    body.add_element(ContentAST.Paragraph([
      "Model: ",
      ContentAST.Equation(f"\\hat{{y}} = {model_eq}", inline=True)
    ]))

    # Data point
    body.add_element(ContentAST.Paragraph([
      "Data point: ",
      ContentAST.Equation(f"x = {self.input_val:.1f}, y = {self.target:.1f}", inline=True)
    ]))

    # Regularization
    body.add_element(ContentAST.Paragraph([
      "L2 regularization coefficient: ",
      ContentAST.Equation(f"\\lambda = {self.lambda_reg}", inline=True)
    ]))

    body.add_element(ContentAST.Paragraph([
      "Calculate the following:"
    ]))

    # Answer table
    table_data = []
    table_data.append(["Calculation", "Value"])

    table_data.append([
      ContentAST.Paragraph(["Prediction ", ContentAST.Equation(r"\hat{y}", inline=True)]),
      ContentAST.Answer(self.answers["prediction"])
    ])

    table_data.append([
      ContentAST.Paragraph(["Base MSE loss: ", ContentAST.Equation(r"L_{base} = (1/2)(y - \hat{y})^2", inline=True)]),
      ContentAST.Answer(self.answers["base_loss"])
    ])

    table_data.append([
      ContentAST.Paragraph(["L2 penalty: ", ContentAST.Equation(r"L_{reg} = (\lambda/2)\sum w_i^2", inline=True)]),
      ContentAST.Answer(self.answers["l2_penalty"])
    ])

    table_data.append([
      ContentAST.Paragraph(["Total loss: ", ContentAST.Equation(r"L_{total} = L_{base} + L_{reg}", inline=True)]),
      ContentAST.Answer(self.answers["total_loss"])
    ])

    table_data.append([
      ContentAST.Paragraph(["Gradient: ", ContentAST.Equation(r"\frac{\partial L_{total}}{\partial w_0}", inline=True)]),
      ContentAST.Answer(self.answers["grad_total_w0"])
    ])

    body.add_element(ContentAST.Table(data=table_data))

    return body

  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph([
      "L2 regularization adds a penalty term to the loss function to prevent overfitting by keeping weights small."
    ]))

    # Step 1: Forward pass
    explanation.add_element(ContentAST.Paragraph([
      "**Step 1: Compute prediction**"
    ]))

    terms = []
    for i, w in enumerate(self.weights):
      if i == 0:
        terms.append(f"{w:.1f}")
      else:
        x_term = f"{self.input_val:.1f}^{i}" if i > 1 else f"{self.input_val:.1f}"
        terms.append(f"{w:.1f} \\times {x_term}")

    explanation.add_element(ContentAST.Equation(
      f"\\hat{{y}} = {' + '.join(terms)} = {self.prediction:.4f}",
      inline=False
    ))

    # Step 2: Base loss
    explanation.add_element(ContentAST.Paragraph([
      "**Step 2: Compute base MSE loss**"
    ]))

    explanation.add_element(ContentAST.Equation(
      f"L_{{base}} = \\frac{{1}}{{2}}(y - \\hat{{y}})^2 = \\frac{{1}}{{2}}({self.target:.1f} - {self.prediction:.4f})^2 = {self.base_loss:.4f}",
      inline=False
    ))

    # Step 3: L2 penalty
    explanation.add_element(ContentAST.Paragraph([
      "**Step 3: Compute L2 penalty**"
    ]))

    weight_squares = [f"{w:.1f}^2" for w in self.weights]
    sum_squares = sum(w**2 for w in self.weights)

    explanation.add_element(ContentAST.Equation(
      f"L_{{reg}} = \\frac{{\\lambda}}{{2}} \\sum w_i^2 = \\frac{{{self.lambda_reg}}}{{2}}({' + '.join(weight_squares)}) = \\frac{{{self.lambda_reg}}}{{2}} \\times {sum_squares:.4f} = {self.l2_penalty:.4f}",
      inline=False
    ))

    # Step 4: Total loss
    explanation.add_element(ContentAST.Paragraph([
      "**Step 4: Compute total loss**"
    ]))

    explanation.add_element(ContentAST.Equation(
      f"L_{{total}} = L_{{base}} + L_{{reg}} = {self.base_loss:.4f} + {self.l2_penalty:.4f} = {self.total_loss:.4f}",
      inline=False
    ))

    # Step 5: Gradient with regularization
    explanation.add_element(ContentAST.Paragraph([
      "**Step 5: Compute gradient with regularization**"
    ]))

    explanation.add_element(ContentAST.Paragraph([
      ContentAST.Equation(r"w_0", inline=True),
      " (the bias term):"
    ]))

    explanation.add_element(ContentAST.Equation(
      f"\\frac{{\\partial L_{{base}}}}{{\\partial w_0}} = -(y - \\hat{{y}}) \\times 1 = -({self.target:.1f} - {self.prediction:.4f}) = {self.grad_base_w0:.4f}",
      inline=False
    ))

    explanation.add_element(ContentAST.Equation(
      f"\\frac{{\\partial L_{{reg}}}}{{\\partial w_0}} = \\lambda w_0 = {self.lambda_reg} \\times {self.weights[0]:.1f} = {self.grad_reg_w0:.4f}",
      inline=False
    ))

    explanation.add_element(ContentAST.Equation(
      f"\\frac{{\\partial L_{{total}}}}{{\\partial w_0}} = {self.grad_base_w0:.4f} + {self.grad_reg_w0:.4f} = {self.grad_total_w0:.4f}",
      inline=False
    ))

    explanation.add_element(ContentAST.Paragraph([
      "The regularization term adds ",
      ContentAST.Equation(f"\\lambda w_0 = {self.grad_reg_w0:.4f}", inline=True),
      " to the gradient, pushing the weight toward zero."
    ]))

    return explanation


@QuestionRegistry.register()
class MomentumOptimizerQuestion(Question, TableQuestionMixin, BodyTemplatesMixin):
  """
  Question asking students to perform gradient descent with momentum.

  Given a function, current weights, gradients, learning rate, and momentum coefficient,
  students calculate:
  - Velocity update using momentum
  - Weight update using the new velocity
  - Comparison to vanilla SGD (optional)
  """

  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.ML_OPTIMIZATION)
    super().__init__(*args, **kwargs)

    self.num_variables = kwargs.get("num_variables", 2)
    self.show_vanilla_sgd = kwargs.get("show_vanilla_sgd", True)

  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)

    # Generate well-conditioned quadratic function
    self.variables, self.function, self.gradient_function, self.equation = \
        generate_function(self.rng, self.num_variables, max_degree=2, use_quadratic=True)

    # Generate current weights (small integers)
    self.current_weights = [
      self.rng.choice([-2, -1, 0, 1, 2])
      for _ in range(self.num_variables)
    ]

    # Calculate gradient at current position
    subs_map = dict(zip(self.variables, self.current_weights))
    g_syms = self.gradient_function.subs(subs_map)
    self.gradients = [float(val) for val in g_syms]

    # Generate previous velocity (for momentum)
    # Start with small or zero velocity
    self.prev_velocity = [
      round(self.rng.uniform(-0.5, 0.5), 2)
      for _ in range(self.num_variables)
    ]

    # Hyperparameters
    self.learning_rate = self.rng.choice([0.01, 0.05, 0.1])
    self.momentum_beta = self.rng.choice([0.8, 0.9])

    # Calculate momentum updates
    # v_new = beta * v_old + (1 - beta) * gradient
    self.new_velocity = [
      self.momentum_beta * v_old + (1 - self.momentum_beta) * grad
      for v_old, grad in zip(self.prev_velocity, self.gradients)
    ]

    # w_new = w_old - alpha * v_new
    self.new_weights = [
      w - self.learning_rate * v
      for w, v in zip(self.current_weights, self.new_velocity)
    ]

    # Calculate vanilla SGD for comparison
    if self.show_vanilla_sgd:
      self.sgd_weights = [
        w - self.learning_rate * grad
        for w, grad in zip(self.current_weights, self.gradients)
      ]

    # Create answers
    self._create_answers()

  def _create_answers(self):
    """Create answer fields."""
    self.answers = {}

    # New velocity
    self.answers["velocity"] = Answer.vector_value("velocity", self.new_velocity)

    # New weights with momentum
    self.answers["weights_momentum"] = Answer.vector_value("weights_momentum", self.new_weights)

    # Vanilla SGD weights for comparison
    if self.show_vanilla_sgd:
      self.answers["weights_sgd"] = Answer.vector_value("weights_sgd", self.sgd_weights)

  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()

    # Question description
    body.add_element(ContentAST.Paragraph([
      "Consider the optimization problem of minimizing the function:"
    ]))

    body.add_element(ContentAST.Equation(
      sp.latex(self.function),
      inline=False
    ))

    body.add_element(ContentAST.Paragraph([
      "The gradient is:"
    ]))

    body.add_element(ContentAST.Equation(
      f"\\nabla f = {sp.latex(self.gradient_function)}",
      inline=False
    ))

    # Current state
    body.add_element(ContentAST.Paragraph([
      "**Current optimization state:**"
    ]))

    body.add_element(ContentAST.Paragraph([
      "Current weights: ",
      ContentAST.Equation(f"{format_vector(self.current_weights)}", inline=True)
    ]))

    body.add_element(ContentAST.Paragraph([
      "Gradient at current position: ",
      ContentAST.Equation(f"{format_vector(self.gradients)}", inline=True)
    ]))

    body.add_element(ContentAST.Paragraph([
      "Previous velocity: ",
      ContentAST.Equation(f"{format_vector(self.prev_velocity)}", inline=True)
    ]))

    # Hyperparameters
    body.add_element(ContentAST.Paragraph([
      "**Hyperparameters:**"
    ]))

    body.add_element(ContentAST.Paragraph([
      "Learning rate: ",
      ContentAST.Equation(f"\\alpha = {self.learning_rate}", inline=True)
    ]))

    body.add_element(ContentAST.Paragraph([
      "Momentum coefficient: ",
      ContentAST.Equation(f"\\beta = {self.momentum_beta}", inline=True)
    ]))

    # Questions
    body.add_element(ContentAST.Paragraph([
      "Calculate the following updates:"
    ]))

    # Answer table
    table_data = []
    table_data.append(["Update Type", "Formula", "Result"])

    table_data.append([
      "New velocity",
      ContentAST.Equation(r"v' = \beta v + (1-\beta)\nabla f", inline=True),
      ContentAST.Answer(self.answers["velocity"])
    ])

    table_data.append([
      "Weights (momentum)",
      ContentAST.Equation(r"w' = w - \alpha v'", inline=True),
      ContentAST.Answer(self.answers["weights_momentum"])
    ])

    if self.show_vanilla_sgd:
      table_data.append([
        "Weights (vanilla SGD)",
        ContentAST.Equation(r"w' = w - \alpha \nabla f", inline=True),
        ContentAST.Answer(self.answers["weights_sgd"])
      ])

    body.add_element(ContentAST.Table(data=table_data))

    return body

  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph([
      "Momentum helps gradient descent by accumulating a velocity vector in directions of "
      "consistent gradient, allowing faster convergence and reduced oscillation."
    ]))

    # Step 1: Calculate new velocity
    explanation.add_element(ContentAST.Paragraph([
      "**Step 1: Update velocity using momentum**"
    ]))

    explanation.add_element(ContentAST.Paragraph([
      "The momentum update formula is:"
    ]))

    explanation.add_element(ContentAST.Equation(
      f"v' = \\beta v + (1 - \\beta) \\nabla f",
      inline=False
    ))

    # Show calculation for each component
    for i in range(self.num_variables):
      var_name = f"x_{i}"
      explanation.add_element(ContentAST.Equation(
        f"v'[{i}] = {self.momentum_beta} \\times {self.prev_velocity[i]:.2f} + "
        f"{1 - self.momentum_beta} \\times {self.gradients[i]:.4f} = {self.new_velocity[i]:.4f}",
        inline=False
      ))

    # Step 2: Update weights with momentum
    explanation.add_element(ContentAST.Paragraph([
      "**Step 2: Update weights using new velocity**"
    ]))

    explanation.add_element(ContentAST.Equation(
      f"w' = w - \\alpha v'",
      inline=False
    ))

    for i in range(self.num_variables):
      explanation.add_element(ContentAST.Equation(
        f"w[{i}] = {self.current_weights[i]} - {self.learning_rate} \\times {self.new_velocity[i]:.4f} = {self.new_weights[i]:.4f}",
        inline=False
      ))

    # Comparison with vanilla SGD
    if self.show_vanilla_sgd:
      explanation.add_element(ContentAST.Paragraph([
        "**Comparison with vanilla SGD:**"
      ]))

      explanation.add_element(ContentAST.Paragraph([
        "Vanilla SGD (no momentum) would update directly using the gradient:"
      ]))

      explanation.add_element(ContentAST.Equation(
        f"w' = w - \\alpha \\nabla f",
        inline=False
      ))

      for i in range(self.num_variables):
        explanation.add_element(ContentAST.Equation(
          f"w[{i}] = {self.current_weights[i]} - {self.learning_rate} \\times {self.gradients[i]:.4f} = {self.sgd_weights[i]:.4f}",
          inline=False
        ))

      explanation.add_element(ContentAST.Paragraph([
        "The momentum update differs because it incorporates the previous velocity, "
        "which can help accelerate learning and smooth out noisy gradients."
      ]))

    return explanation
