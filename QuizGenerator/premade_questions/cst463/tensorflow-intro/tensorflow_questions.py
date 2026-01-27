from __future__ import annotations

import abc
import io
import logging
import re
import numpy as np
import sympy as sp
from typing import List, Tuple, Dict, Any

import QuizGenerator.contentast as ca
from QuizGenerator.question import Question, QuestionRegistry
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

    self.answers["total_weights"] = ca.AnswerTypes.Int(self.total_weights, label="Total weights")

    if self.include_biases:
      self.answers["total_biases"] = ca.AnswerTypes.Int(self.total_biases, label="Total biases")
      self.answers["total_params"] = ca.AnswerTypes.Int(self.total_params, label="Total trainable parameters")
    else:
      self.answers["total_params"] = ca.AnswerTypes.Int(self.total_params, label="Total trainable parameters")

  def _get_body(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    # Question description
    body.add_element(ca.Paragraph([
      "Consider a fully-connected (dense) neural network with the following architecture:"
    ]))

    # Display architecture
    arch_parts = []
    for i, size in enumerate(self.layer_sizes):
      if i > 0:
        arch_parts.append(" → ")
      arch_parts.append(str(size))

    body.add_element(ca.Paragraph([
      "Architecture: " + "".join(arch_parts)
    ]))

    if self.include_biases:
      body.add_element(ca.Paragraph([
        "Each layer includes bias terms."
      ]))

    # Questions
    # Answer table
    table_data = []
    table_data.append(["Parameter Type", "Count"])

    answers.append(self.answers["total_weights"])
    table_data.append([
      "Total weights (connections between layers)",
      self.answers["total_weights"]
    ])

    if self.include_biases:
      answers.append(self.answers["total_biases"])
      table_data.append([
        "Total biases",
        self.answers["total_biases"]
      ])

    answers.append(self.answers["total_params"])
    table_data.append([
      "Total trainable parameters",
      self.answers["total_params"]
    ])

    body.add_element(ca.Table(data=table_data))

    return body, answers

  def get_body(self, **kwargs) -> ca.Section:
    """Build question body (backward compatible interface)."""
    body, _ = self._get_body(**kwargs)
    return body

  def _get_explanation(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph([
      "To count parameters in a dense neural network, we calculate weights and biases for each layer."
    ]))

    explanation.add_element(ca.Paragraph([
      ca.Text("Weights calculation:", emphasis=True)
    ]))

    for i in range(len(self.layer_sizes) - 1):
      input_size = self.layer_sizes[i]
      output_size = self.layer_sizes[i+1]
      weights = self.weights_per_layer[i]

      explanation.add_element(ca.Paragraph([
        f"Layer {i+1} → {i+2}: ",
        ca.Equation(f"{input_size} \\times {output_size} = {weights:,}", inline=True),
        " weights"
      ]))

    explanation.add_element(ca.Paragraph([
      "Total weights: ",
      ca.Equation(
        f"{' + '.join([f'{w:,}' for w in self.weights_per_layer])} = {self.total_weights:,}",
        inline=True
      )
    ]))

    if self.include_biases:
      explanation.add_element(ca.Paragraph([
        ca.Text("Biases calculation:", emphasis=True)
      ]))

      for i in range(len(self.layer_sizes) - 1):
        output_size = self.layer_sizes[i+1]
        biases = self.biases_per_layer[i]

        explanation.add_element(ca.Paragraph([
          f"Layer {i+2}: {biases:,} biases (one per neuron)"
        ]))

      explanation.add_element(ca.Paragraph([
        "Total biases: ",
        ca.Equation(
          f"{' + '.join([f'{b:,}' for b in self.biases_per_layer])} = {self.total_biases:,}",
          inline=True
        )
      ]))

    explanation.add_element(ca.Paragraph([
      ca.Text("Total trainable parameters:", emphasis=True)
    ]))

    if self.include_biases:
      explanation.add_element(ca.Equation(
        f"\\text{{Total}} = {self.total_weights:,} + {self.total_biases:,} = {self.total_params:,}",
        inline=False
      ))
    else:
      explanation.add_element(ca.Equation(
        f"\\text{{Total}} = {self.total_weights:,}",
        inline=False
      ))

    return explanation, []

  def get_explanation(self, **kwargs) -> ca.Section:
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation(**kwargs)
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

    else:
      raise ValueError(f"Unknown activation: {self.activation}")

  def _get_activation_name(self):
    """Get human-readable activation name."""
    names = {
      self.ACTIVATION_RELU: "ReLU",
      self.ACTIVATION_SIGMOID: "Sigmoid",
      self.ACTIVATION_TANH: "Tanh",
      self.ACTIVATION_SOFTMAX: "Softmax",
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

    return ""

  def _create_answers(self):
    """Create answer fields."""
    self.answers = {}

    if self.activation == self.ACTIVATION_SOFTMAX:
      # Softmax: single vector answer
      self.answers["output"] = ca.AnswerTypes.Vector(self.output_vector, label="Output vector")
    else:
      # Element-wise: individual answers
      for i, output in enumerate(self.output_vector):
        key = f"output_{i}"
        self.answers[key] = ca.AnswerTypes.Float(float(output), label=f"Output for input {self.input_vector[i]:.1f}")

  def _get_body(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    # Question description
    body.add_element(ca.Paragraph([
      f"Given the input vector below, compute the output after applying the {self._get_activation_name()} activation function."
    ]))

    # Display formula
    body.add_element(ca.Paragraph([
      "Activation function: ",
      ca.Equation(self._get_activation_formula(), inline=True)
    ]))

    # Input vector
    input_str = ", ".join([f"{x:.1f}" for x in self.input_vector])
    body.add_element(ca.Paragraph([
      "Input: ",
      ca.Equation(f"[{input_str}]", inline=True)
    ]))

    # Answer table
    if self.activation == self.ACTIVATION_SOFTMAX:
      body.add_element(ca.Paragraph([
        "Compute the output vector:"
      ]))

      answers.append(self.answers["output"])
      table_data = []
      table_data.append(["Output Vector"])
      table_data.append([self.answers["output"]])

      body.add_element(ca.Table(data=table_data))

    else:
      body.add_element(ca.Paragraph([
        "Compute the output for each element:"
      ]))

      table_data = []
      table_data.append(["Input", "Output"])

      for i, x in enumerate(self.input_vector):
        answer = self.answers[f"output_{i}"]
        answers.append(answer)
        table_data.append([
          ca.Equation(f"{x:.1f}", inline=True),
          answer
        ])

      body.add_element(ca.Table(data=table_data))

    return body, answers

  def get_body(self, **kwargs) -> ca.Section:
    """Build question body (backward compatible interface)."""
    body, _ = self._get_body(**kwargs)
    return body

  def _get_explanation(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph([
      f"To compute the {self._get_activation_name()} activation, we apply the formula to each input."
    ]))

    if self.activation == self.ACTIVATION_SOFTMAX:
      explanation.add_element(ca.Paragraph([
        ca.Text("Softmax computation:", emphasis=True)
      ]))

      # Show exponentials
      exp_strs = [f"e^{{{x:.1f}}}" for x in self.input_vector]
      explanation.add_element(ca.Paragraph([
        "First, compute exponentials: ",
        ca.Equation(", ".join(exp_strs), inline=True)
      ]))

      # Numerical values
      exp_vals = [np.exp(x) for x in self.input_vector]
      exp_vals_str = ", ".join([f"{e:.4f}" for e in exp_vals])
      explanation.add_element(ca.Paragraph([
        ca.Equation(f"\\approx [{exp_vals_str}]", inline=True)
      ]))

      # Sum
      sum_exp = sum(exp_vals)
      explanation.add_element(ca.Paragraph([
        "Sum: ",
        ca.Equation(f"{sum_exp:.4f}", inline=True)
      ]))

      # Final outputs
      explanation.add_element(ca.Paragraph([
        "Divide each by the sum:"
      ]))

      for i, (exp_val, output) in enumerate(zip(exp_vals, self.output_vector)):
        explanation.add_element(ca.Equation(
          f"\\text{{softmax}}({self.input_vector[i]:.1f}) = \\frac{{{exp_val:.4f}}}{{{sum_exp:.4f}}} = {output:.4f}",
          inline=False
        ))

    else:
      explanation.add_element(ca.Paragraph([
        ca.Text("Element-wise computation:", emphasis=True)
      ]))

      for i, (x, y) in enumerate(zip(self.input_vector, self.output_vector)):
        if self.activation == self.ACTIVATION_RELU:
          explanation.add_element(ca.Equation(
            f"\\text{{ReLU}}({x:.1f}) = \\max(0, {x:.1f}) = {y:.4f}",
            inline=False
          ))

        elif self.activation == self.ACTIVATION_SIGMOID:
          explanation.add_element(ca.Equation(
            f"\\sigma({x:.1f}) = \\frac{{1}}{{1 + e^{{-{x:.1f}}}}} = {y:.4f}",
            inline=False
          ))

        elif self.activation == self.ACTIVATION_TANH:
          explanation.add_element(ca.Equation(
            f"\\tanh({x:.1f}) = {y:.4f}",
            inline=False
          ))

    return explanation, []

  def get_explanation(self, **kwargs) -> ca.Section:
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation(**kwargs)
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

    self.answers["prediction"] = ca.AnswerTypes.Float(float(self.prediction), label="Prediction ŷ")
    self.answers["base_loss"] = ca.AnswerTypes.Float(float(self.base_loss), label="Base MSE loss")
    self.answers["l2_penalty"] = ca.AnswerTypes.Float(float(self.l2_penalty), label="L2 penalty")
    self.answers["total_loss"] = ca.AnswerTypes.Float(float(self.total_loss), label="Total loss")
    self.answers["grad_total_w0"] = ca.AnswerTypes.Float(float(self.grad_total_w0), label="Gradient ∂L/∂w₀")

  def _get_body(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    # Question description
    body.add_element(ca.Paragraph([
      "Consider a simple model with the following parameters:"
    ]))

    # Display weights
    weight_strs = [f"w_{i} = {w:.1f}" for i, w in enumerate(self.weights)]
    body.add_element(ca.Paragraph([
      "Weights: ",
      ca.Equation(", ".join(weight_strs), inline=True)
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
    body.add_element(ca.Paragraph([
      "Model: ",
      ca.Equation(f"\\hat{{y}} = {model_eq}", inline=True)
    ]))

    # Data point
    body.add_element(ca.Paragraph([
      "Data point: ",
      ca.Equation(f"x = {self.input_val:.1f}, y = {self.target:.1f}", inline=True)
    ]))

    # Regularization
    body.add_element(ca.Paragraph([
      "L2 regularization coefficient: ",
      ca.Equation(f"\\lambda = {self.lambda_reg}", inline=True)
    ]))

    body.add_element(ca.Paragraph([
      "Calculate the following:"
    ]))

    # Answer table
    table_data = []
    table_data.append(["Calculation", "Value"])

    answers.append(self.answers["prediction"])
    table_data.append([
      ca.Paragraph(["Prediction ", ca.Equation(r"\hat{y}", inline=True)]),
      self.answers["prediction"]
    ])

    answers.append(self.answers["base_loss"])
    table_data.append([
      ca.Paragraph(["Base MSE loss: ", ca.Equation(r"L_{base} = (1/2)(y - \hat{y})^2", inline=True)]),
      self.answers["base_loss"]
    ])

    answers.append(self.answers["l2_penalty"])
    table_data.append([
      ca.Paragraph(["L2 penalty: ", ca.Equation(r"L_{reg} = (\lambda/2)\sum w_i^2", inline=True)]),
      self.answers["l2_penalty"]
    ])

    answers.append(self.answers["total_loss"])
    table_data.append([
      ca.Paragraph(["Total loss: ", ca.Equation(r"L_{total} = L_{base} + L_{reg}", inline=True)]),
      self.answers["total_loss"]
    ])

    answers.append(self.answers["grad_total_w0"])
    table_data.append([
      ca.Paragraph(["Gradient: ", ca.Equation(r"\frac{\partial L_{total}}{\partial w_0}", inline=True)]),
      self.answers["grad_total_w0"]
    ])

    body.add_element(ca.Table(data=table_data))

    return body, answers

  def get_body(self, **kwargs) -> ca.Section:
    """Build question body (backward compatible interface)."""
    body, _ = self._get_body(**kwargs)
    return body

  def _get_explanation(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph([
      "L2 regularization adds a penalty term to the loss function to prevent overfitting by keeping weights small."
    ]))

    # Step 1: Forward pass
    explanation.add_element(ca.Paragraph([
      ca.Text("Step 1: Compute prediction", emphasis=True)
    ]))

    terms = []
    for i, w in enumerate(self.weights):
      if i == 0:
        terms.append(f"{w:.1f}")
      else:
        x_term = f"{self.input_val:.1f}^{i}" if i > 1 else f"{self.input_val:.1f}"
        terms.append(f"{w:.1f} \\times {x_term}")

    explanation.add_element(ca.Equation(
      f"\\hat{{y}} = {' + '.join(terms)} = {self.prediction:.4f}",
      inline=False
    ))

    # Step 2: Base loss
    explanation.add_element(ca.Paragraph([
      ca.Text("Step 2: Compute base MSE loss", emphasis=True)
    ]))

    explanation.add_element(ca.Equation(
      f"L_{{base}} = \\frac{{1}}{{2}}(y - \\hat{{y}})^2 = \\frac{{1}}{{2}}({self.target:.1f} - {self.prediction:.4f})^2 = {self.base_loss:.4f}",
      inline=False
    ))

    # Step 3: L2 penalty
    explanation.add_element(ca.Paragraph([
      ca.Text("Step 3: Compute L2 penalty", emphasis=True)
    ]))

    weight_squares = [f"{w:.1f}^2" for w in self.weights]
    sum_squares = sum(w**2 for w in self.weights)

    explanation.add_element(ca.Equation(
      f"L_{{reg}} = \\frac{{\\lambda}}{{2}} \\sum w_i^2 = \\frac{{{self.lambda_reg}}}{{2}}({' + '.join(weight_squares)}) = \\frac{{{self.lambda_reg}}}{{2}} \\times {sum_squares:.4f} = {self.l2_penalty:.4f}",
      inline=False
    ))

    # Step 4: Total loss
    explanation.add_element(ca.Paragraph([
      ca.Text("Step 4: Compute total loss", emphasis=True)
    ]))

    explanation.add_element(ca.Equation(
      f"L_{{total}} = L_{{base}} + L_{{reg}} = {self.base_loss:.4f} + {self.l2_penalty:.4f} = {self.total_loss:.4f}",
      inline=False
    ))

    # Step 5: Gradient with regularization
    explanation.add_element(ca.Paragraph([
      ca.Text("Step 5: Compute gradient with regularization", emphasis=True)
    ]))

    explanation.add_element(ca.Paragraph([
      ca.Equation(r"w_0", inline=True),
      " (the bias term):"
    ]))

    explanation.add_element(ca.Equation(
      f"\\frac{{\\partial L_{{base}}}}{{\\partial w_0}} = -(y - \\hat{{y}}) \\times 1 = -({self.target:.1f} - {self.prediction:.4f}) = {self.grad_base_w0:.4f}",
      inline=False
    ))

    explanation.add_element(ca.Equation(
      f"\\frac{{\\partial L_{{reg}}}}{{\\partial w_0}} = \\lambda w_0 = {self.lambda_reg} \\times {self.weights[0]:.1f} = {self.grad_reg_w0:.4f}",
      inline=False
    ))

    explanation.add_element(ca.Equation(
      f"\\frac{{\\partial L_{{total}}}}{{\\partial w_0}} = {self.grad_base_w0:.4f} + {self.grad_reg_w0:.4f} = {self.grad_total_w0:.4f}",
      inline=False
    ))

    explanation.add_element(ca.Paragraph([
      "The regularization term adds ",
      ca.Equation(f"\\lambda w_0 = {self.grad_reg_w0:.4f}", inline=True),
      " to the gradient, pushing the weight toward zero."
    ]))

    return explanation, []

  def get_explanation(self, **kwargs) -> ca.Section:
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation(**kwargs)
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
    self.answers["velocity"] = ca.AnswerTypes.Vector(self.new_velocity, label="New velocity")

    # New weights with momentum
    self.answers["weights_momentum"] = ca.AnswerTypes.Vector(self.new_weights, label="Weights (momentum)")

    # Vanilla SGD weights for comparison
    if self.show_vanilla_sgd:
      self.answers["weights_sgd"] = ca.AnswerTypes.Vector(self.sgd_weights, label="Weights (vanilla SGD)")

  def _get_body(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    # Question description
    body.add_element(ca.Paragraph([
      "Consider the optimization problem of minimizing the function:"
    ]))

    body.add_element(ca.Equation(
      sp.latex(self.function),
      inline=False
    ))

    body.add_element(ca.Paragraph([
      "The gradient is:"
    ]))

    body.add_element(ca.Equation(
      f"\\nabla f = {sp.latex(self.gradient_function)}",
      inline=False
    ))

    # Current state
    body.add_element(ca.Paragraph([
      ca.Text("Current optimization state:", emphasis=True)
    ]))

    body.add_element(ca.Paragraph([
      "Current weights: ",
      ca.Equation(f"{format_vector(self.current_weights)}", inline=True)
    ]))

    body.add_element(ca.Paragraph([
      "Previous velocity: ",
      ca.Equation(f"{format_vector(self.prev_velocity)}", inline=True)
    ]))

    # Hyperparameters
    body.add_element(ca.Paragraph([
      ca.Text("Hyperparameters:", emphasis=True)
    ]))

    body.add_element(ca.Paragraph([
      "Learning rate: ",
      ca.Equation(f"\\alpha = {self.learning_rate}", inline=True)
    ]))

    body.add_element(ca.Paragraph([
      "Momentum coefficient: ",
      ca.Equation(f"\\beta = {self.momentum_beta}", inline=True)
    ]))

    # Questions
    body.add_element(ca.Paragraph([
      "Calculate the following updates:"
    ]))

    # Answer table
    table_data = []
    table_data.append(["Update Type", "Formula", "Result"])

    answers.append(self.answers["velocity"])
    table_data.append([
      "New velocity",
      ca.Equation(r"v' = \beta v + (1-\beta)\nabla f", inline=True),
      self.answers["velocity"]
    ])

    answers.append(self.answers["weights_momentum"])
    table_data.append([
      "Weights (momentum)",
      ca.Equation(r"w' = w - \alpha v'", inline=True),
      self.answers["weights_momentum"]
    ])

    if self.show_vanilla_sgd:
      answers.append(self.answers["weights_sgd"])
      table_data.append([
        "Weights (vanilla SGD)",
        ca.Equation(r"w' = w - \alpha \nabla f", inline=True),
        self.answers["weights_sgd"]
      ])

    body.add_element(ca.Table(data=table_data))

    return body, answers

  def get_body(self, **kwargs) -> ca.Section:
    """Build question body (backward compatible interface)."""
    body, _ = self._get_body(**kwargs)
    return body

  def _get_explanation(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph([
      "Momentum helps gradient descent by accumulating a velocity vector in directions of "
      "consistent gradient, allowing faster convergence and reduced oscillation."
    ]))

    # Step 1: Calculate new velocity
    explanation.add_element(ca.Paragraph([
      ca.Text("Step 1: Update velocity using momentum", emphasis=True)
    ]))

    explanation.add_element(ca.Paragraph([
      "The momentum update formula is:"
    ]))

    explanation.add_element(ca.Equation(
      f"v' = \\beta v + (1 - \\beta) \\nabla f",
      inline=False
    ))

    # Show calculation for each component
    digits = ca.Answer.DEFAULT_ROUNDING_DIGITS
    for i in range(self.num_variables):
      var_name = f"x_{i}"
      # Round all intermediate values to avoid floating point precision issues
      beta_times_v = round(self.momentum_beta * self.prev_velocity[i], digits)
      one_minus_beta = round(1 - self.momentum_beta, digits)
      one_minus_beta_times_grad = round((1 - self.momentum_beta) * self.gradients[i], digits)

      explanation.add_element(ca.Equation(
        f"v'[{i}] = {self.momentum_beta} \\times {self.prev_velocity[i]:.{digits}f} + "
        f"{one_minus_beta:.{digits}f} \\times {self.gradients[i]:.{digits}f} = "
        f"{beta_times_v:.{digits}f} + {one_minus_beta_times_grad:.{digits}f} = {self.new_velocity[i]:.{digits}f}",
        inline=False
      ))

    # Step 2: Update weights with momentum
    explanation.add_element(ca.Paragraph([
      ca.Text("Step 2: Update weights using new velocity", emphasis=True)
    ]))

    explanation.add_element(ca.Equation(
      f"w' = w - \\alpha v'",
      inline=False
    ))

    for i in range(self.num_variables):
      explanation.add_element(ca.Equation(
        f"w[{i}] = {self.current_weights[i]} - {self.learning_rate} \\times {self.new_velocity[i]:.4f} = {self.new_weights[i]:.4f}",
        inline=False
      ))

    # Comparison with vanilla SGD
    if self.show_vanilla_sgd:
      explanation.add_element(ca.Paragraph([
        ca.Text("Comparison with vanilla SGD:", emphasis=True)
      ]))

      explanation.add_element(ca.Paragraph([
        "Vanilla SGD (no momentum) would update directly using the gradient:"
      ]))

      explanation.add_element(ca.Equation(
        f"w' = w - \\alpha \\nabla f",
        inline=False
      ))

      for i in range(self.num_variables):
        explanation.add_element(ca.Equation(
          f"w[{i}] = {self.current_weights[i]} - {self.learning_rate} \\times {self.gradients[i]:.4f} = {self.sgd_weights[i]:.4f}",
          inline=False
        ))

      explanation.add_element(ca.Paragraph([
        "The momentum update differs because it incorporates the previous velocity, "
        "which can help accelerate learning and smooth out noisy gradients."
      ]))

    return explanation, []

  def get_explanation(self, **kwargs) -> ca.Section:
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation(**kwargs)
    return explanation
