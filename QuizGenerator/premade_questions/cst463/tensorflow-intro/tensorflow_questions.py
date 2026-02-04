from __future__ import annotations

import logging
import numpy as np
import sympy as sp
from typing import List, Tuple

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

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    context.num_layers = kwargs.get("num_layers")
    context.include_biases = kwargs.get("include_biases", True)

    # Generate random architecture
    if context.num_layers is None:
      context.num_layers = context.rng.choice([3, 4])

    # Generate layer sizes
    # Input layer: common sizes for typical problems
    input_sizes = [28*28, 32*32, 784, 1024, 64, 128]
    context.layer_sizes = [context.rng.choice(input_sizes)]

    # Hidden layers: reasonable sizes
    for _ in range(context.num_layers - 2):
      hidden_size = context.rng.choice([32, 64, 128, 256, 512])
      context.layer_sizes.append(hidden_size)

    # Output layer: typical classification sizes
    output_size = context.rng.choice([2, 10, 100, 1000])
    context.layer_sizes.append(output_size)

    # Calculate correct answers
    context.total_weights = 0
    context.total_biases = 0
    context.weights_per_layer = []
    context.biases_per_layer = []

    for i in range(len(context.layer_sizes) - 1):
      weights = context.layer_sizes[i] * context.layer_sizes[i+1]
      biases = context.layer_sizes[i+1] if context.include_biases else 0

      context.weights_per_layer.append(weights)
      context.biases_per_layer.append(biases)

      context.total_weights += weights
      context.total_biases += biases

    context.total_params = context.total_weights + context.total_biases
    return context

  @classmethod
  def _build_body(cls, context) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    # Question description
    body.add_element(ca.Paragraph([
      "Consider a fully-connected (dense) neural network with the following architecture:"
    ]))

    # Display architecture
    arch_parts = []
    for i, size in enumerate(context.layer_sizes):
      if i > 0:
        arch_parts.append(" → ")
      arch_parts.append(str(size))

    body.add_element(ca.Paragraph([
      "Architecture: " + "".join(arch_parts)
    ]))

    if context.include_biases:
      body.add_element(ca.Paragraph([
        "Each layer includes bias terms."
      ]))

    # Questions
    # Answer table
    table_data = []
    table_data.append(["Parameter Type", "Count"])

    total_weights_answer = ca.AnswerTypes.Int(context.total_weights, label="Total weights")
    total_biases_answer = None
    total_params_answer = ca.AnswerTypes.Int(context.total_params, label="Total trainable parameters")

    answers.append(total_weights_answer)
    table_data.append([
      "Total weights (connections between layers)",
      total_weights_answer
    ])

    if context.include_biases:
      total_biases_answer = ca.AnswerTypes.Int(context.total_biases, label="Total biases")
      answers.append(total_biases_answer)
      table_data.append([
        "Total biases",
        total_biases_answer
      ])

    answers.append(total_params_answer)
    table_data.append([
      "Total trainable parameters",
      total_params_answer
    ])

    body.add_element(ca.Table(data=table_data))

    return body, answers

  @classmethod
  def _build_explanation(cls, context) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph([
      "To count parameters in a dense neural network, we calculate weights and biases for each layer."
    ]))

    explanation.add_element(ca.Paragraph([
      ca.Text("Weights calculation:", emphasis=True)
    ]))

    for i in range(len(context.layer_sizes) - 1):
      input_size = context.layer_sizes[i]
      output_size = context.layer_sizes[i+1]
      weights = context.weights_per_layer[i]

      explanation.add_element(ca.Paragraph([
        f"Layer {i+1} → {i+2}: ",
        ca.Equation(f"{input_size} \\times {output_size} = {weights:,}", inline=True),
        " weights"
      ]))

    explanation.add_element(ca.Paragraph([
      "Total weights: ",
      ca.Equation(
        f"{' + '.join([f'{w:,}' for w in context.weights_per_layer])} = {context.total_weights:,}",
        inline=True
      )
    ]))

    if context.include_biases:
      explanation.add_element(ca.Paragraph([
        ca.Text("Biases calculation:", emphasis=True)
      ]))

      for i in range(len(context.layer_sizes) - 1):
        output_size = context.layer_sizes[i+1]
        biases = context.biases_per_layer[i]

        explanation.add_element(ca.Paragraph([
          f"Layer {i+2}: {biases:,} biases (one per neuron)"
        ]))

      explanation.add_element(ca.Paragraph([
        "Total biases: ",
        ca.Equation(
          f"{' + '.join([f'{b:,}' for b in context.biases_per_layer])} = {context.total_biases:,}",
          inline=True
        )
      ]))

    explanation.add_element(ca.Paragraph([
      ca.Text("Total trainable parameters:", emphasis=True)
    ]))

    if context.include_biases:
      explanation.add_element(ca.Equation(
        f"\\text{{Total}} = {context.total_weights:,} + {context.total_biases:,} = {context.total_params:,}",
        inline=False
      ))
    else:
      explanation.add_element(ca.Equation(
        f"\\text{{Total}} = {context.total_weights:,}",
        inline=False
      ))

    return explanation, []


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

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    context.vector_size = kwargs.get("vector_size")
    context.activation = kwargs.get("activation")

    # Generate random input vector
    if context.vector_size is None:
      context.vector_size = context.rng.choice([3, 4, 5])

    context.input_vector = [
      round(context.rng.uniform(-3, 3), 1)
      for _ in range(context.vector_size)
    ]

    # Select activation function
    if context.activation is None:
      activations = [
        cls.ACTIVATION_RELU,
        cls.ACTIVATION_SIGMOID,
        cls.ACTIVATION_TANH,
        cls.ACTIVATION_SOFTMAX,
      ]
      context.activation = context.rng.choice(activations)

    # For leaky ReLU, set alpha
    context.leaky_alpha = 0.01

    # Compute outputs
    context.output_vector = cls._compute_activation(context.activation, context.input_vector)
    return context

  @staticmethod
  def _compute_activation(activation, inputs):
    """Compute activation function output."""
    if activation == ActivationFunctionComputationQuestion.ACTIVATION_RELU:
      return [max(0, x) for x in inputs]

    elif activation == ActivationFunctionComputationQuestion.ACTIVATION_SIGMOID:
      return [1 / (1 + np.exp(-x)) for x in inputs]

    elif activation == ActivationFunctionComputationQuestion.ACTIVATION_TANH:
      return [np.tanh(x) for x in inputs]

    elif activation == ActivationFunctionComputationQuestion.ACTIVATION_SOFTMAX:
      # Subtract max for numerical stability
      exp_vals = [np.exp(x - max(inputs)) for x in inputs]
      sum_exp = sum(exp_vals)
      return [e / sum_exp for e in exp_vals]

    else:
      raise ValueError(f"Unknown activation: {activation}")

  @staticmethod
  def _get_activation_name(activation):
    """Get human-readable activation name."""
    names = {
      ActivationFunctionComputationQuestion.ACTIVATION_RELU: "ReLU",
      ActivationFunctionComputationQuestion.ACTIVATION_SIGMOID: "Sigmoid",
      ActivationFunctionComputationQuestion.ACTIVATION_TANH: "Tanh",
      ActivationFunctionComputationQuestion.ACTIVATION_SOFTMAX: "Softmax",
    }
    return names.get(activation, "Unknown")

  @staticmethod
  def _get_activation_formula(activation):
    """Get LaTeX formula for activation function."""
    if activation == ActivationFunctionComputationQuestion.ACTIVATION_RELU:
      return r"\text{ReLU}(x) = \max(0, x)"

    elif activation == ActivationFunctionComputationQuestion.ACTIVATION_SIGMOID:
      return r"\sigma(x) = \frac{1}{1 + e^{-x}}"

    elif activation == ActivationFunctionComputationQuestion.ACTIVATION_TANH:
      return r"\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}"

    elif activation == ActivationFunctionComputationQuestion.ACTIVATION_SOFTMAX:
      return r"\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}"

    return ""

  @classmethod
  def _build_body(cls, context) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    # Question description
    body.add_element(ca.Paragraph([
      f"Given the input vector below, compute the output after applying the {cls._get_activation_name(context.activation)} activation function."
    ]))

    # Display formula
    body.add_element(ca.Paragraph([
      "Activation function: ",
      ca.Equation(cls._get_activation_formula(context.activation), inline=True)
    ]))

    # Input vector
    input_str = ", ".join([f"{x:.1f}" for x in context.input_vector])
    body.add_element(ca.Paragraph([
      "Input: ",
      ca.Equation(f"[{input_str}]", inline=True)
    ]))

    # Answer table
    if context.activation == cls.ACTIVATION_SOFTMAX:
      body.add_element(ca.Paragraph([
        "Compute the output vector:"
      ]))

      output_answer = ca.AnswerTypes.Vector(context.output_vector, label="Output vector")
      answers.append(output_answer)
      table_data = []
      table_data.append(["Output Vector"])
      table_data.append([output_answer])

      body.add_element(ca.Table(data=table_data))

    else:
      body.add_element(ca.Paragraph([
        "Compute the output for each element:"
      ]))

      table_data = []
      table_data.append(["Input", "Output"])

      for i, x in enumerate(context.input_vector):
        answer = ca.AnswerTypes.Float(
          float(context.output_vector[i]),
          label=f"Output for input {context.input_vector[i]:.1f}"
        )
        answers.append(answer)
        table_data.append([
          ca.Equation(f"{x:.1f}", inline=True),
          answer
        ])

      body.add_element(ca.Table(data=table_data))

    return body, answers

  @classmethod
  def _build_explanation(cls, context) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph([
      f"To compute the {cls._get_activation_name(context.activation)} activation, we apply the formula to each input."
    ]))

    if context.activation == cls.ACTIVATION_SOFTMAX:
      explanation.add_element(ca.Paragraph([
        ca.Text("Softmax computation:", emphasis=True)
      ]))

      # Show exponentials
      exp_strs = [f"e^{{{x:.1f}}}" for x in context.input_vector]
      explanation.add_element(ca.Paragraph([
        "First, compute exponentials: ",
        ca.Equation(", ".join(exp_strs), inline=True)
      ]))

      # Numerical values
      exp_vals = [np.exp(x) for x in context.input_vector]
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

      for i, (exp_val, output) in enumerate(zip(exp_vals, context.output_vector)):
        explanation.add_element(ca.Equation(
          f"\\text{{softmax}}({context.input_vector[i]:.1f}) = \\frac{{{exp_val:.4f}}}{{{sum_exp:.4f}}} = {output:.4f}",
          inline=False
        ))

    else:
      explanation.add_element(ca.Paragraph([
        ca.Text("Element-wise computation:", emphasis=True)
      ]))

      for i, (x, y) in enumerate(zip(context.input_vector, context.output_vector)):
        if context.activation == cls.ACTIVATION_RELU:
          explanation.add_element(ca.Equation(
            f"\\text{{ReLU}}({x:.1f}) = \\max(0, {x:.1f}) = {y:.4f}",
            inline=False
          ))

        elif context.activation == cls.ACTIVATION_SIGMOID:
          explanation.add_element(ca.Equation(
            f"\\sigma({x:.1f}) = \\frac{{1}}{{1 + e^{{-{x:.1f}}}}} = {y:.4f}",
            inline=False
          ))

        elif context.activation == cls.ACTIVATION_TANH:
          explanation.add_element(ca.Equation(
            f"\\tanh({x:.1f}) = {y:.4f}",
            inline=False
          ))

    return explanation, []


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

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    context.num_weights = kwargs.get("num_weights")

    # Generate small network (2-4 weights for simplicity)
    if context.num_weights is None:
      context.num_weights = context.rng.choice([2, 3, 4])

    # Generate weights (small values)
    context.weights = [
      round(context.rng.uniform(-2, 2), 1)
      for _ in range(context.num_weights)
    ]

    # Generate input and target
    context.input_val = round(context.rng.uniform(-3, 3), 1)
    context.target = round(context.rng.uniform(-5, 5), 1)

    # Regularization coefficient
    context.lambda_reg = context.rng.choice([0.01, 0.05, 0.1, 0.5])

    # Forward pass (simple linear combination for simplicity)
    # prediction = sum(w_i * input^i) for i in 0..n
    # This gives us a polynomial: w0 + w1*x + w2*x^2 + ...
    context.prediction = sum(
      w * (context.input_val ** i)
      for i, w in enumerate(context.weights)
    )

    # Calculate losses
    context.base_loss = 0.5 * (context.target - context.prediction) ** 2
    context.l2_penalty = (context.lambda_reg / 2) * sum(w**2 for w in context.weights)
    context.total_loss = context.base_loss + context.l2_penalty

    # Calculate gradient for first weight (w0, the bias term)
    # dL_base/dw0 = -(target - prediction) * dPrediction/dw0
    # dPrediction/dw0 = input^0 = 1
    # dL_reg/dw0 = lambda * w0
    # dL_total/dw0 = dL_base/dw0 + dL_reg/dw0

    context.grad_base_w0 = -(context.target - context.prediction) * 1  # derivative of w0*x^0
    context.grad_reg_w0 = context.lambda_reg * context.weights[0]
    context.grad_total_w0 = context.grad_base_w0 + context.grad_reg_w0
    return context

  @classmethod
  def _build_body(cls, context) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    # Question description
    body.add_element(ca.Paragraph([
      "Consider a simple model with the following parameters:"
    ]))

    # Display weights
    weight_strs = [f"w_{i} = {w:.1f}" for i, w in enumerate(context.weights)]
    body.add_element(ca.Paragraph([
      "Weights: ",
      ca.Equation(", ".join(weight_strs), inline=True)
    ]))

    # Model equation
    terms = []
    for i, w in enumerate(context.weights):
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
      ca.Equation(f"x = {context.input_val:.1f}, y = {context.target:.1f}", inline=True)
    ]))

    # Regularization
    body.add_element(ca.Paragraph([
      "L2 regularization coefficient: ",
      ca.Equation(f"\\lambda = {context.lambda_reg}", inline=True)
    ]))

    body.add_element(ca.Paragraph([
      "Calculate the following:"
    ]))

    # Answer table
    table_data = []
    table_data.append(["Calculation", "Value"])

    prediction_answer = ca.AnswerTypes.Float(float(context.prediction), label="Prediction ŷ")
    base_loss_answer = ca.AnswerTypes.Float(float(context.base_loss), label="Base MSE loss")
    l2_penalty_answer = ca.AnswerTypes.Float(float(context.l2_penalty), label="L2 penalty")
    total_loss_answer = ca.AnswerTypes.Float(float(context.total_loss), label="Total loss")
    grad_total_w0_answer = ca.AnswerTypes.Float(float(context.grad_total_w0), label="Gradient ∂L/∂w₀")

    answers.append(prediction_answer)
    table_data.append([
      ca.Paragraph(["Prediction ", ca.Equation(r"\hat{y}", inline=True)]),
      prediction_answer
    ])

    answers.append(base_loss_answer)
    table_data.append([
      ca.Paragraph(["Base MSE loss: ", ca.Equation(r"L_{base} = (1/2)(y - \hat{y})^2", inline=True)]),
      base_loss_answer
    ])

    answers.append(l2_penalty_answer)
    table_data.append([
      ca.Paragraph(["L2 penalty: ", ca.Equation(r"L_{reg} = (\lambda/2)\sum w_i^2", inline=True)]),
      l2_penalty_answer
    ])

    answers.append(total_loss_answer)
    table_data.append([
      ca.Paragraph(["Total loss: ", ca.Equation(r"L_{total} = L_{base} + L_{reg}", inline=True)]),
      total_loss_answer
    ])

    answers.append(grad_total_w0_answer)
    table_data.append([
      ca.Paragraph(["Gradient: ", ca.Equation(r"\frac{\partial L_{total}}{\partial w_0}", inline=True)]),
      grad_total_w0_answer
    ])

    body.add_element(ca.Table(data=table_data))

    return body, answers

  @classmethod
  def _build_explanation(cls, context) -> Tuple[ca.Section, List[ca.Answer]]:
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
    for i, w in enumerate(context.weights):
      if i == 0:
        terms.append(f"{w:.1f}")
      else:
        x_term = f"{context.input_val:.1f}^{i}" if i > 1 else f"{context.input_val:.1f}"
        terms.append(f"{w:.1f} \\times {x_term}")

    explanation.add_element(ca.Equation(
      f"\\hat{{y}} = {' + '.join(terms)} = {context.prediction:.4f}",
      inline=False
    ))

    # Step 2: Base loss
    explanation.add_element(ca.Paragraph([
      ca.Text("Step 2: Compute base MSE loss", emphasis=True)
    ]))

    explanation.add_element(ca.Equation(
      f"L_{{base}} = \\frac{{1}}{{2}}(y - \\hat{{y}})^2 = \\frac{{1}}{{2}}({context.target:.1f} - {context.prediction:.4f})^2 = {context.base_loss:.4f}",
      inline=False
    ))

    # Step 3: L2 penalty
    explanation.add_element(ca.Paragraph([
      ca.Text("Step 3: Compute L2 penalty", emphasis=True)
    ]))

    weight_squares = [f"{w:.1f}^2" for w in context.weights]
    sum_squares = sum(w**2 for w in context.weights)

    explanation.add_element(ca.Equation(
      f"L_{{reg}} = \\frac{{\\lambda}}{{2}} \\sum w_i^2 = \\frac{{{context.lambda_reg}}}{{2}}({' + '.join(weight_squares)}) = \\frac{{{context.lambda_reg}}}{{2}} \\times {sum_squares:.4f} = {context.l2_penalty:.4f}",
      inline=False
    ))

    # Step 4: Total loss
    explanation.add_element(ca.Paragraph([
      ca.Text("Step 4: Compute total loss", emphasis=True)
    ]))

    explanation.add_element(ca.Equation(
      f"L_{{total}} = L_{{base}} + L_{{reg}} = {context.base_loss:.4f} + {context.l2_penalty:.4f} = {context.total_loss:.4f}",
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
      f"\\frac{{\\partial L_{{base}}}}{{\\partial w_0}} = -(y - \\hat{{y}}) \\times 1 = -({context.target:.1f} - {context.prediction:.4f}) = {context.grad_base_w0:.4f}",
      inline=False
    ))

    explanation.add_element(ca.Equation(
      f"\\frac{{\\partial L_{{reg}}}}{{\\partial w_0}} = \\lambda w_0 = {context.lambda_reg} \\times {context.weights[0]:.1f} = {context.grad_reg_w0:.4f}",
      inline=False
    ))

    explanation.add_element(ca.Equation(
      f"\\frac{{\\partial L_{{total}}}}{{\\partial w_0}} = {context.grad_base_w0:.4f} + {context.grad_reg_w0:.4f} = {context.grad_total_w0:.4f}",
      inline=False
    ))

    explanation.add_element(ca.Paragraph([
      "The regularization term adds ",
      ca.Equation(f"\\lambda w_0 = {context.grad_reg_w0:.4f}", inline=True),
      " to the gradient, pushing the weight toward zero."
    ]))

    return explanation, []


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

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    context.num_variables = kwargs.get("num_variables", 2)
    context.show_vanilla_sgd = kwargs.get("show_vanilla_sgd", True)

    # Generate well-conditioned quadratic function
    context.variables, context.function, context.gradient_function, context.equation = generate_function(
      context.rng,
      context.num_variables,
      max_degree=2,
      use_quadratic=True
    )

    # Generate current weights (small integers)
    context.current_weights = [
      context.rng.choice([-2, -1, 0, 1, 2])
      for _ in range(context.num_variables)
    ]

    # Calculate gradient at current position
    subs_map = dict(zip(context.variables, context.current_weights))
    g_syms = context.gradient_function.subs(subs_map)
    context.gradients = [float(val) for val in g_syms]

    # Generate previous velocity (for momentum)
    # Start with small or zero velocity
    context.prev_velocity = [
      round(context.rng.uniform(-0.5, 0.5), 2)
      for _ in range(context.num_variables)
    ]

    # Hyperparameters
    context.learning_rate = context.rng.choice([0.01, 0.05, 0.1])
    context.momentum_beta = context.rng.choice([0.8, 0.9])

    # Calculate momentum updates
    # v_new = beta * v_old + (1 - beta) * gradient
    context.new_velocity = [
      context.momentum_beta * v_old + (1 - context.momentum_beta) * grad
      for v_old, grad in zip(context.prev_velocity, context.gradients)
    ]

    # w_new = w_old - alpha * v_new
    context.new_weights = [
      w - context.learning_rate * v
      for w, v in zip(context.current_weights, context.new_velocity)
    ]

    # Calculate vanilla SGD for comparison
    if context.show_vanilla_sgd:
      context.sgd_weights = [
        w - context.learning_rate * grad
        for w, grad in zip(context.current_weights, context.gradients)
      ]
    return context

  @classmethod
  def _build_body(cls, context) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    # Question description
    body.add_element(ca.Paragraph([
      "Consider the optimization problem of minimizing the function:"
    ]))

    body.add_element(ca.Equation(
      sp.latex(context.function),
      inline=False
    ))

    body.add_element(ca.Paragraph([
      "The gradient is:"
    ]))

    body.add_element(ca.Equation(
      f"\\nabla f = {sp.latex(context.gradient_function)}",
      inline=False
    ))

    # Current state
    body.add_element(ca.Paragraph([
      ca.Text("Current optimization state:", emphasis=True)
    ]))

    body.add_element(ca.Paragraph([
      "Current weights: ",
      ca.Equation(f"{format_vector(context.current_weights)}", inline=True)
    ]))

    body.add_element(ca.Paragraph([
      "Previous velocity: ",
      ca.Equation(f"{format_vector(context.prev_velocity)}", inline=True)
    ]))

    # Hyperparameters
    body.add_element(ca.Paragraph([
      ca.Text("Hyperparameters:", emphasis=True)
    ]))

    body.add_element(ca.Paragraph([
      "Learning rate: ",
      ca.Equation(f"\\alpha = {context.learning_rate}", inline=True)
    ]))

    body.add_element(ca.Paragraph([
      "Momentum coefficient: ",
      ca.Equation(f"\\beta = {context.momentum_beta}", inline=True)
    ]))

    # Questions
    body.add_element(ca.Paragraph([
      "Calculate the following updates:"
    ]))

    # Answer table
    table_data = []
    table_data.append(["Update Type", "Formula", "Result"])

    velocity_answer = ca.AnswerTypes.Vector(context.new_velocity, label="New velocity")
    weights_momentum_answer = ca.AnswerTypes.Vector(context.new_weights, label="Weights (momentum)")
    weights_sgd_answer = None

    answers.append(velocity_answer)
    table_data.append([
      "New velocity",
      ca.Equation(r"v' = \beta v + (1-\beta)\nabla f", inline=True),
      velocity_answer
    ])

    answers.append(weights_momentum_answer)
    table_data.append([
      "Weights (momentum)",
      ca.Equation(r"w' = w - \alpha v'", inline=True),
      weights_momentum_answer
    ])

    if context.show_vanilla_sgd:
      weights_sgd_answer = ca.AnswerTypes.Vector(context.sgd_weights, label="Weights (vanilla SGD)")
      answers.append(weights_sgd_answer)
      table_data.append([
        "Weights (vanilla SGD)",
        ca.Equation(r"w' = w - \alpha \nabla f", inline=True),
        weights_sgd_answer
      ])

    body.add_element(ca.Table(data=table_data))

    return body, answers

  @classmethod
  def _build_explanation(cls, context) -> Tuple[ca.Section, List[ca.Answer]]:
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
    for i in range(context.num_variables):
      var_name = f"x_{i}"
      # Round all intermediate values to avoid floating point precision issues
      beta_times_v = round(context.momentum_beta * context.prev_velocity[i], digits)
      one_minus_beta = round(1 - context.momentum_beta, digits)
      one_minus_beta_times_grad = round((1 - context.momentum_beta) * context.gradients[i], digits)

      explanation.add_element(ca.Equation(
        f"v'[{i}] = {context.momentum_beta} \\times {context.prev_velocity[i]:.{digits}f} + "
        f"{one_minus_beta:.{digits}f} \\times {context.gradients[i]:.{digits}f} = "
        f"{beta_times_v:.{digits}f} + {one_minus_beta_times_grad:.{digits}f} = {context.new_velocity[i]:.{digits}f}",
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

    for i in range(context.num_variables):
      explanation.add_element(ca.Equation(
        f"w[{i}] = {context.current_weights[i]} - {context.learning_rate} \\times {context.new_velocity[i]:.4f} = {context.new_weights[i]:.4f}",
        inline=False
      ))

    # Comparison with vanilla SGD
    if context.show_vanilla_sgd:
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

      for i in range(context.num_variables):
        explanation.add_element(ca.Equation(
          f"w[{i}] = {context.current_weights[i]} - {context.learning_rate} \\times {context.gradients[i]:.4f} = {context.sgd_weights[i]:.4f}",
          inline=False
        ))

      explanation.add_element(ca.Paragraph([
        "The momentum update differs because it incorporates the previous velocity, "
        "which can help accelerate learning and smooth out noisy gradients."
      ]))

    return explanation, []
