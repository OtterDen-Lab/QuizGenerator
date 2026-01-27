from __future__ import annotations

import abc
import io
import logging
import math
import numpy as np
import uuid
import os
from typing import List, Tuple, Dict, Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import QuizGenerator.contentast as ca
from QuizGenerator.question import Question, QuestionRegistry
from QuizGenerator.mixins import TableQuestionMixin, BodyTemplatesMixin
from ..models.matrices import MatrixQuestion

log = logging.getLogger(__name__)


class SimpleNeuralNetworkBase(MatrixQuestion, abc.ABC):
  """
  Base class for simple neural network questions.

  Generates a small feedforward network:
  - 2-3 input neurons
  - 2 hidden neurons (single hidden layer)
  - 1 output neuron
  - Random weights and biases
  - Runs forward pass and stores all activations
  """

  # Activation function types
  ACTIVATION_SIGMOID = "sigmoid"
  ACTIVATION_RELU = "relu"
  ACTIVATION_LINEAR = "linear"

  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.ML_OPTIMIZATION)
    super().__init__(*args, **kwargs)

    # Network architecture parameters
    self.num_inputs = kwargs.get("num_inputs", 2)
    self.num_hidden = kwargs.get("num_hidden", 2)
    self.num_outputs = kwargs.get("num_outputs", 1)

    # Configuration
    self.activation_function = None
    self.use_bias = kwargs.get("use_bias", True)
    self.param_digits = kwargs.get("param_digits", 1)  # Precision for weights/biases

    # Network parameters (weights and biases)
    self.W1 = None  # Input to hidden weights (num_hidden x num_inputs)
    self.b1 = None  # Hidden layer biases (num_hidden,)
    self.W2 = None  # Hidden to output weights (num_outputs x num_hidden)
    self.b2 = None  # Output layer biases (num_outputs,)

    # Input data and forward pass results
    self.X = None  # Input values (num_inputs,)
    self.z1 = None  # Hidden layer pre-activation (num_hidden,)
    self.a1 = None  # Hidden layer activations (num_hidden,)
    self.z2 = None  # Output layer pre-activation (num_outputs,)
    self.a2 = None  # Output layer activation (prediction)

    # Target and loss (for backprop questions)
    self.y_target = None
    self.loss = None

    # Gradients (for backprop questions)
    self.dL_da2 = None  # Gradient of loss w.r.t. output
    self.da2_dz2 = None  # Gradient of activation w.r.t. pre-activation
    self.dL_dz2 = None  # Gradient of loss w.r.t. output pre-activation

  def _generate_network(self, weight_range=(-2, 2), input_range=(-3, 3)):
    """Generate random network parameters and input."""
    # Generate weights using MatrixQuestion's rounded matrix method
    # Use param_digits to match display precision in tables and explanations
    self.W1 = self.get_rounded_matrix(
      (self.num_hidden, self.num_inputs),
      low=weight_range[0],
      high=weight_range[1],
      digits_to_round=self.param_digits
    )

    self.W2 = self.get_rounded_matrix(
      (self.num_outputs, self.num_hidden),
      low=weight_range[0],
      high=weight_range[1],
      digits_to_round=self.param_digits
    )

    # Generate biases
    if self.use_bias:
      self.b1 = self.get_rounded_matrix(
        (self.num_hidden,),
        low=weight_range[0],
        high=weight_range[1],
        digits_to_round=self.param_digits
      )
      self.b2 = self.get_rounded_matrix(
        (self.num_outputs,),
        low=weight_range[0],
        high=weight_range[1],
        digits_to_round=self.param_digits
      )
    else:
      self.b1 = np.zeros(self.num_hidden)
      self.b2 = np.zeros(self.num_outputs)

    # Generate input values (keep as integers for simplicity)
    self.X = self.get_rounded_matrix(
      (self.num_inputs,),
      low=input_range[0],
      high=input_range[1],
      digits_to_round=0  # Round to integers
    )

  def _select_activation_function(self):
    """Randomly select an activation function."""
    activations = [
      self.ACTIVATION_SIGMOID,
      self.ACTIVATION_RELU
    ]
    self.activation_function = self.rng.choice(activations)

  def _apply_activation(self, z, function_type=None):
    """Apply activation function to pre-activation values."""
    if function_type is None:
      function_type = self.activation_function

    if function_type == self.ACTIVATION_SIGMOID:
      return 1 / (1 + np.exp(-z))
    elif function_type == self.ACTIVATION_RELU:
      return np.maximum(0, z)
    elif function_type == self.ACTIVATION_LINEAR:
      return z
    else:
      raise ValueError(f"Unknown activation function: {function_type}")

  def _activation_derivative(self, z, function_type=None):
    """Compute derivative of activation function."""
    if function_type is None:
      function_type = self.activation_function

    if function_type == self.ACTIVATION_SIGMOID:
      a = self._apply_activation(z, function_type)
      return a * (1 - a)
    elif function_type == self.ACTIVATION_RELU:
      return np.where(z > 0, 1, 0)
    elif function_type == self.ACTIVATION_LINEAR:
      return np.ones_like(z)
    else:
      raise ValueError(f"Unknown activation function: {function_type}")

  def _forward_pass(self):
    """Run forward pass through the network."""
    # Hidden layer
    self.z1 = self.W1 @ self.X + self.b1
    self.a1 = self._apply_activation(self.z1)

    # Output layer
    self.z2 = self.W2 @ self.a1 + self.b2
    self.a2 = self._apply_activation(self.z2, self.ACTIVATION_SIGMOID)  # Sigmoid output for binary classification

    # Round all computed values to display precision to ensure students can reproduce calculations
    # We display z and a values with 4 decimal places
    self.z1 = np.round(self.z1, 4)
    self.a1 = np.round(self.a1, 4)
    self.z2 = np.round(self.z2, 4)
    self.a2 = np.round(self.a2, 4)

    return self.a2

  def _compute_loss(self, y_target):
    """Compute binary cross-entropy loss."""
    self.y_target = y_target
    # BCE: L = -[y log(ŷ) + (1-y) log(1-ŷ)]
    # Add small epsilon to prevent log(0)
    epsilon = 1e-15
    y_pred = np.clip(self.a2[0], epsilon, 1 - epsilon)
    self.loss = -(y_target * np.log(y_pred) + (1 - y_target) * np.log(1 - y_pred))
    return self.loss

  def _compute_output_gradient(self):
    """Compute gradient of loss w.r.t. output."""
    # For BCE loss with sigmoid activation, the gradient simplifies beautifully:
    # dL/dz2 = ŷ - y (this is the combined derivative of BCE loss and sigmoid activation)
    #
    # Derivation:
    # BCE: L = -[y log(ŷ) + (1-y) log(1-ŷ)]
    # dL/dŷ = -[y/ŷ - (1-y)/(1-ŷ)]
    # Sigmoid: ŷ = σ(z), dŷ/dz = ŷ(1-ŷ)
    # Chain rule: dL/dz = dL/dŷ * dŷ/dz = ŷ - y

    self.dL_dz2 = self.a2[0] - self.y_target

    # Store intermediate values for explanation purposes
    # Clip to prevent division by zero (same epsilon as in loss calculation)
    epsilon = 1e-15
    y_pred_clipped = np.clip(self.a2[0], epsilon, 1 - epsilon)
    self.dL_da2 = -(self.y_target / y_pred_clipped - (1 - self.y_target) / (1 - y_pred_clipped))
    self.da2_dz2 = self.a2[0] * (1 - self.a2[0])

    return self.dL_dz2

  def _compute_gradient_W2(self, hidden_idx):
    """Compute gradient ∂L/∂W2[0, hidden_idx]."""
    # ∂L/∂w = dL/dz2 * ∂z2/∂w = dL/dz2 * a1[hidden_idx]
    return float(self.dL_dz2 * self.a1[hidden_idx])

  def _compute_gradient_W1(self, hidden_idx, input_idx):
    """Compute gradient ∂L/∂W1[hidden_idx, input_idx]."""
    # dL/dz1[hidden_idx] = dL/dz2 * ∂z2/∂a1[hidden_idx] * ∂a1/∂z1[hidden_idx]
    #                     = dL/dz2 * W2[0, hidden_idx] * activation'(z1[hidden_idx])

    dz2_da1 = self.W2[0, hidden_idx]
    da1_dz1 = self._activation_derivative(self.z1[hidden_idx])

    dL_dz1 = self.dL_dz2 * dz2_da1 * da1_dz1

    # ∂L/∂w = dL/dz1 * ∂z1/∂w = dL/dz1 * X[input_idx]
    return float(dL_dz1 * self.X[input_idx])

  def _get_activation_name(self):
    """Get human-readable activation function name."""
    if self.activation_function == self.ACTIVATION_SIGMOID:
      return "sigmoid"
    elif self.activation_function == self.ACTIVATION_RELU:
      return "ReLU"
    elif self.activation_function == self.ACTIVATION_LINEAR:
      return "linear"
    return "unknown"

  def _get_activation_formula(self):
    """Get LaTeX formula for activation function."""
    if self.activation_function == self.ACTIVATION_SIGMOID:
      return r"\sigma(z) = \frac{1}{1 + e^{-z}}"
    elif self.activation_function == self.ACTIVATION_RELU:
      return r"\text{ReLU}(z) = \max(0, z)"
    elif self.activation_function == self.ACTIVATION_LINEAR:
      return r"f(z) = z"
    return ""

  def _generate_parameter_table(self, include_activations=False, include_training_context=False):
    """
    Generate side-by-side tables showing all network parameters.

    Args:
      include_activations: If True, include computed activation values
      include_training_context: If True, include target, loss, etc. (for backprop questions)

    Returns:
      ca.TableGroup with network parameters in two side-by-side tables
    """
    # Left table: Inputs & Weights
    left_data = []
    left_data.append(["Symbol", "Value"])

    # Input values
    for i in range(self.num_inputs):
      left_data.append([
        ca.Equation(f"x_{i+1}", inline=True),
        f"{self.X[i]:.1f}"  # Inputs are always integers or 1 decimal
      ])

    # Weights from input to hidden
    for j in range(self.num_hidden):
      for i in range(self.num_inputs):
        left_data.append([
          ca.Equation(f"w_{{{j+1}{i+1}}}", inline=True),
          f"{self.W1[j, i]:.{self.param_digits}f}"
        ])

    # Weights from hidden to output
    for i in range(self.num_hidden):
      left_data.append([
        ca.Equation(f"w_{i+3}", inline=True),
        f"{self.W2[0, i]:.{self.param_digits}f}"
      ])

    # Right table: Biases, Activations, Training context
    right_data = []
    right_data.append(["Symbol", "Value"])

    # Hidden layer biases
    if self.use_bias:
      for j in range(self.num_hidden):
        right_data.append([
          ca.Equation(f"b_{j+1}", inline=True),
          f"{self.b1[j]:.{self.param_digits}f}"
        ])

    # Output bias
    if self.use_bias:
      right_data.append([
        ca.Equation(r"b_{out}", inline=True),
        f"{self.b2[0]:.{self.param_digits}f}"
      ])

    # Hidden layer activations (if computed and requested)
    if include_activations and self.a1 is not None:
      for i in range(self.num_hidden):
        right_data.append([
          ca.Equation(f"h_{i+1}", inline=True),
          f"{self.a1[i]:.4f}"
        ])

    # Output activation (if computed and requested)
    if include_activations and self.a2 is not None:
      right_data.append([
        ca.Equation(r"\hat{y}", inline=True),
        f"{self.a2[0]:.4f}"
      ])

    # Training context (target, loss - for backprop questions)
    if include_training_context:
      if self.y_target is not None:
        right_data.append([
          ca.Equation("y", inline=True),
          f"{int(self.y_target)}"  # Binary target (0 or 1)
        ])

      if self.loss is not None:
        right_data.append([
          ca.Equation("L", inline=True),
          f"{self.loss:.4f}"
        ])

    # Create table group
    table_group = ca.TableGroup()
    table_group.add_table(ca.Table(data=left_data))
    table_group.add_table(ca.Table(data=right_data))

    return table_group

  def _generate_network_diagram(self, show_weights=True, show_activations=False):
    """
    Generate a simple, clean network diagram.

    Args:
      show_weights: If True, display weights on edges
      show_activations: If True, display activation values on nodes

    Returns:
      BytesIO buffer containing PNG image
    """
    # Create figure with tight layout and equal aspect ratio
    fig = plt.figure(figsize=(8, 2.5))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')  # Keep circles circular
    ax.axis('off')

    # Node radius
    r = 0.15

    # Layer x-positions
    input_x = 0.5
    hidden_x = 2.0
    output_x = 3.5

    # Calculate y-positions for nodes (top to bottom order)
    def get_y_positions(n, include_bias=False):
      # If including bias, need one more position at the top
      total_nodes = n + 1 if include_bias else n
      if total_nodes == 1:
        return [1.0]
      spacing = min(2.0 / (total_nodes - 1), 0.6)
      # Start from top
      start = 1.0 + (total_nodes - 1) * spacing / 2
      positions = [start - i * spacing for i in range(total_nodes)]
      return positions

    # Input layer: bias (if present) at top, then x_1, x_2, ... going down
    input_positions = get_y_positions(self.num_inputs, include_bias=self.use_bias)
    if self.use_bias:
      bias1_y = input_positions[0]
      input_y = input_positions[1:]  # x_1 is second (below bias), x_2 is third, etc.
    else:
      bias1_y = None
      input_y = input_positions

    # Hidden layer: bias (if present) at top, then h_1, h_2, ... going down
    hidden_positions = get_y_positions(self.num_hidden, include_bias=self.use_bias)
    if self.use_bias:
      bias2_y = hidden_positions[0]
      hidden_y = hidden_positions[1:]
    else:
      bias2_y = None
      hidden_y = hidden_positions

    # Output layer: centered
    output_y = [1.0]

    # Draw edges first (so they're behind nodes)
    # Input to hidden
    for i in range(self.num_inputs):
      for j in range(self.num_hidden):
        ax.plot([input_x, hidden_x], [input_y[i], hidden_y[j]],
                'k-', linewidth=1, alpha=0.7, zorder=1)
        if show_weights:
          label_x = input_x + 0.3
          label_y = input_y[i] + (hidden_y[j] - input_y[i]) * 0.2
          # Use LaTeX math mode for proper subscript rendering
          weight_label = f'$w_{{{j+1}{i+1}}}$'
          ax.text(label_x, label_y, weight_label, fontsize=8,
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))

    # Bias to hidden
    if self.use_bias:
      for j in range(self.num_hidden):
        ax.plot([input_x, hidden_x], [bias1_y, hidden_y[j]],
                'k-', linewidth=1, alpha=0.7, zorder=1)
        if show_weights:
          label_x = input_x + 0.3
          label_y = bias1_y + (hidden_y[j] - bias1_y) * 0.2
          bias_label = f'$b_{{{j+1}}}$'
          ax.text(label_x, label_y, bias_label, fontsize=8,
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))

    # Hidden to output
    for i in range(self.num_hidden):
      ax.plot([hidden_x, output_x], [hidden_y[i], output_y[0]],
              'k-', linewidth=1, alpha=0.7, zorder=1)
      if show_weights:
        label_x = hidden_x + 0.3
        label_y = hidden_y[i] + (output_y[0] - hidden_y[i]) * 0.2
        weight_label = f'$w_{{{i+3}}}$'
        ax.text(label_x, label_y, weight_label, fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))

    # Bias to output
    if self.use_bias:
      ax.plot([hidden_x, output_x], [bias2_y, output_y[0]],
              'k-', linewidth=1, alpha=0.7, zorder=1)
      if show_weights:
        label_x = hidden_x + 0.3
        label_y = bias2_y + (output_y[0] - bias2_y) * 0.2
        bias_label = r'$b_{out}$'
        ax.text(label_x, label_y, bias_label, fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))

    # Draw nodes
    # Input nodes
    for i, y in enumerate(input_y):
      circle = plt.Circle((input_x, y), r, facecolor='lightgray',
                         edgecolor='black', linewidth=1.5, zorder=10)
      ax.add_patch(circle)
      label = f'$x_{{{i+1}}}$' if not show_activations else f'$x_{{{i+1}}}$={self.X[i]:.1f}'
      ax.text(input_x - r - 0.15, y, label, fontsize=10, ha='right', va='center')

    # Bias nodes
    if self.use_bias:
      circle = plt.Circle((input_x, bias1_y), r, facecolor='lightgray',
                         edgecolor='black', linewidth=1.5, zorder=10)
      ax.add_patch(circle)
      ax.text(input_x, bias1_y, '1', fontsize=10, ha='center', va='center', weight='bold')

      circle = plt.Circle((hidden_x, bias2_y), r, facecolor='lightgray',
                         edgecolor='black', linewidth=1.5, zorder=10)
      ax.add_patch(circle)
      ax.text(hidden_x, bias2_y, '1', fontsize=10, ha='center', va='center', weight='bold')

    # Hidden nodes
    for i, y in enumerate(hidden_y):
      circle = plt.Circle((hidden_x, y), r, facecolor='lightblue',
                         edgecolor='black', linewidth=1.5, zorder=10)
      ax.add_patch(circle)
      ax.plot([hidden_x, hidden_x], [y - r*0.7, y + r*0.7], 'k-', linewidth=1.2, zorder=11)
      ax.text(hidden_x - r*0.35, y, r'$\Sigma$', fontsize=11, ha='center', va='center', zorder=12)
      ax.text(hidden_x + r*0.35, y, r'$f$', fontsize=10, ha='center', va='center', zorder=12, style='italic')
      if show_activations and self.a1 is not None:
        ax.text(hidden_x, y - r - 0.15, f'{self.a1[i]:.2f}', fontsize=8, ha='center', va='top')

    # Output node
    y = output_y[0]
    circle = plt.Circle((output_x, y), r, facecolor='lightblue',
                       edgecolor='black', linewidth=1.5, zorder=10)
    ax.add_patch(circle)
    ax.plot([output_x, output_x], [y - r*0.7, y + r*0.7], 'k-', linewidth=1.2, zorder=11)
    ax.text(output_x - r*0.35, y, r'$\Sigma$', fontsize=11, ha='center', va='center', zorder=12)
    ax.text(output_x + r*0.35, y, r'$f$', fontsize=10, ha='center', va='center', zorder=12, style='italic')
    label = r'$\hat{y}$' if not show_activations else f'$\\hat{{y}}$={self.a2[0]:.2f}'
    ax.text(output_x + r + 0.15, y, label, fontsize=10, ha='left', va='center')

    # Save to buffer with minimal padding
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.0)
    plt.close(fig)
    buffer.seek(0)

    return buffer

  def _generate_ascii_network(self):
    """Generate ASCII art representation of the network for alt-text."""
    lines = []
    lines.append("Network Architecture:")
    lines.append("")
    lines.append("Input Layer:     Hidden Layer:      Output Layer:")

    # For 2 inputs, 2 hidden, 1 output
    if self.num_inputs == 2 and self.num_hidden == 2:
      lines.append(f"   x₁ ----[w₁₁]---→ h₁ ----[w₃]----→")
      lines.append(f"        \\      /     \\          /")
      lines.append(f"         \\    /       \\        /")
      lines.append(f"          \\  /         \\      /       ŷ")
      lines.append(f"           \\/           \\    /")
      lines.append(f"           /\\            \\  /")
      lines.append(f"          /  \\            \\/")
      lines.append(f"         /    \\           /\\")
      lines.append(f"        /      \\         /  \\")
      lines.append(f"   x₂ ----[w₂₁]---→ h₂ ----[w₄]----→")
    else:
      # Generic representation
      for i in range(max(self.num_inputs, self.num_hidden)):
        parts = []
        if i < self.num_inputs:
          parts.append(f"   x₁{i+1}")
        else:
          parts.append("      ")
        parts.append(" ---→ ")
        if i < self.num_hidden:
          parts.append(f"h₁{i+1}")
        else:
          parts.append("  ")
        parts.append(" ---→ ")
        if i == self.num_hidden // 2:
          parts.append("ŷ")
        lines.append("".join(parts))

    lines.append("")
    lines.append(f"Activation function: {self._get_activation_name()}")

    return "\n".join(lines)


@QuestionRegistry.register()
class ForwardPassQuestion(SimpleNeuralNetworkBase):
  """
  Question asking students to calculate forward pass through a simple network.

  Students calculate:
  - Hidden layer activations (h₁, h₂)
  - Final output (ŷ)
  """

  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)

    # Generate network
    self._generate_network()
    self._select_activation_function()

    # Run forward pass to get correct answers
    self._forward_pass()

    # Create answer fields
    self._create_answers()

  def _create_answers(self):
    """Create answer fields for forward pass values."""
    self.answers = {}

    # Hidden layer activations
    for i in range(self.num_hidden):
      key = f"h{i+1}"
      self.answers[key] = ca.AnswerTypes.Float(float(self.a1[i]), label=f"h_{i + 1}")

    # Output
    self.answers["y_pred"] = ca.AnswerTypes.Float(float(self.a2[0]), label="ŷ")

  def _get_body(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    # Question description
    body.add_element(ca.Paragraph([
      f"Given the neural network below with {self._get_activation_name()} activation "
      f"in the hidden layer and sigmoid activation in the output layer (for binary classification), "
      f"calculate the forward pass for the given input values."
    ]))

    # Network diagram
    body.add_element(
      ca.Picture(
        img_data=self._generate_network_diagram(show_weights=True, show_activations=False),
        caption=f"Neural network architecture"
      )
    )

    # Network parameters table
    body.add_element(self._generate_parameter_table(include_activations=False))

    # Activation function
    body.add_element(ca.Paragraph([
      f"**Hidden layer activation:** {self._get_activation_name()}"
    ]))

    # Collect answers
    for i in range(self.num_hidden):
      answers.append(self.answers[f"h{i+1}"])

    answers.append(self.answers["y_pred"])

    body.add_element(ca.AnswerBlock(answers))

    return body, answers

  def get_body(self, **kwargs) -> ca.Section:
    """Build question body (backward compatible interface)."""
    body, _ = self._get_body(**kwargs)
    return body

  def _get_explanation(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph([
      "To solve this problem, we need to compute the forward pass through the network."
    ]))

    # Hidden layer calculations
    explanation.add_element(ca.Paragraph([
      "**Step 1: Calculate hidden layer pre-activations**"
    ]))

    for i in range(self.num_hidden):
      # Build equation for z_i
      terms = []
      for j in range(self.num_inputs):
        terms.append(f"({self.W1[i,j]:.{self.param_digits}f})({self.X[j]:.1f})")

      z_calc = " + ".join(terms)
      if self.use_bias:
        z_calc += f" + {self.b1[i]:.{self.param_digits}f}"

      explanation.add_element(ca.Equation(
        f"z_{i+1} = {z_calc} = {self.z1[i]:.4f}",
        inline=False
      ))

    # Hidden layer activations
    explanation.add_element(ca.Paragraph([
      f"**Step 2: Apply {self._get_activation_name()} activation**"
    ]))

    for i in range(self.num_hidden):
      if self.activation_function == self.ACTIVATION_SIGMOID:
        explanation.add_element(ca.Equation(
          f"h_{i+1} = \\sigma(z_{i+1}) = \\frac{{1}}{{1 + e^{{-{self.z1[i]:.4f}}}}} = {self.a1[i]:.4f}",
          inline=False
        ))
      elif self.activation_function == self.ACTIVATION_RELU:
        explanation.add_element(ca.Equation(
          f"h_{i+1} = \\text{{ReLU}}(z_{i+1}) = \\max(0, {self.z1[i]:.4f}) = {self.a1[i]:.4f}",
          inline=False
        ))
      else:
        explanation.add_element(ca.Equation(
          f"h_{i+1} = z_{i+1} = {self.a1[i]:.4f}",
          inline=False
        ))

    # Output layer
    explanation.add_element(ca.Paragraph([
      "**Step 3: Calculate output (with sigmoid activation)**"
    ]))

    terms = []
    for j in range(self.num_hidden):
      terms.append(f"({self.W2[0,j]:.{self.param_digits}f})({self.a1[j]:.4f})")

    z_out_calc = " + ".join(terms)
    if self.use_bias:
      z_out_calc += f" + {self.b2[0]:.{self.param_digits}f}"

    explanation.add_element(ca.Equation(
      f"z_{{out}} = {z_out_calc} = {self.z2[0]:.4f}",
      inline=False
    ))

    explanation.add_element(ca.Equation(
      f"\\hat{{y}} = \\sigma(z_{{out}}) = \\frac{{1}}{{1 + e^{{-{self.z2[0]:.4f}}}}} = {self.a2[0]:.4f}",
      inline=False
    ))

    explanation.add_element(ca.Paragraph([
      "(Note: The output layer uses sigmoid activation for binary classification, "
      "so the output is between 0 and 1, representing the probability of class 1)"
    ]))

    return explanation, []

  def get_explanation(self, **kwargs) -> ca.Section:
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation(**kwargs)
    return explanation


@QuestionRegistry.register()
class BackpropGradientQuestion(SimpleNeuralNetworkBase):
  """
  Question asking students to calculate gradients using backpropagation.

  Given a completed forward pass, students calculate:
  - Gradients for multiple specific weights (∂L/∂w)
  """

  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)

    # Generate network
    self._generate_network()
    self._select_activation_function()

    # Run forward pass
    self._forward_pass()

    # Generate binary target (0 or 1)
    # Choose the opposite of what the network predicts to create meaningful gradients
    if self.a2[0] > 0.5:
      self.y_target = 0
    else:
      self.y_target = 1
    self._compute_loss(self.y_target)
    # Round loss to display precision (4 decimal places)
    self.loss = round(self.loss, 4)
    self._compute_output_gradient()

    # Create answer fields for specific weight gradients
    self._create_answers()

  def _create_answers(self):
    """Create answer fields for weight gradients."""
    self.answers = {}

    # Ask for gradients of 2-3 weights
    # Include at least one from each layer

    # Gradient for W2 (hidden to output)
    for i in range(self.num_hidden):
      key = f"dL_dw2_{i}"
      self.answers[key] = ca.AnswerTypes.Float(self._compute_gradient_W2(i), label=f"∂L/∂w_{i + 3}")

    # Gradient for W1 (input to hidden) - pick first hidden neuron
    for j in range(self.num_inputs):
      key = f"dL_dw1_0{j}"
      self.answers[key] = ca.AnswerTypes.Float(self._compute_gradient_W1(0, j), label=f"∂L/∂w_1{j + 1}")

  def _get_body(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    # Question description
    body.add_element(ca.Paragraph([
      f"Given the neural network below with {self._get_activation_name()} activation "
      f"in the hidden layer and sigmoid activation in the output layer (for binary classification), "
      f"a forward pass has been completed with the values shown. "
      f"Calculate the gradients (∂L/∂w) for the specified weights using backpropagation."
    ]))

    # Network diagram
    body.add_element(
      ca.Picture(
        img_data=self._generate_network_diagram(show_weights=True, show_activations=False),
        caption=f"Neural network architecture"
      )
    )

    # Network parameters and forward pass results table
    body.add_element(self._generate_parameter_table(include_activations=True, include_training_context=True))

    # Activation function
    body.add_element(ca.Paragraph([
      f"**Hidden layer activation:** {self._get_activation_name()}"
    ]))

    body.add_element(ca.Paragraph([
      "**Calculate the following gradients:**"
    ]))

    # Collect W2 gradient answers
    for i in range(self.num_hidden):
      answers.append(self.answers[f"dL_dw2_{i}"])

    # Collect W1 gradient answers (first hidden neuron)
    for j in range(self.num_inputs):
      answers.append(self.answers[f"dL_dw1_0{j}"])

    body.add_element(ca.AnswerBlock(answers))

    return body, answers

  def get_body(self, **kwargs) -> ca.Section:
    """Build question body (backward compatible interface)."""
    body, _ = self._get_body(**kwargs)
    return body

  def _get_explanation(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph([
      "To solve this problem, we use the chain rule to compute gradients via backpropagation."
    ]))

    # Output layer gradient
    explanation.add_element(ca.Paragraph([
      "**Step 1: Compute output layer gradient**"
    ]))

    explanation.add_element(ca.Paragraph([
      "For binary cross-entropy loss with sigmoid output activation, "
      "the gradient with respect to the pre-activation simplifies beautifully:"
    ]))

    explanation.add_element(ca.Equation(
      f"\\frac{{\\partial L}}{{\\partial z_{{out}}}} = \\hat{{y}} - y = {self.a2[0]:.4f} - {int(self.y_target)} = {self.dL_dz2:.4f}",
      inline=False
    ))

    explanation.add_element(ca.Paragraph([
      "(This elegant result comes from combining the BCE loss derivative and sigmoid activation derivative)"
    ]))

    # W2 gradients
    explanation.add_element(ca.Paragraph([
      "**Step 2: Gradients for hidden-to-output weights**"
    ]))

    explanation.add_element(ca.Paragraph([
      "Using the chain rule:"
    ]))

    for i in range(self.num_hidden):
      grad = self._compute_gradient_W2(i)
      explanation.add_element(ca.Equation(
        f"\\frac{{\\partial L}}{{\\partial w_{i+3}}} = \\frac{{\\partial L}}{{\\partial z_{{out}}}} \\cdot \\frac{{\\partial z_{{out}}}}{{\\partial w_{i+3}}} = {self.dL_dz2:.4f} \\cdot {self.a1[i]:.4f} = {grad:.4f}",
        inline=False
      ))

    # W1 gradients
    explanation.add_element(ca.Paragraph([
      "**Step 3: Gradients for input-to-hidden weights**"
    ]))

    explanation.add_element(ca.Paragraph([
      "First, compute the gradient flowing back to hidden layer:"
    ]))

    for j in range(self.num_inputs):
      # Compute intermediate values
      dz2_da1 = self.W2[0, 0]
      da1_dz1 = self._activation_derivative(self.z1[0])
      dL_dz1 = self.dL_dz2 * dz2_da1 * da1_dz1

      grad = self._compute_gradient_W1(0, j)

      if self.activation_function == self.ACTIVATION_SIGMOID:
        act_deriv_str = f"\\sigma'(z_1) = h_1(1-h_1) = {self.a1[0]:.4f}(1-{self.a1[0]:.4f}) = {da1_dz1:.4f}"
      elif self.activation_function == self.ACTIVATION_RELU:
        act_deriv_str = f"\\text{{ReLU}}'(z_1) = \\mathbb{{1}}(z_1 > 0) = {da1_dz1:.4f}"
      else:
        act_deriv_str = f"1"

      explanation.add_element(ca.Equation(
        f"\\frac{{\\partial L}}{{\\partial w_{{1{j+1}}}}} = \\frac{{\\partial L}}{{\\partial z_{{out}}}} \\cdot w_{3} \\cdot {act_deriv_str} \\cdot x_{j+1} = {self.dL_dz2:.4f} \\cdot {dz2_da1:.4f} \\cdot {da1_dz1:.4f} \\cdot {self.X[j]:.1f} = {grad:.4f}",
        inline=False
      ))

    return explanation, []

  def get_explanation(self, **kwargs) -> ca.Section:
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation(**kwargs)
    return explanation


@QuestionRegistry.register()
class EnsembleAveragingQuestion(Question):
  """
  Question asking students to combine predictions from multiple models (ensemble).

  Students calculate:
  - Mean prediction (for regression)
  - Optionally: variance or other statistics
  """

  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.ML_OPTIMIZATION)
    super().__init__(*args, **kwargs)

    self.num_models = kwargs.get("num_models", 5)
    self.predictions = None

  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)

    # Generate predictions from multiple models
    # Use a range that makes sense for typical regression problems
    base_value = self.rng.uniform(0, 10)
    self.predictions = [
      base_value + self.rng.uniform(-2, 2)
      for _ in range(self.num_models)
    ]

    # Round to make calculations easier
    self.predictions = [round(p, 1) for p in self.predictions]

    # Create answers
    self._create_answers()

  def _create_answers(self):
    """Create answer fields for ensemble statistics."""
    self.answers = {}

    # Mean prediction
    mean_pred = np.mean(self.predictions)
    self.answers["mean"] = ca.AnswerTypes.Float(float(mean_pred), label="Mean (average)")

    # Median (optional, but useful)
    median_pred = np.median(self.predictions)
    self.answers["median"] = ca.AnswerTypes.Float(float(median_pred), label="Median")

  def _get_body(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    # Question description
    body.add_element(ca.Paragraph([
      f"You have trained {self.num_models} different regression models on the same dataset. "
      f"For a particular test input, each model produces the following predictions:"
    ]))

    # Show predictions
    pred_list = ", ".join([f"{p:.1f}" for p in self.predictions])
    body.add_element(ca.Paragraph([
      f"Model predictions: {pred_list}"
    ]))

    # Question
    body.add_element(ca.Paragraph([
      "To create an ensemble, calculate the combined prediction using the following methods:"
    ]))

    # Collect answers
    answers.append(self.answers["mean"])
    answers.append(self.answers["median"])

    body.add_element(ca.AnswerBlock(answers))

    return body, answers

  def get_body(self, **kwargs) -> ca.Section:
    """Build question body (backward compatible interface)."""
    body, _ = self._get_body(**kwargs)
    return body

  def _get_explanation(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph([
      "Ensemble methods combine predictions from multiple models to create a more robust prediction."
    ]))

    # Mean calculation
    explanation.add_element(ca.Paragraph([
      "**Mean (Bagging approach):**"
    ]))

    pred_sum = " + ".join([f"{p:.1f}" for p in self.predictions])
    mean_val = np.mean(self.predictions)

    explanation.add_element(ca.Equation(
      f"\\text{{mean}} = \\frac{{{pred_sum}}}{{{self.num_models}}} = \\frac{{{sum(self.predictions):.1f}}}{{{self.num_models}}} = {mean_val:.4f}",
      inline=False
    ))

    # Median calculation
    explanation.add_element(ca.Paragraph([
      "**Median:**"
    ]))

    sorted_preds = sorted(self.predictions)
    sorted_str = ", ".join([f"{p:.1f}" for p in sorted_preds])
    median_val = np.median(self.predictions)

    explanation.add_element(ca.Paragraph([
      f"Sorted predictions: {sorted_str}"
    ]))

    if self.num_models % 2 == 1:
      mid_idx = self.num_models // 2
      explanation.add_element(ca.Paragraph([
        f"Middle value (position {mid_idx + 1}): {median_val:.1f}"
      ]))
    else:
      mid_idx1 = self.num_models // 2 - 1
      mid_idx2 = self.num_models // 2
      explanation.add_element(ca.Paragraph([
        f"Average of middle two values (positions {mid_idx1 + 1} and {mid_idx2 + 1}): "
        f"({sorted_preds[mid_idx1]:.1f} + {sorted_preds[mid_idx2]:.1f}) / 2 = {median_val:.1f}"
      ]))

    return explanation, []

  def get_explanation(self, **kwargs) -> ca.Section:
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation(**kwargs)
    return explanation


@QuestionRegistry.register()
class EndToEndTrainingQuestion(SimpleNeuralNetworkBase):
  """
  End-to-end training step question.

  Students perform a complete training iteration:
  1. Forward pass → prediction
  2. Loss calculation (MSE)
  3. Backpropagation → gradients for specific weights
  4. Weight update → new weight values
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.learning_rate = None
    self.new_W1 = None
    self.new_W2 = None

  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)

    # Generate network
    self._generate_network()
    self._select_activation_function()

    # Run forward pass
    self._forward_pass()

    # Generate binary target (0 or 1)
    # Choose the opposite of what the network predicts to create meaningful gradients
    if self.a2[0] > 0.5:
      self.y_target = 0
    else:
      self.y_target = 1
    self._compute_loss(self.y_target)
    # Round loss to display precision (4 decimal places)
    self.loss = round(self.loss, 4)
    self._compute_output_gradient()

    # Set learning rate (use small value for stability)
    self.learning_rate = round(self.rng.uniform(0.05, 0.2), 2)

    # Compute updated weights
    self._compute_weight_updates()

    # Create answers
    self._create_answers()

  def _compute_weight_updates(self):
    """Compute new weights after gradient descent step."""
    # Update W2
    self.new_W2 = np.copy(self.W2)
    for i in range(self.num_hidden):
      grad = self._compute_gradient_W2(i)
      self.new_W2[0, i] = self.W2[0, i] - self.learning_rate * grad

    # Update W1 (first hidden neuron only for simplicity)
    self.new_W1 = np.copy(self.W1)
    for j in range(self.num_inputs):
      grad = self._compute_gradient_W1(0, j)
      self.new_W1[0, j] = self.W1[0, j] - self.learning_rate * grad

  def _create_answers(self):
    """Create answer fields for all steps."""
    self.answers = {}

    # Forward pass answers
    self.answers["y_pred"] = ca.AnswerTypes.Float(float(self.a2[0]), label="1. Forward Pass - Network output ŷ")

    # Loss answer
    self.answers["loss"] = ca.AnswerTypes.Float(float(self.loss), label="2. Loss")

    # Gradient answers (for key weights)
    self.answers["grad_w3"] = ca.AnswerTypes.Float(self._compute_gradient_W2(0), label="3. Gradient ∂L/∂w₃")
    self.answers["grad_w11"] = ca.AnswerTypes.Float(self._compute_gradient_W1(0, 0), label="4. Gradient ∂L/∂w₁₁")

    # Updated weight answers
    self.answers["new_w3"] = ca.AnswerTypes.Float(float(self.new_W2[0, 0]), label="5. Updated w₃:")
    self.answers["new_w11"] = ca.AnswerTypes.Float(float(self.new_W1[0, 0]), label="6. Updated w₁₁:")

  def _get_body(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    # Question description
    body.add_element(ca.Paragraph([
      f"Given the neural network below with {self._get_activation_name()} activation "
      f"in the hidden layer and sigmoid activation in the output layer (for binary classification), "
      f"perform one complete training step (forward pass, loss calculation, "
      f"backpropagation, and weight update) for the given input and target."
    ]))

    # Network diagram
    body.add_element(
      ca.Picture(
        img_data=self._generate_network_diagram(show_weights=True, show_activations=False)
      )
    )

    # Training parameters
    body.add_element(ca.Paragraph([
      "**Training parameters:**"
    ]))

    body.add_element(ca.Paragraph([
      "Input: ",
      ca.Equation(f"x_1 = {self.X[0]:.1f}", inline=True),
      ", ",
      ca.Equation(f"x_2 = {self.X[1]:.1f}", inline=True)
    ]))

    body.add_element(ca.Paragraph([
      "Target: ",
      ca.Equation(f"y = {int(self.y_target)}", inline=True)
    ]))

    body.add_element(ca.Paragraph([
      "Learning rate: ",
      ca.Equation(f"\\alpha = {self.learning_rate}", inline=True)
    ]))

    body.add_element(ca.Paragraph([
      f"**Hidden layer activation:** {self._get_activation_name()}"
    ]))

    # Network parameters table
    body.add_element(self._generate_parameter_table(include_activations=False))

    # Collect answers
    answers.append(self.answers["y_pred"])
    answers.append(self.answers["loss"])
    answers.append(self.answers["grad_w3"])
    answers.append(self.answers["grad_w11"])
    answers.append(self.answers["new_w3"])
    answers.append(self.answers["new_w11"])

    body.add_element(ca.AnswerBlock(answers))

    return body, answers

  def get_body(self, **kwargs) -> ca.Section:
    """Build question body (backward compatible interface)."""
    body, _ = self._get_body(**kwargs)
    return body

  def _get_explanation(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph([
      "This problem requires performing one complete training iteration. Let's go through each step."
    ]))

    # Step 1: Forward pass
    explanation.add_element(ca.Paragraph([
      "**Step 1: Forward Pass**"
    ]))

    # Hidden layer
    z1_0 = self.W1[0, 0] * self.X[0] + self.W1[0, 1] * self.X[1] + self.b1[0]
    explanation.add_element(ca.Equation(
      f"z_1 = w_{{11}} x_1 + w_{{12}} x_2 + b_1 = {self.W1[0,0]:.{self.param_digits}f} \\cdot {self.X[0]:.1f} + {self.W1[0,1]:.{self.param_digits}f} \\cdot {self.X[1]:.1f} + {self.b1[0]:.{self.param_digits}f} = {self.z1[0]:.4f}",
      inline=False
    ))

    explanation.add_element(ca.Equation(
      f"h_1 = {self._get_activation_name()}(z_1) = {self.a1[0]:.4f}",
      inline=False
    ))

    # Similarly for h2 (abbreviated)
    explanation.add_element(ca.Equation(
      f"h_2 = {self.a1[1]:.4f} \\text{{ (calculated similarly)}}",
      inline=False
    ))

    # Output (pre-activation)
    z2 = self.W2[0, 0] * self.a1[0] + self.W2[0, 1] * self.a1[1] + self.b2[0]
    explanation.add_element(ca.Equation(
      f"z_{{out}} = w_3 h_1 + w_4 h_2 + b_2 = {self.W2[0,0]:.{self.param_digits}f} \\cdot {self.a1[0]:.4f} + {self.W2[0,1]:.{self.param_digits}f} \\cdot {self.a1[1]:.4f} + {self.b2[0]:.{self.param_digits}f} = {self.z2[0]:.4f}",
      inline=False
    ))

    # Output (sigmoid activation)
    explanation.add_element(ca.Equation(
      f"\\hat{{y}} = \\sigma(z_{{out}}) = \\frac{{1}}{{1 + e^{{-{self.z2[0]:.4f}}}}} = {self.a2[0]:.4f}",
      inline=False
    ))

    # Step 2: Loss
    explanation.add_element(ca.Paragraph([
      "**Step 2: Calculate Loss (Binary Cross-Entropy)**"
    ]))

    # Show the full BCE formula first
    explanation.add_element(ca.Equation(
      f"L = -[y \\log(\\hat{{y}}) + (1-y) \\log(1-\\hat{{y}})]",
      inline=False
    ))

    # Then evaluate it
    if self.y_target == 1:
      explanation.add_element(ca.Equation(
        f"L = -[1 \\cdot \\log({self.a2[0]:.4f}) + 0 \\cdot \\log(1-{self.a2[0]:.4f})] = -\\log({self.a2[0]:.4f}) = {self.loss:.4f}",
        inline=False
      ))
    else:
      explanation.add_element(ca.Equation(
        f"L = -[0 \\cdot \\log({self.a2[0]:.4f}) + 1 \\cdot \\log(1-{self.a2[0]:.4f})] = -\\log({1-self.a2[0]:.4f}) = {self.loss:.4f}",
        inline=False
      ))

    # Step 3: Gradients
    explanation.add_element(ca.Paragraph([
      "**Step 3: Compute Gradients**"
    ]))

    explanation.add_element(ca.Paragraph([
      "For BCE with sigmoid, the output layer gradient simplifies to:"
    ]))

    explanation.add_element(ca.Equation(
      f"\\frac{{\\partial L}}{{\\partial z_{{out}}}} = \\hat{{y}} - y = {self.a2[0]:.4f} - {int(self.y_target)} = {self.dL_dz2:.4f}",
      inline=False
    ))

    grad_w3 = self._compute_gradient_W2(0)
    explanation.add_element(ca.Equation(
      f"\\frac{{\\partial L}}{{\\partial w_3}} = \\frac{{\\partial L}}{{\\partial z_{{out}}}} \\cdot h_1 = {self.dL_dz2:.4f} \\cdot {self.a1[0]:.4f} = {grad_w3:.4f}",
      inline=False
    ))

    grad_w11 = self._compute_gradient_W1(0, 0)
    dz2_da1 = self.W2[0, 0]
    da1_dz1 = self._activation_derivative(self.z1[0])

    if self.activation_function == self.ACTIVATION_SIGMOID:
      act_deriv_str = f"h_1(1-h_1)"
    elif self.activation_function == self.ACTIVATION_RELU:
      act_deriv_str = f"\\text{{ReLU}}'(z_1)"
    else:
      act_deriv_str = f"1"

    explanation.add_element(ca.Equation(
      f"\\frac{{\\partial L}}{{\\partial w_{{11}}}} = \\frac{{\\partial L}}{{\\partial z_{{out}}}} \\cdot w_3 \\cdot {act_deriv_str} \\cdot x_1 = {self.dL_dz2:.4f} \\cdot {dz2_da1:.4f} \\cdot {da1_dz1:.4f} \\cdot {self.X[0]:.1f} = {grad_w11:.4f}",
      inline=False
    ))

    # Step 4: Weight updates
    explanation.add_element(ca.Paragraph([
      "**Step 4: Update Weights**"
    ]))

    new_w3 = self.new_W2[0, 0]
    explanation.add_element(ca.Equation(
      f"w_3^{{new}} = w_3 - \\alpha \\frac{{\\partial L}}{{\\partial w_3}} = {self.W2[0,0]:.{self.param_digits}f} - {self.learning_rate} \\cdot {grad_w3:.4f} = {new_w3:.4f}",
      inline=False
    ))

    new_w11 = self.new_W1[0, 0]
    explanation.add_element(ca.Equation(
      f"w_{{11}}^{{new}} = w_{{11}} - \\alpha \\frac{{\\partial L}}{{\\partial w_{{11}}}} = {self.W1[0,0]:.{self.param_digits}f} - {self.learning_rate} \\cdot {grad_w11:.4f} = {new_w11:.4f}",
      inline=False
    ))

    explanation.add_element(ca.Paragraph([
      "These updated weights would be used in the next training iteration."
    ]))

    return explanation, []

  def get_explanation(self, **kwargs) -> ca.Section:
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation(**kwargs)
    return explanation
