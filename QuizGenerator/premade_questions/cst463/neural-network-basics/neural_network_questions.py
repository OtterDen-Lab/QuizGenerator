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

from QuizGenerator.contentast import ContentAST
from QuizGenerator.question import Question, Answer, QuestionRegistry
from QuizGenerator.mixins import TableQuestionMixin, BodyTemplatesMixin

log = logging.getLogger(__name__)


class SimpleNeuralNetworkBase(Question, abc.ABC):
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
    # Generate weights (using small values for numerical stability)
    self.W1 = np.array([
      [self.rng.uniform(weight_range[0], weight_range[1])
       for _ in range(self.num_inputs)]
      for _ in range(self.num_hidden)
    ])

    self.W2 = np.array([
      [self.rng.uniform(weight_range[0], weight_range[1])
       for _ in range(self.num_hidden)]
      for _ in range(self.num_outputs)
    ])

    # Generate biases
    if self.use_bias:
      self.b1 = np.array([
        self.rng.uniform(weight_range[0], weight_range[1])
        for _ in range(self.num_hidden)
      ])
      self.b2 = np.array([
        self.rng.uniform(weight_range[0], weight_range[1])
        for _ in range(self.num_outputs)
      ])
    else:
      self.b1 = np.zeros(self.num_hidden)
      self.b2 = np.zeros(self.num_outputs)

    # Round weights to make calculations cleaner
    self.W1 = np.round(self.W1 * 2) / 2  # Round to nearest 0.5
    self.W2 = np.round(self.W2 * 2) / 2
    self.b1 = np.round(self.b1 * 2) / 2
    self.b2 = np.round(self.b2 * 2) / 2

    # Generate input values
    self.X = np.array([
      self.rng.uniform(input_range[0], input_range[1])
      for _ in range(self.num_inputs)
    ])
    self.X = np.round(self.X)  # Use integer inputs for simplicity

  def _select_activation_function(self):
    """Randomly select an activation function."""
    activations = [
      self.ACTIVATION_SIGMOID,
      self.ACTIVATION_RELU,
      self.ACTIVATION_LINEAR
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
    self.a2 = self._apply_activation(self.z2, self.ACTIVATION_LINEAR)  # Linear output

    return self.a2

  def _compute_loss(self, y_target):
    """Compute MSE loss."""
    self.y_target = y_target
    self.loss = 0.5 * (y_target - self.a2[0]) ** 2
    return self.loss

  def _compute_output_gradient(self):
    """Compute gradient of loss w.r.t. output."""
    # For MSE loss: dL/da2 = -(y - a2)
    self.dL_da2 = -(self.y_target - self.a2[0])

    # For linear output activation: da2/dz2 = 1
    self.da2_dz2 = 1.0

    # Chain rule: dL/dz2 = dL/da2 * da2/dz2
    self.dL_dz2 = self.dL_da2 * self.da2_dz2

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

  def _generate_network_diagram(self, show_weights=True, show_activations=False):
    """
    Generate a clean, simple network diagram using matplotlib.

    Matches the class diagram style with:
    - Horizontal alignment of layers
    - Vertical grouping within layers
    - Sigma/f notation inside nodes
    - Clear weight labels on edges
    - Explicit bias nodes (labeled "1")

    Args:
      show_weights: If True, display weights on edges
      show_activations: If True, display activation values on nodes

    Returns:
      BytesIO buffer containing PNG image
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.axis('off')
    ax.set_aspect('equal')  # Keep circles circular

    # Layer positions - horizontal alignment
    layer_x = [0.15, 0.5, 0.85]

    # Vertical positioning - spread out evenly
    input_y_center = 0.5
    hidden_y_center = 0.5
    output_y = 0.5

    # Spacing for multiple neurons
    y_spacing = 0.22

    # Calculate positions for regular inputs
    input_y = [input_y_center + y_spacing * (0.5 - i/(max(self.num_inputs-1, 1)))
               for i in range(self.num_inputs)]
    hidden_y = [hidden_y_center + y_spacing * (0.5 - i/(max(self.num_hidden-1, 1)))
                for i in range(self.num_hidden)]

    # Add bias node positions (above regular inputs)
    if self.use_bias:
      bias1_y = max(input_y) + y_spacing * 0.8  # Bias for hidden layer
      bias2_y = max(hidden_y) + y_spacing * 0.8  # Bias for output layer
    else:
      bias1_y = None
      bias2_y = None

    # Node positions
    input_pos = [(layer_x[0], y) for y in input_y]
    hidden_pos = [(layer_x[1], y) for y in hidden_y]
    output_pos = [(layer_x[2], output_y)]

    if self.use_bias:
      bias1_pos = (layer_x[0], bias1_y)
      bias2_pos = (layer_x[1], bias2_y)
    else:
      bias1_pos = None
      bias2_pos = None

    # Draw edges with weights
    node_radius = 0.04

    # Input to hidden (regular inputs)
    for i, (x1, y1) in enumerate(input_pos):
      for j, (x2, y2) in enumerate(hidden_pos):
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=0.8, alpha=0.6, zorder=1)

        if show_weights:
          # Position weight label near the source node (left side)
          label_x = x1 + 0.06
          label_y = y1 + (y2 - y1) * 0.15  # Slight offset toward destination

          if not show_activations:
            weight_text = f"{self.W1[j, i]:.1f}"
          else:
            weight_text = f"$w_{{{j+1}{i+1}}}$"

          ax.text(label_x, label_y, weight_text, fontsize=7,
                  ha='left', va='center',
                  bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                           edgecolor='none', alpha=0.9))

    # Bias to hidden
    if self.use_bias and bias1_pos:
      x1, y1 = bias1_pos
      for j, (x2, y2) in enumerate(hidden_pos):
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=0.8, alpha=0.6, zorder=1)

        if show_weights:
          # Position bias weight label near source
          label_x = x1 + 0.06
          label_y = y1 + (y2 - y1) * 0.15

          if not show_activations:
            weight_text = f"{self.b1[j]:.1f}"
          else:
            weight_text = f"$b_{{{j+1}}}$"

          ax.text(label_x, label_y, weight_text, fontsize=7,
                  ha='left', va='center',
                  bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                           edgecolor='none', alpha=0.9))

    # Hidden to output (regular hidden neurons)
    for i, (x1, y1) in enumerate(hidden_pos):
      x2, y2 = output_pos[0]
      ax.plot([x1, x2], [y1, y2], 'k-', linewidth=0.8, alpha=0.6, zorder=1)

      if show_weights:
        # Position weight label near source
        label_x = x1 + 0.06
        label_y = y1 + (y2 - y1) * 0.15

        if not show_activations:
          weight_text = f"{self.W2[0, i]:.1f}"
        else:
          weight_text = f"$w_{{{i+self.num_inputs*self.num_hidden+1}}}$"

        ax.text(label_x, label_y, weight_text, fontsize=7,
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                         edgecolor='none', alpha=0.9))

    # Bias to output
    if self.use_bias and bias2_pos:
      x1, y1 = bias2_pos
      x2, y2 = output_pos[0]
      ax.plot([x1, x2], [y1, y2], 'k-', linewidth=0.8, alpha=0.6, zorder=1)

      if show_weights:
        # Position bias weight label near source
        label_x = x1 + 0.06
        label_y = y1 + (y2 - y1) * 0.15

        if not show_activations:
          weight_text = f"{self.b2[0]:.1f}"
        else:
          weight_text = "$b_{out}$"

        ax.text(label_x, label_y, weight_text, fontsize=7,
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                         edgecolor='none', alpha=0.9))

    # Draw nodes
    # Input nodes - simple circles with labels outside
    for i, (x, y) in enumerate(input_pos):
      circle = plt.Circle((x, y), node_radius*0.6, facecolor='lightgray',
                         edgecolor='black', linewidth=1, zorder=10)
      ax.add_patch(circle)

      # Label to the left
      if show_activations:
        label_text = f"$x_{{{i+1}}}$={self.X[i]:.1f}"
      else:
        label_text = f"$x_{{{i+1}}}$"
      ax.text(x - 0.10, y, label_text, fontsize=9, ha='right', va='center')

    # Bias nodes - small circles labeled "1"
    if self.use_bias and bias1_pos:
      x, y = bias1_pos
      circle = plt.Circle((x, y), node_radius*0.6, facecolor='lightgray',
                         edgecolor='black', linewidth=1, zorder=10)
      ax.add_patch(circle)
      # Label inside the circle
      ax.text(x, y, "1", fontsize=8, ha='center', va='center', weight='bold')

    if self.use_bias and bias2_pos:
      x, y = bias2_pos
      circle = plt.Circle((x, y), node_radius*0.6, facecolor='lightgray',
                         edgecolor='black', linewidth=1, zorder=10)
      ax.add_patch(circle)
      # Label inside the circle
      ax.text(x, y, "1", fontsize=8, ha='center', va='center', weight='bold')

    # Hidden nodes - larger circles with Σ/f inside
    for i, (x, y) in enumerate(hidden_pos):
      # Main circle
      circle = plt.Circle((x, y), node_radius, facecolor='lightblue',
                         edgecolor='black', linewidth=1.2, zorder=10)
      ax.add_patch(circle)

      # Vertical divider line
      ax.plot([x, x], [y - node_radius*0.7, y + node_radius*0.7],
              'k-', linewidth=1, zorder=11)

      # Sigma symbol on left, f on right
      ax.text(x - node_radius*0.35, y, r'$\Sigma$', fontsize=10,
              ha='center', va='center', zorder=12)
      ax.text(x + node_radius*0.35, y, r'$f$', fontsize=9,
              ha='center', va='center', zorder=12, style='italic')

      # Activation value or label below if showing activations
      if show_activations and self.a1 is not None:
        ax.text(x, y - node_radius - 0.04, f"{self.a1[i]:.2f}",
                fontsize=7, ha='center', va='top')

    # Output node
    x, y = output_pos[0]
    circle = plt.Circle((x, y), node_radius, facecolor='lightblue',
                       edgecolor='black', linewidth=1.2, zorder=10)
    ax.add_patch(circle)

    # Divider
    ax.plot([x, x], [y - node_radius*0.7, y + node_radius*0.7],
            'k-', linewidth=1, zorder=11)

    # Sigma and f
    ax.text(x - node_radius*0.35, y, r'$\Sigma$', fontsize=10,
            ha='center', va='center', zorder=12)
    ax.text(x + node_radius*0.35, y, r'$f$', fontsize=9,
            ha='center', va='center', zorder=12, style='italic')

    # Label to the right
    if show_activations and self.a2 is not None:
      label_text = f"$\\hat{{y}}$={self.a2[0]:.2f}"
    else:
      label_text = r"$\hat{y}$"
    ax.text(x + 0.10, y, label_text, fontsize=9, ha='left', va='center')

    # Set axis limits - tight to the content
    ax.set_xlim(-0.05, 1.05)
    if self.use_bias:
      ax.set_ylim(0.05, 0.95)
    else:
      ax.set_ylim(0.1, 0.9)

    # Save to buffer with minimal padding
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150,
                facecolor='white', edgecolor='none', pad_inches=0.1)
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
      self.answers[key] = Answer.float_value(key, float(self.a1[i]))

    # Output
    self.answers["y_pred"] = Answer.float_value("y_pred", float(self.a2[0]))

  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()

    # Question description
    body.add_element(ContentAST.Paragraph([
      f"Given the neural network below with {self._get_activation_name()} activation "
      f"in the hidden layer, calculate the forward pass for the given input values."
    ]))

    # Network diagram
    body.add_element(
      ContentAST.Picture(
        img_data=self._generate_network_diagram(show_weights=True, show_activations=False),
        caption=f"Neural network with {self._get_activation_name()} activation"
      )
    )

    # Input values
    input_vals = []
    for i in range(self.num_inputs):
      if i > 0:
        input_vals.append(", ")
      input_vals.append(ContentAST.Equation(f"x_{i+1} = {self.X[i]:.1f}", inline=True))

    body.add_element(ContentAST.Paragraph(["Input values: "] + input_vals))

    # Activation function formula
    body.add_element(ContentAST.Paragraph([
      f"Activation function: ",
      ContentAST.Equation(self._get_activation_formula(), inline=True)
    ]))

    # Answer table
    body.add_element(ContentAST.Paragraph([
      "Calculate the following values:"
    ]))

    # Create table for answers
    table_data = []
    table_data.append(["Value", "Your Answer"])

    for i in range(self.num_hidden):
      table_data.append([
        ContentAST.Paragraph([ContentAST.Equation(f"h_{i+1}", inline=True), f" (hidden neuron {i+1} output)"]),
        ContentAST.Answer(self.answers[f"h{i+1}"])
      ])

    table_data.append([
      ContentAST.Paragraph([ContentAST.Equation(r"\hat{y}", inline=True), " (network output)"]),
      ContentAST.Answer(self.answers["y_pred"])
    ])

    body.add_element(ContentAST.Table(data=table_data))

    return body

  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph([
      "To solve this problem, we need to compute the forward pass through the network."
    ]))

    # Hidden layer calculations
    explanation.add_element(ContentAST.Paragraph([
      "**Step 1: Calculate hidden layer pre-activations**"
    ]))

    for i in range(self.num_hidden):
      # Build equation for z_i
      terms = []
      for j in range(self.num_inputs):
        terms.append(f"({self.W1[i,j]:.1f})({self.X[j]:.1f})")

      z_calc = " + ".join(terms)
      if self.use_bias:
        z_calc += f" + {self.b1[i]:.1f}"

      explanation.add_element(ContentAST.Equation(
        f"z_{i+1} = {z_calc} = {self.z1[i]:.4f}",
        inline=False
      ))

    # Hidden layer activations
    explanation.add_element(ContentAST.Paragraph([
      f"**Step 2: Apply {self._get_activation_name()} activation**"
    ]))

    for i in range(self.num_hidden):
      if self.activation_function == self.ACTIVATION_SIGMOID:
        explanation.add_element(ContentAST.Equation(
          f"h_{i+1} = \\sigma(z_{i+1}) = \\frac{{1}}{{1 + e^{{-{self.z1[i]:.4f}}}}} = {self.a1[i]:.4f}",
          inline=False
        ))
      elif self.activation_function == self.ACTIVATION_RELU:
        explanation.add_element(ContentAST.Equation(
          f"h_{i+1} = \\text{{ReLU}}(z_{i+1}) = \\max(0, {self.z1[i]:.4f}) = {self.a1[i]:.4f}",
          inline=False
        ))
      else:
        explanation.add_element(ContentAST.Equation(
          f"h_{i+1} = z_{i+1} = {self.a1[i]:.4f}",
          inline=False
        ))

    # Output layer
    explanation.add_element(ContentAST.Paragraph([
      "**Step 3: Calculate output**"
    ]))

    terms = []
    for j in range(self.num_hidden):
      terms.append(f"({self.W2[0,j]:.1f})({self.a1[j]:.4f})")

    z_out_calc = " + ".join(terms)
    if self.use_bias:
      z_out_calc += f" + {self.b2[0]:.1f}"

    explanation.add_element(ContentAST.Equation(
      f"z_{{out}} = {z_out_calc} = {self.z2[0]:.4f}",
      inline=False
    ))

    explanation.add_element(ContentAST.Equation(
      f"\\hat{{y}} = z_{{out}} = {self.a2[0]:.4f}",
      inline=False
    ))

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

    # Generate target and compute loss
    # Target should be different from output to create meaningful gradients
    self.y_target = float(self.a2[0] + self.rng.uniform(1, 3) * self.rng.choice([-1, 1]))
    self._compute_loss(self.y_target)
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
      self.answers[key] = Answer.auto_float(key, self._compute_gradient_W2(i))

    # Gradient for W1 (input to hidden) - pick first hidden neuron
    for j in range(self.num_inputs):
      key = f"dL_dw1_0{j}"
      self.answers[key] = Answer.auto_float(key, self._compute_gradient_W1(0, j))

  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()

    # Question description
    body.add_element(ContentAST.Paragraph([
      f"Given the neural network below with {self._get_activation_name()} activation "
      f"in the hidden layer, a forward pass has been completed with the values shown. "
      f"Calculate the gradients (∂L/∂w) for the specified weights using backpropagation."
    ]))

    # Network diagram with activations
    body.add_element(
      ContentAST.Picture(
        img_data=self._generate_network_diagram(show_weights=True, show_activations=True),
        caption=f"Neural network with computed activations"
      )
    )

    # Show forward pass results
    body.add_element(ContentAST.Paragraph([
      "**Forward pass results:**"
    ]))

    body.add_element(ContentAST.Paragraph([
      "Target: ",
      ContentAST.Equation(f"y = {self.y_target:.2f}", inline=True)
    ]))

    body.add_element(ContentAST.Paragraph([
      "Prediction: ",
      ContentAST.Equation(f"\\hat{{y}} = {self.a2[0]:.4f}", inline=True)
    ]))

    body.add_element(ContentAST.Paragraph([
      "Loss (MSE): ",
      ContentAST.Equation(f"L = (1/2)(y - \\hat{{y}})^2 = {self.loss:.4f}", inline=True)
    ]))

    # Activation function reminder
    body.add_element(ContentAST.Paragraph([
      f"Activation function: ",
      ContentAST.Equation(self._get_activation_formula(), inline=True)
    ]))

    # Answer table
    body.add_element(ContentAST.Paragraph([
      "Calculate the following gradients:"
    ]))

    table_data = []
    table_data.append(["Gradient", "Description", "Your Answer"])

    # W2 gradients
    for i in range(self.num_hidden):
      table_data.append([
        ContentAST.Equation(f"∂L / ∂w_{i+3}", inline=True),
        ContentAST.Paragraph([f"Weight from ", ContentAST.Equation(f"h_{i+1}", inline=True), " to output"]),
        ContentAST.Answer(self.answers[f"dL_dw2_{i}"])
      ])

    # W1 gradients (first hidden neuron)
    for j in range(self.num_inputs):
      table_data.append([
        ContentAST.Equation(f"∂L / ∂w_{{1{j+1}}}", inline=True),
        ContentAST.Paragraph([f"Weight from ", ContentAST.Equation(f"x_{j+1}", inline=True), " to ", ContentAST.Equation("h_1", inline=True)]),
        ContentAST.Answer(self.answers[f"dL_dw1_0{j}"])
      ])

    body.add_element(ContentAST.Table(data=table_data))

    return body

  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph([
      "To solve this problem, we use the chain rule to compute gradients via backpropagation."
    ]))

    # Output layer gradient
    explanation.add_element(ContentAST.Paragraph([
      "**Step 1: Compute output layer gradient**"
    ]))

    explanation.add_element(ContentAST.Paragraph([
      "For MSE loss with linear output activation:"
    ]))

    explanation.add_element(ContentAST.Equation(
      f"\\frac{{\\partial L}}{{\\partial \\hat{{y}}}} = -(y - \\hat{{y}}) = -({self.y_target:.2f} - {self.a2[0]:.4f}) = {self.dL_da2:.4f}",
      inline=False
    ))

    # W2 gradients
    explanation.add_element(ContentAST.Paragraph([
      "**Step 2: Gradients for hidden-to-output weights**"
    ]))

    explanation.add_element(ContentAST.Paragraph([
      "Using the chain rule:"
    ]))

    for i in range(self.num_hidden):
      grad = self._compute_gradient_W2(i)
      explanation.add_element(ContentAST.Equation(
        f"\\frac{{\\partial L}}{{\\partial w_{i+3}}} = \\frac{{\\partial L}}{{\\partial \\hat{{y}}}} \\cdot \\frac{{\\partial \\hat{{y}}}}{{\\partial w_{i+3}}} = {self.dL_da2:.4f} \\cdot {self.a1[i]:.4f} = {grad:.4f}",
        inline=False
      ))

    # W1 gradients
    explanation.add_element(ContentAST.Paragraph([
      "**Step 3: Gradients for input-to-hidden weights**"
    ]))

    explanation.add_element(ContentAST.Paragraph([
      "First, compute the gradient flowing back to hidden layer:"
    ]))

    for j in range(self.num_inputs):
      # Compute intermediate values
      dz2_da1 = self.W2[0, 0]
      da1_dz1 = self._activation_derivative(self.z1[0])
      dL_dz1 = self.dL_dz2 * dz2_da1 * da1_dz1

      grad = self._compute_gradient_W1(0, j)

      if self.activation_function == self.ACTIVATION_SIGMOID:
        act_deriv_str = f"\\sigma(z_1)(1-\\sigma(z_1)) = {self.a1[0]:.4f}(1-{self.a1[0]:.4f}) = {da1_dz1:.4f}"
      elif self.activation_function == self.ACTIVATION_RELU:
        act_deriv_str = f"\\mathbb{{1}}(z_1 > 0) = {da1_dz1:.4f}"
      else:
        act_deriv_str = f"1"

      explanation.add_element(ContentAST.Equation(
        f"\\frac{{\\partial L}}{{\\partial w_{{1{j+1}}}}} = \\frac{{\\partial L}}{{\\partial \\hat{{y}}}} \\cdot w_{3} \\cdot {act_deriv_str} \\cdot x_{j+1} = {self.dL_da2:.4f} \\cdot {dz2_da1:.4f} \\cdot {da1_dz1:.4f} \\cdot {self.X[j]:.1f} = {grad:.4f}",
        inline=False
      ))

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
    self.answers["mean"] = Answer.float_value("mean", float(mean_pred))

    # Median (optional, but useful)
    median_pred = np.median(self.predictions)
    self.answers["median"] = Answer.float_value("median", float(median_pred))

  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()

    # Question description
    body.add_element(ContentAST.Paragraph([
      f"You have trained {self.num_models} different regression models on the same dataset. "
      f"For a particular test input, each model produces the following predictions:"
    ]))

    # Show predictions
    pred_list = ", ".join([f"{p:.1f}" for p in self.predictions])
    body.add_element(ContentAST.Paragraph([
      f"Model predictions: {pred_list}"
    ]))

    # Question
    body.add_element(ContentAST.Paragraph([
      "To create an ensemble, calculate the combined prediction using the following methods:"
    ]))

    # Answer table
    table_data = []
    table_data.append(["Method", "Combined Prediction"])

    table_data.append([
      "Mean (average)",
      ContentAST.Answer(self.answers["mean"])
    ])

    table_data.append([
      "Median",
      ContentAST.Answer(self.answers["median"])
    ])

    body.add_element(ContentAST.Table(data=table_data))

    return body

  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph([
      "Ensemble methods combine predictions from multiple models to create a more robust prediction."
    ]))

    # Mean calculation
    explanation.add_element(ContentAST.Paragraph([
      "**Mean (Bagging approach):**"
    ]))

    pred_sum = " + ".join([f"{p:.1f}" for p in self.predictions])
    mean_val = np.mean(self.predictions)

    explanation.add_element(ContentAST.Equation(
      f"\\text{{mean}} = \\frac{{{pred_sum}}}{{{self.num_models}}} = \\frac{{{sum(self.predictions):.1f}}}{{{self.num_models}}} = {mean_val:.4f}",
      inline=False
    ))

    # Median calculation
    explanation.add_element(ContentAST.Paragraph([
      "**Median:**"
    ]))

    sorted_preds = sorted(self.predictions)
    sorted_str = ", ".join([f"{p:.1f}" for p in sorted_preds])
    median_val = np.median(self.predictions)

    explanation.add_element(ContentAST.Paragraph([
      f"Sorted predictions: {sorted_str}"
    ]))

    if self.num_models % 2 == 1:
      mid_idx = self.num_models // 2
      explanation.add_element(ContentAST.Paragraph([
        f"Middle value (position {mid_idx + 1}): {median_val:.1f}"
      ]))
    else:
      mid_idx1 = self.num_models // 2 - 1
      mid_idx2 = self.num_models // 2
      explanation.add_element(ContentAST.Paragraph([
        f"Average of middle two values (positions {mid_idx1 + 1} and {mid_idx2 + 1}): "
        f"({sorted_preds[mid_idx1]:.1f} + {sorted_preds[mid_idx2]:.1f}) / 2 = {median_val:.1f}"
      ]))

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

    # Generate target and compute loss
    self.y_target = float(self.a2[0] + self.rng.uniform(1, 3) * self.rng.choice([-1, 1]))
    self._compute_loss(self.y_target)
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
    self.answers["y_pred"] = Answer.float_value("y_pred", float(self.a2[0]))

    # Loss answer
    self.answers["loss"] = Answer.float_value("loss", float(self.loss))

    # Gradient answers (for key weights)
    self.answers["grad_w3"] = Answer.auto_float("grad_w3", self._compute_gradient_W2(0))
    self.answers["grad_w11"] = Answer.auto_float("grad_w11", self._compute_gradient_W1(0, 0))

    # Updated weight answers
    self.answers["new_w3"] = Answer.float_value("new_w3", float(self.new_W2[0, 0]))
    self.answers["new_w11"] = Answer.float_value("new_w11", float(self.new_W1[0, 0]))

  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()

    # Question description
    body.add_element(ContentAST.Paragraph([
      f"Given the neural network below, perform one complete training step (forward pass, "
      f"loss calculation, backpropagation, and weight update) for the given input and target."
    ]))

    # Network diagram
    body.add_element(
      ContentAST.Picture(
        img_data=self._generate_network_diagram(show_weights=True, show_activations=False),
        caption=f"Neural network (before training)"
      )
    )

    # Training parameters
    body.add_element(ContentAST.Paragraph([
      "**Training parameters:**"
    ]))

    body.add_element(ContentAST.Paragraph([
      "Input: ",
      ContentAST.Equation(f"x_1 = {self.X[0]:.1f}", inline=True),
      ", ",
      ContentAST.Equation(f"x_2 = {self.X[1]:.1f}", inline=True)
    ]))

    body.add_element(ContentAST.Paragraph([
      "Target: ",
      ContentAST.Equation(f"y = {self.y_target:.2f}", inline=True)
    ]))

    body.add_element(ContentAST.Paragraph([
      "Learning rate: ",
      ContentAST.Equation(f"\\alpha = {self.learning_rate}", inline=True)
    ]))

    body.add_element(ContentAST.Paragraph([
      f"Activation: {self._get_activation_name()} - ",
      ContentAST.Equation(self._get_activation_formula(), inline=True)
    ]))

    # Answer table
    body.add_element(ContentAST.Paragraph([
      "Calculate the following values for this training step:"
    ]))

    table_data = []
    table_data.append(["Step", "Calculation", "Your Answer"])

    table_data.append([
      "1. Forward Pass",
      ContentAST.Paragraph(["Network output ", ContentAST.Equation(r"\hat{y}", inline=True)]),
      ContentAST.Answer(self.answers["y_pred"])
    ])

    table_data.append([
      "2. Loss",
      ContentAST.Paragraph(["MSE: ", ContentAST.Equation(r"L = (1/2)(y - \hat{y})^2", inline=True)]),
      ContentAST.Answer(self.answers["loss"])
    ])

    table_data.append([
      "3. Gradient",
      ContentAST.Paragraph([ContentAST.Equation(r"∂L / ∂w_3", inline=True), " (weight ", ContentAST.Equation("h_1", inline=True), " ", ContentAST.Equation(r"\to", inline=True), " ", ContentAST.Equation(r"\hat{y}", inline=True), ")"]),
      ContentAST.Answer(self.answers["grad_w3"])
    ])

    table_data.append([
      "4. Gradient",
      ContentAST.Paragraph([ContentAST.Equation(r"∂L / ∂w_{11}", inline=True), " (weight ", ContentAST.Equation("x_1", inline=True), " ", ContentAST.Equation(r"\to", inline=True), " ", ContentAST.Equation("h_1", inline=True), ")"]),
      ContentAST.Answer(self.answers["grad_w11"])
    ])

    table_data.append([
      "5. Update",
      ContentAST.Paragraph(["New ", ContentAST.Equation("w_3", inline=True), ": ", ContentAST.Equation(r"w_3' = w_3 - α(∂L / ∂w_3)", inline=True)]),
      ContentAST.Answer(self.answers["new_w3"])
    ])

    table_data.append([
      "6. Update",
      ContentAST.Paragraph(["New ", ContentAST.Equation("w_{11}", inline=True), " after update"]),
      ContentAST.Answer(self.answers["new_w11"])
    ])

    body.add_element(ContentAST.Table(data=table_data))

    return body

  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()

    explanation.add_element(ContentAST.Paragraph([
      "This problem requires performing one complete training iteration. Let's go through each step."
    ]))

    # Step 1: Forward pass
    explanation.add_element(ContentAST.Paragraph([
      "**Step 1: Forward Pass**"
    ]))

    # Hidden layer
    z1_0 = self.W1[0, 0] * self.X[0] + self.W1[0, 1] * self.X[1] + self.b1[0]
    explanation.add_element(ContentAST.Equation(
      f"z_1 = w_{{11}} x_1 + w_{{12}} x_2 + b_1 = {self.W1[0,0]:.1f} \\cdot {self.X[0]:.1f} + {self.W1[0,1]:.1f} \\cdot {self.X[1]:.1f} + {self.b1[0]:.1f} = {self.z1[0]:.4f}",
      inline=False
    ))

    explanation.add_element(ContentAST.Equation(
      f"h_1 = {self._get_activation_name()}(z_1) = {self.a1[0]:.4f}",
      inline=False
    ))

    # Similarly for h2 (abbreviated)
    explanation.add_element(ContentAST.Equation(
      f"h_2 = {self.a1[1]:.4f} \\text{{ (calculated similarly)}}",
      inline=False
    ))

    # Output
    z2 = self.W2[0, 0] * self.a1[0] + self.W2[0, 1] * self.a1[1] + self.b2[0]
    explanation.add_element(ContentAST.Equation(
      f"\\hat{{y}} = w_3 h_1 + w_4 h_2 + b_2 = {self.W2[0,0]:.1f} \\cdot {self.a1[0]:.4f} + {self.W2[0,1]:.1f} \\cdot {self.a1[1]:.4f} + {self.b2[0]:.1f} = {self.a2[0]:.4f}",
      inline=False
    ))

    # Step 2: Loss
    explanation.add_element(ContentAST.Paragraph([
      "**Step 2: Calculate Loss**"
    ]))

    explanation.add_element(ContentAST.Equation(
      f"L = \\frac{{1}}{{2}}(y - \\hat{{y}})^2 = \\frac{{1}}{{2}}({self.y_target:.2f} - {self.a2[0]:.4f})^2 = {self.loss:.4f}",
      inline=False
    ))

    # Step 3: Gradients
    explanation.add_element(ContentAST.Paragraph([
      "**Step 3: Compute Gradients**"
    ]))

    explanation.add_element(ContentAST.Paragraph([
      "Loss gradient:"
    ]))

    explanation.add_element(ContentAST.Equation(
      f"\\frac{{\\partial L}}{{\\partial \\hat{{y}}}} = -(y - \\hat{{y}}) = {self.dL_da2:.4f}",
      inline=False
    ))

    grad_w3 = self._compute_gradient_W2(0)
    explanation.add_element(ContentAST.Equation(
      f"\\frac{{\\partial L}}{{\\partial w_3}} = \\frac{{\\partial L}}{{\\partial \\hat{{y}}}} \\cdot h_1 = {self.dL_da2:.4f} \\cdot {self.a1[0]:.4f} = {grad_w3:.4f}",
      inline=False
    ))

    grad_w11 = self._compute_gradient_W1(0, 0)
    dz2_da1 = self.W2[0, 0]
    da1_dz1 = self._activation_derivative(self.z1[0])

    explanation.add_element(ContentAST.Equation(
      f"\\frac{{\\partial L}}{{\\partial w_{{11}}}} = \\frac{{\\partial L}}{{\\partial \\hat{{y}}}} \\cdot w_3 \\cdot \\sigma'(z_1) \\cdot x_1 = {self.dL_da2:.4f} \\cdot {dz2_da1:.4f} \\cdot {da1_dz1:.4f} \\cdot {self.X[0]:.1f} = {grad_w11:.4f}",
      inline=False
    ))

    # Step 4: Weight updates
    explanation.add_element(ContentAST.Paragraph([
      "**Step 4: Update Weights**"
    ]))

    new_w3 = self.new_W2[0, 0]
    explanation.add_element(ContentAST.Equation(
      f"w_3^{{new}} = w_3 - \\alpha \\frac{{\\partial L}}{{\\partial w_3}} = {self.W2[0,0]:.1f} - {self.learning_rate} \\cdot {grad_w3:.4f} = {new_w3:.4f}",
      inline=False
    ))

    new_w11 = self.new_W1[0, 0]
    explanation.add_element(ContentAST.Equation(
      f"w_{{11}}^{{new}} = w_{{11}} - \\alpha \\frac{{\\partial L}}{{\\partial w_{{11}}}} = {self.W1[0,0]:.1f} - {self.learning_rate} \\cdot {grad_w11:.4f} = {new_w11:.4f}",
      inline=False
    ))

    explanation.add_element(ContentAST.Paragraph([
      "These updated weights would be used in the next training iteration."
    ]))

    return explanation
