from __future__ import annotations

import abc
import io
import logging

import matplotlib.pyplot as plt
import numpy as np

import QuizGenerator.generation.contentast as ca
from QuizGenerator.generation.question import Question, QuestionRegistry

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

  # Activation function types.  The hidden layer is intentionally fixed to
  # ReLU for these introductory exercises; sigmoid remains the binary output
  # activation.
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

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    self = context

    self.num_inputs = kwargs.get("num_inputs", getattr(self, "num_inputs", 2))
    self.num_hidden = kwargs.get("num_hidden", getattr(self, "num_hidden", 2))
    self.num_outputs = kwargs.get("num_outputs", getattr(self, "num_outputs", 1))
    self.use_bias = kwargs.get("use_bias", getattr(self, "use_bias", True))
    self.param_digits = kwargs.get("param_digits", getattr(self, "param_digits", 1))

    self.rng.seed(rng_seed)
    self._np_rng = np.random.RandomState(rng_seed)
    return context

  def _generate_network(self, weight_range=(-2, 2), input_range=(-3, 3)):
    """Generate random network parameters and input."""
    # Generate weights using MatrixQuestion's rounded matrix method
    # Use param_digits to match display precision in tables and explanations
    self.W1 = self.get_rounded_matrix(
      self._np_rng,
      (self.num_hidden, self.num_inputs),
      weight_range[0],
      weight_range[1],
      self.param_digits
    )

    self.W2 = self.get_rounded_matrix(
      self._np_rng,
      (self.num_outputs, self.num_hidden),
      weight_range[0],
      weight_range[1],
      self.param_digits
    )

    # Generate biases
    if self.use_bias:
      self.b1 = self.get_rounded_matrix(
        self._np_rng,
        (self.num_hidden,),
        weight_range[0],
        weight_range[1],
        self.param_digits
      )
      self.b2 = self.get_rounded_matrix(
        self._np_rng,
        (self.num_outputs,),
        weight_range[0],
        weight_range[1],
        self.param_digits
      )
    else:
      self.b1 = np.zeros(self.num_hidden)
      self.b2 = np.zeros(self.num_outputs)

    # Generate input values (keep as integers for simplicity)
    self.X = self.get_rounded_matrix(
      self._np_rng,
      (self.num_inputs,),
      input_range[0],
      input_range[1],
      0  # Round to integers
    )

  def _select_activation_function(self):
    """Use ReLU in the hidden layer for all introductory MLP exercises."""
    self.activation_function = self.ACTIVATION_RELU

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

  def _forward_pass(self, *, round_values=True):
    """Run forward pass through the network.

    Args:
      round_values: Whether to replace the computed values with their displayed
        four-decimal approximations.  A completed-pass exercise uses those
        displayed values as its inputs; a compute-it-yourself exercise retains
        calculator precision until its final answers are rounded.
    """
    # Hidden layer
    self.z1 = self.W1 @ self.X + self.b1
    self.a1 = self._apply_activation(self.z1)

    # Output layer
    self.z2 = self.W2 @ self.a1 + self.b2
    self.a2 = self._apply_activation(self.z2, self.ACTIVATION_SIGMOID)  # Sigmoid output for binary classification

    if round_values:
      # A completed-forward-pass problem treats these displayed values as the
      # authoritative inputs to subsequent calculations.
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

  def _hidden_activation_derivative(self, hidden_idx):
    """Return the derivative for a hidden activation used in backpropagation."""
    if self.activation_function == self.ACTIVATION_SIGMOID:
      # In a completed-pass exercise, h is a supplied, rounded value.  Use it
      # directly so the gradient is reproducible from the displayed table.
      return self.a1[hidden_idx] * (1 - self.a1[hidden_idx])
    return self._activation_derivative(self.z1[hidden_idx])

  def _compute_gradient_W1(self, hidden_idx, input_idx):
    """Compute gradient ∂L/∂W1[hidden_idx, input_idx]."""
    # dL/dz1[hidden_idx] = dL/dz2 * ∂z2/∂a1[hidden_idx] * ∂a1/∂z1[hidden_idx]
    #                     = dL/dz2 * W2[0, hidden_idx] * activation'(z1[hidden_idx])

    dz2_da1 = self.W2[0, hidden_idx]
    da1_dz1 = self._hidden_activation_derivative(hidden_idx)

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
          ca.Equation(f"h_{{\\mathrm{{pre}},{i+1}}} = z_{i+1}", inline=True),
          f"{self.z1[i]:.4f}"
        ])
        right_data.append([
          ca.Equation(f"h_{i+1}", inline=True),
          f"{self.a1[i]:.4f}"
        ])

    # Binary-classifier output values (if computed and requested)
    if include_activations and self.a2 is not None:
      right_data.append([
        ca.Equation(r"z_{out} 	ext{(logit)}", inline=True),
        f"{self.z2[0]:.4f}"
      ])
      right_data.append([
        ca.Equation(r"\hat{y} = P(y=1)", inline=True),
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
    # Create figure with space for the binary-classification output path.
    fig = plt.figure(figsize=(10, 2.8))
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
      ax.text(hidden_x, y, f'$h_{{{i+1}}}$', fontsize=10, ha='center', va='center', zorder=12)
      if show_activations and self.a1 is not None:
        ax.text(hidden_x, y - r - 0.15, f'{self.a1[i]:.2f}', fontsize=8, ha='center', va='top')

    ax.text(hidden_x, min(hidden_y) - 0.45, 'Hidden layer (ReLU)',
            fontsize=9, ha='center', va='center', color='#12355b')

    # Output node
    y = output_y[0]
    circle = plt.Circle((output_x, y), r, facecolor='lightblue',
                       edgecolor='black', linewidth=1.5, zorder=10)
    ax.add_patch(circle)
    label = r'$\hat{y}$' if not show_activations else f'$\\hat{{y}}$={self.a2[0]:.2f}'
    ax.text(output_x, y, label, fontsize=10, ha='center', va='center', zorder=12)

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
    lines.append("Input Layer:     Hidden Layer (ReLU):      Output:")

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
    lines.append("Hidden activation: ReLU")
    lines.append("Output: yhat (sigmoid activation for binary classification)")

    return "\n".join(lines)


@QuestionRegistry.register()
class ForwardPassQuestion(SimpleNeuralNetworkBase):
  """
  Question asking students to calculate forward pass through a simple network.

  Students calculate:
  - Hidden layer activations (h₁, h₂)
  - Final output (ŷ)
  """

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    self = context

    # Generate network
    self._generate_network()
    self._select_activation_function()

    # Run forward pass to get correct answers
    self._forward_pass()
    return context

  @classmethod
  def _build_body(cls, context) -> tuple[ca.Section, list[ca.Answer]]:
    """Build question body and collect answers."""
    self = context
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

    for i in range(self.num_hidden):
      answers.append(ca.AnswerTypes.Float(float(self.a1[i]), label=f"h_{i + 1}"))

    answers.append(ca.AnswerTypes.Float(float(self.a2[0]), label="ŷ"))

    body.add_element(ca.AnswerBlock(answers))

    return body, answers

  @classmethod
  def _build_explanation(cls, context) -> tuple[ca.Section, list[ca.Answer]]:
    """Build question explanation."""
    self = context
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


@QuestionRegistry.register()
class BackpropGradientQuestion(SimpleNeuralNetworkBase):
  """
  Question asking students to calculate gradients using backpropagation.

  Given a completed forward pass, students calculate:
  - Gradients for multiple specific weights (∂L/∂w)
  """

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    self = context

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
    return context

  @classmethod
  def is_interesting_ctx(cls, context) -> bool:
    """Reject ReLU networks whose hidden layer is entirely inactive.

    When every ReLU hidden unit is inactive, every weight gradient requested by
    this exercise is zero.  Let ``Question.instantiate`` advance the seed and
    generate an example that actually practices the chain rule instead.
    """
    return (
      super().is_interesting_ctx(context)
      and not (
        context.activation_function == cls.ACTIVATION_RELU
        and np.all(context.a1 == 0)
      )
    )

  @classmethod
  def _build_body(cls, context) -> tuple[ca.Section, list[ca.Answer]]:
    """Build question body and collect answers."""
    self = context
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
      "Use binary cross-entropy for the loss: ",
      ca.Equation(r"L = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]", inline=True),
      "."
    ]))

    body.add_element(ca.Paragraph([
      "**Calculate the following gradients:**"
    ]))

    for i in range(self.num_hidden):
      answers.append(ca.AnswerTypes.Float(
        self._compute_gradient_W2(i),
        label=f"∂L/∂w_{i + 3}"
      ))

    for j in range(self.num_inputs):
      answers.append(ca.AnswerTypes.Float(
        self._compute_gradient_W1(0, j),
        label=f"∂L/∂w_1{j + 1}"
      ))

    body.add_element(ca.AnswerBlock(answers))

    return body, answers

  @classmethod
  def _build_explanation(cls, context) -> tuple[ca.Section, list[ca.Answer]]:
    """Build question explanation."""
    self = context
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
      "First, compute the activation derivative and the gradient flowing back to the first hidden unit:"
    ]))

    dz2_da1 = self.W2[0, 0]
    da1_dz1 = self._hidden_activation_derivative(0)
    dL_dz1 = self.dL_dz2 * dz2_da1 * da1_dz1

    if self.activation_function == self.ACTIVATION_SIGMOID:
      explanation.add_element(ca.Equation(
        f"\\sigma'(z_1) = h_1(1-h_1) = {self.a1[0]:.4f}(1-{self.a1[0]:.4f}) = {da1_dz1:.4f}",
        inline=False
      ))
    elif self.activation_function == self.ACTIVATION_RELU:
      explanation.add_element(ca.Equation(
        f"\\text{{ReLU}}'(z_1) = \\mathbb{{1}}(z_1 > 0) = {da1_dz1:.4f}",
        inline=False
      ))

    explanation.add_element(ca.Equation(
      f"\\frac{{\\partial L}}{{\\partial z_1}} = \\frac{{\\partial L}}{{\\partial z_{{out}}}} \\cdot w_3 \\cdot f'(z_1) = {self.dL_dz2:.4f} \\cdot {dz2_da1:.4f} \\cdot {da1_dz1:.4f} = {dL_dz1:.4f}",
      inline=False
    ))

    for j in range(self.num_inputs):
      grad = self._compute_gradient_W1(0, j)
      explanation.add_element(ca.Equation(
        f"\\frac{{\\partial L}}{{\\partial w_{{1{j+1}}}}} = \\frac{{\\partial L}}{{\\partial z_1}} \\cdot x_{j+1} = {dL_dz1:.4f} \\cdot {self.X[j]:.1f} = {grad:.4f}",
        inline=False
      ))

    return explanation, []


@QuestionRegistry.register()
class TwoClassSoftmaxBackpropQuestion(SimpleNeuralNetworkBase):
  """Backpropagation through a ReLU hidden layer and two-class softmax output."""

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    # This is deliberately a two-logit classifier.  Do not allow a config
    # value to silently turn it back into the one-logit sigmoid exercise.
    context_kwargs = dict(kwargs)
    context_kwargs["num_outputs"] = 2
    context = super()._build_context(rng_seed=rng_seed, **context_kwargs)
    self = context

    self._generate_network()
    self._select_activation_function()
    self._forward_pass()

    # Choose the class other than the model's current prediction so the
    # softmax error signal is nontrivial for both output logits.
    self.y_target = 1 - int(np.argmax(self.a2))
    self.y_one_hot = np.zeros(2)
    self.y_one_hot[self.y_target] = 1
    self._compute_loss(self.y_target)
    self.loss = round(self.loss, 4)
    self._compute_output_gradient()
    return context

  def _forward_pass(self, *, round_values=True):
    """Run a ReLU-hidden, two-logit softmax forward pass."""
    self.z1 = self.W1 @ self.X + self.b1
    self.a1 = self._apply_activation(self.z1)
    self.z2 = self.W2 @ self.a1 + self.b2  # two output logits

    shifted_logits = self.z2 - np.max(self.z2)
    exp_logits = np.exp(shifted_logits)
    self.a2 = exp_logits / np.sum(exp_logits)

    if round_values:
      self.z1 = np.round(self.z1, 4)
      self.a1 = np.round(self.a1, 4)
      self.z2 = np.round(self.z2, 4)
      self.a2 = np.round(self.a2, 4)
    return self.a2

  def _compute_loss(self, y_target):
    """Compute cross-entropy loss for the selected target class."""
    self.y_target = y_target
    epsilon = 1e-15
    self.loss = -np.log(np.clip(self.a2[y_target], epsilon, 1 - epsilon))
    return self.loss

  def _compute_output_gradient(self):
    """Compute the softmax-plus-cross-entropy gradient for both logits."""
    self.dL_dz2 = self.a2 - self.y_one_hot
    return self.dL_dz2

  def _compute_gradient_W2(self, output_idx, hidden_idx):
    """Compute ∂L/∂W2[output_idx, hidden_idx]."""
    return float(self.dL_dz2[output_idx] * self.a1[hidden_idx])

  def _compute_hidden_gradient(self, hidden_idx):
    """Compute ∂L/∂h[hidden_idx] by collecting both output contributions."""
    return float(np.dot(self.W2[:, hidden_idx], self.dL_dz2))

  def _compute_gradient_W1(self, hidden_idx, input_idx):
    """Compute ∂L/∂W1[hidden_idx, input_idx]."""
    dL_dh = self._compute_hidden_gradient(hidden_idx)
    dL_dz1 = dL_dh * self._hidden_activation_derivative(hidden_idx)
    return float(dL_dz1 * self.X[input_idx])

  @classmethod
  def is_interesting_ctx(cls, context) -> bool:
    """Require a live first hidden unit and nonzero requested gradients."""
    first_hidden_live = context.a1[0] > 0
    requested_gradients = [
      context._compute_gradient_W2(0, 0),
      context._compute_gradient_W2(1, 0),
      context._compute_gradient_W1(0, 0),
    ]
    return (
      super().is_interesting_ctx(context)
      and first_hidden_live
      and all(abs(gradient) > 1e-10 for gradient in requested_gradients)
    )

  def _generate_parameter_table(self):
    """Show scalar edge weights alongside the two-logit forward pass."""
    left_data = [["Symbol", "Value"]]
    for i in range(self.num_inputs):
      left_data.append([ca.Equation(f"x_{i+1}", inline=True), f"{self.X[i]:.1f}"])
    for hidden_idx in range(self.num_hidden):
      for input_idx in range(self.num_inputs):
        left_data.append([
          ca.Equation(f"w^{{(1)}}_{{{hidden_idx+1},{input_idx+1}}}", inline=True),
          f"{self.W1[hidden_idx, input_idx]:.{self.param_digits}f}"
        ])
    for output_idx in range(2):
      for hidden_idx in range(self.num_hidden):
        left_data.append([
          ca.Equation(f"w^{{(2)}}_{{{output_idx+1},{hidden_idx+1}}}", inline=True),
          f"{self.W2[output_idx, hidden_idx]:.{self.param_digits}f}"
        ])

    right_data = [["Symbol", "Value"]]
    for hidden_idx in range(self.num_hidden):
      right_data.append([
        ca.Equation(f"b^{{(1)}}_{hidden_idx+1}", inline=True),
        f"{self.b1[hidden_idx]:.{self.param_digits}f}"
      ])
    for output_idx in range(2):
      right_data.append([
        ca.Equation(f"b^{{(2)}}_{output_idx+1}", inline=True),
        f"{self.b2[output_idx]:.{self.param_digits}f}"
      ])
    for hidden_idx in range(self.num_hidden):
      right_data.append([
        ca.Equation(f"h_{{\\mathrm{{pre}},{hidden_idx+1}}}", inline=True),
        f"{self.z1[hidden_idx]:.4f}"
      ])
      right_data.append([
        ca.Equation(f"h_{hidden_idx+1}", inline=True),
        f"{self.a1[hidden_idx]:.4f}"
      ])
    for output_idx in range(2):
      right_data.append([
        ca.Equation(f"o_{output_idx+1}", inline=True),
        f"{self.z2[output_idx]:.4f}"
      ])
      right_data.append([
        ca.Equation(f"\\hat{{y}}_{output_idx+1}", inline=True),
        f"{self.a2[output_idx]:.4f}"
      ])
    target_str = ", ".join(str(int(value)) for value in self.y_one_hot)
    right_data.append([ca.Equation("y", inline=True), f"[{target_str}]"])
    right_data.append([ca.Equation("L", inline=True), f"{self.loss:.4f}"])

    table_group = ca.TableGroup()
    table_group.add_table(ca.Table(data=left_data))
    table_group.add_table(ca.Table(data=right_data))
    return table_group

  def _generate_network_diagram(self):
    """Draw a two-logit network with individually labeled output edges."""
    # Keep the image compact enough that Canvas does not downscale its labels.
    fig = plt.figure(figsize=(8, 2.75))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    input_x, hidden_x, output_x = 0.4, 1.65, 2.9
    input_y = hidden_y = output_y = [1.3, 0.7]
    radius = 0.13

    for input_idx in range(self.num_inputs):
      for hidden_idx in range(self.num_hidden):
        ax.plot([input_x, hidden_x], [input_y[input_idx], hidden_y[hidden_idx]], 'k-', linewidth=1, alpha=0.7)
        horizontal = input_idx == hidden_idx
        label_t = 0.22 if horizontal else 0.72
        label_x = input_x + 0.26 if horizontal else hidden_x - 0.42
        label_y = input_y[input_idx] + (hidden_y[hidden_idx] - input_y[input_idx]) * label_t
        ax.text(label_x, label_y, f'$w^{{(1)}}_{{{hidden_idx+1},{input_idx+1}}}$', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none'))
    for hidden_idx in range(self.num_hidden):
      for output_idx in range(2):
        ax.plot([hidden_x, output_x], [hidden_y[hidden_idx], output_y[output_idx]], 'k-', linewidth=1, alpha=0.7)
        horizontal = hidden_idx == output_idx
        label_t = 0.22 if horizontal else 0.72
        label_x = hidden_x + 0.27 if horizontal else output_x - 0.48
        label_y = hidden_y[hidden_idx] + (output_y[output_idx] - hidden_y[hidden_idx]) * label_t
        ax.text(label_x, label_y, f'$w^{{(2)}}_{{{output_idx+1},{hidden_idx+1}}}$', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none'))

    for input_idx, y in enumerate(input_y):
      ax.add_patch(plt.Circle((input_x, y), radius, facecolor='lightgray', edgecolor='black', linewidth=1.5, zorder=10))
      ax.text(input_x - radius - 0.12, y, f'$x_{{{input_idx+1}}}$', fontsize=9, ha='right', va='center', zorder=11)
    for hidden_idx, y in enumerate(hidden_y):
      ax.add_patch(plt.Circle((hidden_x, y), radius, facecolor='lightblue', edgecolor='black', linewidth=1.5, zorder=10))
      ax.text(hidden_x, y, f'$h_{{{hidden_idx+1}}}$', fontsize=9, ha='center', va='center', zorder=11)
    for output_idx, y in enumerate(output_y):
      ax.add_patch(plt.Circle((output_x, y), radius, facecolor='lightblue', edgecolor='black', linewidth=1.5, zorder=10))
      ax.text(output_x, y, f'$o_{{{output_idx+1}}}$', fontsize=9, ha='center', va='center', zorder=11)

    ax.text(hidden_x, 0.25, 'Hidden layer (ReLU)', fontsize=8, ha='center', color='#12355b')
    ax.text(output_x, 1.65, 'Output logits', fontsize=8, ha='center', color='#12355b')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.0)
    plt.close(fig)
    buffer.seek(0)
    return buffer

  @classmethod
  def _build_body(cls, context) -> tuple[ca.Section, list[ca.Answer]]:
    self = context
    body = ca.Section()
    answers = []
    body.add_element(ca.Paragraph([
      "Given the two-class neural network below, the hidden layer uses ReLU and the two output logits use softmax. "
      "A completed forward pass is shown. Compute the requested individual weight gradients using backpropagation."
    ]))
    body.add_element(ca.Picture(img_data=self._generate_network_diagram(), caption="Two-logit softmax classifier"))
    body.add_element(self._generate_parameter_table())
    body.add_element(ca.Paragraph([
      "Use cross-entropy loss: ",
      ca.Equation(r"L = -\sum_j y_j\log(\hat{y}_j)", inline=True),
      "."
    ]))
    answers.extend([
      ca.AnswerTypes.Float(self._compute_gradient_W2(0, 0), label="∂L/∂w⁽²⁾₁₁"),
      ca.AnswerTypes.Float(self._compute_gradient_W2(1, 0), label="∂L/∂w⁽²⁾₂₁"),
      ca.AnswerTypes.Float(self._compute_gradient_W1(0, 0), label="∂L/∂w⁽¹⁾₁₁"),
    ])
    body.add_element(ca.AnswerBlock(answers))
    return body, answers

  @classmethod
  def _build_explanation(cls, context) -> tuple[ca.Section, list[ca.Answer]]:
    self = context
    explanation = ca.Section()
    explanation.add_element(ca.Paragraph([
      "Backpropagation starts with the softmax-plus-cross-entropy error vector, then moves backward one local derivative at a time."
    ]))
    delta_values = ", ".join(f"{value:.4f}" for value in self.dL_dz2)
    target_values = ", ".join(str(int(value)) for value in self.y_one_hot)
    prediction_values = ", ".join(f"{value:.4f}" for value in self.a2)
    explanation.add_element(ca.Equation(
      f"\\frac{{\\partial L}}{{\\partial o}} = \\hat{{y}} - y = [{prediction_values}] - [{target_values}] = [{delta_values}]",
      inline=False
    ))
    for output_idx in range(2):
      gradient = self._compute_gradient_W2(output_idx, 0)
      explanation.add_element(ca.Equation(
        f"\\frac{{\\partial L}}{{\\partial w^{{(2)}}_{{{output_idx+1},1}}}} = \\frac{{\\partial L}}{{\\partial o_{output_idx+1}}}h_1 = {self.dL_dz2[output_idx]:.4f} \\cdot {self.a1[0]:.4f} = {gradient:.4f}",
        inline=False
      ))

    dL_dh1 = self._compute_hidden_gradient(0)
    relu_derivative = self._hidden_activation_derivative(0)
    dL_dhpre1 = dL_dh1 * relu_derivative
    explanation.add_element(ca.Equation(
      f"\\frac{{\\partial L}}{{\\partial h_1}} = \\sum_{{j=1}}^2 w^{{(2)}}_{{j,1}}\\frac{{\\partial L}}{{\\partial o_j}} = {self.W2[0,0]:.4f} \\cdot {self.dL_dz2[0]:.4f} + {self.W2[1,0]:.4f} \\cdot {self.dL_dz2[1]:.4f} = {dL_dh1:.4f}",
      inline=False
    ))
    explanation.add_element(ca.Equation(
      f"\\text{{ReLU}}'(h_{{\\mathrm{{pre}},1}}) = {relu_derivative:.0f}",
      inline=False
    ))
    gradient_w11 = self._compute_gradient_W1(0, 0)
    explanation.add_element(ca.Equation(
      f"\\frac{{\\partial L}}{{\\partial w^{{(1)}}_{{1,1}}}} = \\frac{{\\partial L}}{{\\partial h_{{\\mathrm{{pre}},1}}}} \\cdot x_1 = {dL_dhpre1:.4f} \\cdot {self.X[0]:.1f} = {gradient_w11:.4f}",
      inline=False
    ))
    return explanation, []


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

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    self = context
    self.num_models = kwargs.get("num_models", getattr(self, "num_models", 5))

    # Generate predictions from multiple models
    # Use a range that makes sense for typical regression problems
    base_value = self.rng.uniform(0, 10)
    self.predictions = [
      base_value + self.rng.uniform(-2, 2)
      for _ in range(self.num_models)
    ]

    # Round to make calculations easier
    self.predictions = [round(p, 1) for p in self.predictions]
    return context

  @classmethod
  def _build_body(cls, context) -> tuple[ca.Section, list[ca.Answer]]:
    """Build question body and collect answers."""
    self = context
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

    mean_pred = np.mean(self.predictions)
    median_pred = np.median(self.predictions)
    answers.append(ca.AnswerTypes.Float(float(mean_pred), label="Mean (average)"))
    answers.append(ca.AnswerTypes.Float(float(median_pred), label="Median"))

    body.add_element(ca.AnswerBlock(answers))

    return body, answers

  @classmethod
  def _build_explanation(cls, context) -> tuple[ca.Section, list[ca.Answer]]:
    """Build question explanation."""
    self = context
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


@QuestionRegistry.register()
class EndToEndTrainingQuestion(SimpleNeuralNetworkBase):
  """
  End-to-end training step question.

  Students perform the requested parts of a training iteration:
  1. Forward pass → prediction
  2. Loss calculation (binary cross-entropy)
  3. Backpropagation → gradients for specific weights
  4. Weight update → new values for those weights only (not biases)
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.learning_rate = None
    self.new_W1 = None
    self.new_W2 = None

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    self = context

    # Generate network
    self._generate_network()
    self._select_activation_function()

    # Retain calculator precision; only final student answers are rounded.
    self._forward_pass(round_values=False)

    # Generate binary target (0 or 1)
    # Choose the opposite of what the network predicts to create meaningful gradients
    if self.a2[0] > 0.5:
      self.y_target = 0
    else:
      self.y_target = 1
    self._compute_loss(self.y_target)
    self._compute_output_gradient()

    # Set learning rate (use small value for stability)
    self.learning_rate = round(self.rng.uniform(0.05, 0.2), 2)

    # Compute updated weights
    self._compute_weight_updates()
    return context

  @classmethod
  def is_interesting_ctx(cls, context) -> bool:
    """Reject ReLU networks whose hidden layer is entirely inactive."""
    return (
      super().is_interesting_ctx(context)
      and not (
        context.activation_function == cls.ACTIVATION_RELU
        and np.all(context.a1 == 0)
      )
    )

  def _compute_weight_updates(self):
    """Compute the requested weight-only gradient descent updates."""
    # This exercise intentionally updates only w3 and w11.  Biases and all
    # other weights remain unchanged so the question scope matches its answers.
    self.new_W2 = np.copy(self.W2)
    self.new_W1 = np.copy(self.W1)
    self.new_W2[0, 0] = self.W2[0, 0] - self.learning_rate * self._compute_gradient_W2(0)
    self.new_W1[0, 0] = self.W1[0, 0] - self.learning_rate * self._compute_gradient_W1(0, 0)

  @classmethod
  def _build_body(cls, context) -> tuple[ca.Section, list[ca.Answer]]:
    """Build question body and collect answers."""
    self = context
    body = ca.Section()
    answers = []

    # Question description
    body.add_element(ca.Paragraph([
      f"Given the neural network below with {self._get_activation_name()} activation "
      f"in the hidden layer and sigmoid activation in the output layer (for binary classification), "
      f"perform the requested parts of a training step: forward pass, binary cross-entropy loss, "
      f"backpropagation, and updates to the specified weights only (not biases)."
    ]))

    body.add_element(ca.Paragraph([
      "Use full calculator precision for intermediate calculations and round each submitted answer to four decimal places."
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

    answers.append(ca.AnswerTypes.Float(
      float(self.a2[0]),
      label="1. Forward Pass - Network output ŷ"
    ))
    answers.append(ca.AnswerTypes.Float(float(self.loss), label="2. Loss"))
    answers.append(ca.AnswerTypes.Float(
      self._compute_gradient_W2(0),
      label="3. Gradient ∂L/∂w₃"
    ))
    answers.append(ca.AnswerTypes.Float(
      self._compute_gradient_W1(0, 0),
      label="4. Gradient ∂L/∂w₁₁"
    ))
    answers.append(ca.AnswerTypes.Float(float(self.new_W2[0, 0]), label="5. Updated w₃:"))
    answers.append(ca.AnswerTypes.Float(float(self.new_W1[0, 0]), label="6. Updated w₁₁:"))

    body.add_element(ca.AnswerBlock(answers))

    return body, answers

  @classmethod
  def _build_explanation(cls, context) -> tuple[ca.Section, list[ca.Answer]]:
    """Build question explanation."""
    self = context
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph([
      "This problem walks through a training step for the specified weights only; biases are not updated. "
      "The displayed decimal values are approximations—use full calculator precision until rounding each final answer to four decimal places."
    ]))

    # Step 1: Forward pass
    explanation.add_element(ca.Paragraph([
      "**Step 1: Forward Pass**"
    ]))

    # Hidden layer
    explanation.add_element(ca.Equation(
      f"z_1 = w_{{11}} x_1 + w_{{12}} x_2 + b_1 = {self.W1[0,0]:.{self.param_digits}f} \\cdot {self.X[0]:.1f} + {self.W1[0,1]:.{self.param_digits}f} \\cdot {self.X[1]:.1f} + {self.b1[0]:.{self.param_digits}f} = {self.z1[0]:.4f}",
      inline=False
    ))

    explanation.add_element(ca.Equation(
      f"h_1 = {self._get_activation_name()}(z_1) \\approx {self.a1[0]:.4f}",
      inline=False
    ))

    # Similarly for h2 (abbreviated)
    explanation.add_element(ca.Equation(
      f"h_2 \\approx {self.a1[1]:.4f} \\text{{ (calculated similarly)}}",
      inline=False
    ))

    # Output (pre-activation)
    explanation.add_element(ca.Equation(
      f"z_{{out}} = w_3 h_1 + w_4 h_2 + b_2 \\approx {self.W2[0,0]:.{self.param_digits}f} \\cdot {self.a1[0]:.4f} + {self.W2[0,1]:.{self.param_digits}f} \\cdot {self.a1[1]:.4f} + {self.b2[0]:.{self.param_digits}f} \\approx {self.z2[0]:.4f}",
      inline=False
    ))

    # Output (sigmoid activation)
    explanation.add_element(ca.Equation(
      f"\\hat{{y}} = \\sigma(z_{{out}}) \\approx \\frac{{1}}{{1 + e^{{-{self.z2[0]:.4f}}}}} \\approx {self.a2[0]:.4f}",
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
        f"L = -\\log(\\hat{{y}}) \\approx -\\log({self.a2[0]:.4f}) \\approx {self.loss:.4f}",
        inline=False
      ))
    else:
      explanation.add_element(ca.Equation(
        f"L = -\\log(1-\\hat{{y}}) \\approx -\\log({1-self.a2[0]:.4f}) \\approx {self.loss:.4f}",
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
      f"\\frac{{\\partial L}}{{\\partial z_{{out}}}} = \\hat{{y}} - y \\approx {self.a2[0]:.4f} - {int(self.y_target)} \\approx {self.dL_dz2:.4f}",
      inline=False
    ))

    grad_w3 = self._compute_gradient_W2(0)
    explanation.add_element(ca.Equation(
      f"\\frac{{\\partial L}}{{\\partial w_3}} = \\frac{{\\partial L}}{{\\partial z_{{out}}}} \\cdot h_1 \\approx {self.dL_dz2:.4f} \\cdot {self.a1[0]:.4f} \\approx {grad_w3:.4f}",
      inline=False
    ))

    grad_w11 = self._compute_gradient_W1(0, 0)
    dz2_da1 = self.W2[0, 0]
    da1_dz1 = self._hidden_activation_derivative(0)

    if self.activation_function == self.ACTIVATION_SIGMOID:
      act_deriv_str = f"h_1(1-h_1)"
    elif self.activation_function == self.ACTIVATION_RELU:
      act_deriv_str = f"\\text{{ReLU}}'(z_1)"
    else:
      act_deriv_str = f"1"

    explanation.add_element(ca.Equation(
      f"\\frac{{\\partial L}}{{\\partial w_{{11}}}} = \\frac{{\\partial L}}{{\\partial z_{{out}}}} \\cdot w_3 \\cdot {act_deriv_str} \\cdot x_1 \\approx {self.dL_dz2:.4f} \\cdot {dz2_da1:.4f} \\cdot {da1_dz1:.4f} \\cdot {self.X[0]:.1f} \\approx {grad_w11:.4f}",
      inline=False
    ))

    # Step 4: Weight updates
    explanation.add_element(ca.Paragraph([
      "**Step 4: Update Weights**"
    ]))

    new_w3 = self.new_W2[0, 0]
    explanation.add_element(ca.Equation(
      f"w_3^{{new}} = w_3 - \\alpha \\frac{{\\partial L}}{{\\partial w_3}} \\approx {self.W2[0,0]:.{self.param_digits}f} - {self.learning_rate} \\cdot {grad_w3:.4f} \\approx {new_w3:.4f}",
      inline=False
    ))

    new_w11 = self.new_W1[0, 0]
    explanation.add_element(ca.Equation(
      f"w_{{11}}^{{new}} = w_{{11}} - \\alpha \\frac{{\\partial L}}{{\\partial w_{{11}}}} \\approx {self.W1[0,0]:.{self.param_digits}f} - {self.learning_rate} \\cdot {grad_w11:.4f} \\approx {new_w11:.4f}",
      inline=False
    ))

    explanation.add_element(ca.Paragraph([
      "This exercise updates only ",
      ca.Equation(r"w_3", inline=True),
      " and ",
      ca.Equation(r"w_{11}", inline=True),
      "; biases and all other weights remain unchanged."
    ]))

    return explanation, []
