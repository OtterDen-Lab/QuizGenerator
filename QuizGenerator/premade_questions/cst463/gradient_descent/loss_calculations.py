from __future__ import annotations

import abc
import logging
import math
from typing import List, Tuple

import QuizGenerator.contentast as ca
from QuizGenerator.mixins import BodyTemplatesMixin, TableQuestionMixin
from QuizGenerator.question import Question, QuestionRegistry

log = logging.getLogger(__name__)


# Note: This file migrates to the _build_body()/_build_explanation() pattern


class LossQuestion(Question, TableQuestionMixin, BodyTemplatesMixin, abc.ABC):
  """Base class for loss function calculation questions."""

  DEFAULT_NUM_SAMPLES = 5
  DEFAULT_NUM_INPUT_FEATURES = 2
  DEFAULT_VECTOR_INPUTS = False

  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.ML_OPTIMIZATION)
    super().__init__(*args, **kwargs)

    self.num_samples = kwargs.get("num_samples", self.DEFAULT_NUM_SAMPLES)
    self.num_samples = max(3, min(10, self.num_samples))  # Constrain to 3-10 range

    self.num_input_features = kwargs.get("num_input_features", self.DEFAULT_NUM_INPUT_FEATURES)
    self.num_input_features = max(1, min(5, self.num_input_features))  # Constrain to 1-5 features
    self.vector_inputs = kwargs.get("vector_inputs", self.DEFAULT_VECTOR_INPUTS)  # Whether to show inputs as vectors

    # Generate sample data
    self.data = []
    self.individual_losses = []
    self.overall_loss = 0.0

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    """Generate new random data and calculate losses."""
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    cls._populate_context(context, **kwargs)
    # Update configurable parameters if provided
    context.num_samples = max(3, min(10, kwargs.get("num_samples", cls.DEFAULT_NUM_SAMPLES)))
    context.num_input_features = max(1, min(5, kwargs.get("num_input_features", cls.DEFAULT_NUM_INPUT_FEATURES)))
    context.vector_inputs = kwargs.get("vector_inputs", cls.DEFAULT_VECTOR_INPUTS)

    # Generate data + losses
    cls._generate_data(context)
    cls._calculate_losses(context)
    return context

  @classmethod
  def _populate_context(cls, context, **kwargs):
    """Hook for subclasses to add required context before data generation."""
    return context

  @classmethod
  @abc.abstractmethod
  def _generate_data(cls, context):
    """Generate sample data appropriate for this loss function type."""
    pass

  @classmethod
  @abc.abstractmethod
  def _calculate_losses(cls, context):
    """Calculate individual and overall losses."""
    pass

  @classmethod
  @abc.abstractmethod
  def _get_loss_function_name(cls, context) -> str:
    """Return the name of the loss function."""
    pass

  @classmethod
  @abc.abstractmethod
  def _get_loss_function_formula(cls, context) -> str:
    """Return the LaTeX formula for the loss function."""
    pass

  @classmethod
  @abc.abstractmethod
  def _get_loss_function_short_name(cls, context) -> str:
    """Return the short name of the loss function (used in question body)."""
    pass

  @classmethod
  def _build_loss_answers(cls, context) -> Tuple[List[ca.Answer], ca.Answer]:
    answers = [
      ca.AnswerTypes.Float(context.individual_losses[i], label=f"Sample {i + 1} loss")
      for i in range(context.num_samples)
    ]
    overall = ca.AnswerTypes.Float(context.overall_loss, label="Overall loss")
    return answers, overall

  @classmethod
  def _build_body(cls, context) -> Tuple[ca.Element, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    # Question description
    body.add_element(ca.Paragraph([
      f"Given the dataset below, calculate the {cls._get_loss_function_short_name(context)} for each sample "
      f"and the overall {cls._get_loss_function_short_name(context)}."
    ]))

    # Data table (contains individual loss answers)
    loss_answers, overall_answer = cls._build_loss_answers(context)
    body.add_element(cls._create_data_table(context, loss_answers))
    answers.extend(loss_answers)

    # Overall loss question
    body.add_element(ca.Paragraph([
      f"Overall {cls._get_loss_function_short_name(context)}: "
    ]))
    answers.append(overall_answer)
    body.add_element(overall_answer)

    return body, answers

  @classmethod
  @abc.abstractmethod
  def _create_data_table(cls, context, loss_answers: List[ca.Answer]) -> ca.Element:
    """Create the data table with answer fields."""
    pass

  @classmethod
  def _build_explanation(cls, context) -> Tuple[ca.Element, List[ca.Answer]]:
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph([
      f"To calculate the {cls._get_loss_function_name(context)}, we apply the formula to each sample:"
    ]))

    explanation.add_element(ca.Equation(cls._get_loss_function_formula(context), inline=False))

    # Step-by-step calculations
    explanation.add_element(cls._create_calculation_steps(context))

    # Completed table
    explanation.add_element(ca.Paragraph(["Completed table:"]))
    explanation.add_element(cls._create_completed_table(context))

    # Overall loss calculation
    explanation.add_element(cls._create_overall_loss_explanation(context))

    return explanation, []

  @classmethod
  @abc.abstractmethod
  def _create_calculation_steps(cls, context) -> ca.Element:
    """Create step-by-step calculation explanations."""
    pass

  @classmethod
  @abc.abstractmethod
  def _create_completed_table(cls, context) -> ca.Element:
    """Create the completed table with all values filled in."""
    pass

  @classmethod
  @abc.abstractmethod
  def _create_overall_loss_explanation(cls, context) -> ca.Element:
    """Create explanation for overall loss calculation."""
    pass


@QuestionRegistry.register("LossQuestion_Linear")
class LossQuestion_Linear(LossQuestion):
  """Linear regression with Mean Squared Error (MSE) loss."""

  DEFAULT_NUM_OUTPUT_VARS = 1

  def __init__(self, *args, **kwargs):
    self.num_output_vars = kwargs.get("num_output_vars", self.DEFAULT_NUM_OUTPUT_VARS)
    self.num_output_vars = max(1, min(5, self.num_output_vars))  # Constrain to 1-5 range
    super().__init__(*args, **kwargs)

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    return super()._build_context(rng_seed=rng_seed, **kwargs)

  @classmethod
  def _populate_context(cls, context, **kwargs):
    context.num_output_vars = max(
      1,
      min(5, kwargs.get("num_output_vars", cls.DEFAULT_NUM_OUTPUT_VARS))
    )
    return context

  @classmethod
  def _generate_data(cls, context):
    """Generate regression data with continuous target values."""
    context.data = {}
    context["data"] = []

    for _ in range(context.num_samples):
      sample = {}

      # Generate input features (rounded to 2 decimal places)
      sample['inputs'] = [
        round(context.rng.uniform(-100, 100), 2)
        for _ in range(context.num_input_features)
      ]

      # Generate true values (y) - multiple outputs if specified (rounded to 2 decimal places)
      if context.num_output_vars == 1:
        sample['true_values'] = round(context.rng.uniform(-100, 100), 2)
      else:
        sample['true_values'] = [
          round(context.rng.uniform(-100, 100), 2)
          for _ in range(context.num_output_vars)
        ]

      # Generate predictions (p) - multiple outputs if specified (rounded to 2 decimal places)
      if context.num_output_vars == 1:
        sample['predictions'] = round(context.rng.uniform(-100, 100), 2)
      else:
        sample['predictions'] = [
          round(context.rng.uniform(-100, 100), 2)
          for _ in range(context.num_output_vars)
        ]

      context["data"].append(sample)

  @classmethod
  def _calculate_losses(cls, context):
    """Calculate MSE for each sample and overall."""
    context.individual_losses = []
    total_loss = 0.0

    for sample in context["data"]:
      if context.num_output_vars == 1:
        # Single output MSE: (y - p)^2
        loss = (sample['true_values'] - sample['predictions']) ** 2
      else:
        # Multi-output MSE: sum of (y_i - p_i)^2
        loss = sum(
          (y - p) ** 2
          for y, p in zip(sample['true_values'], sample['predictions'])
        )

      context.individual_losses.append(loss)
      total_loss += loss

    # Overall MSE is average of individual losses
    context.overall_loss = total_loss / context.num_samples

  @classmethod
  def _get_loss_function_name(cls, context) -> str:
    return "Mean Squared Error (MSE)"

  @classmethod
  def _get_loss_function_short_name(cls, context) -> str:
    return "MSE"

  @classmethod
  def _get_loss_function_formula(cls, context) -> str:
    if context.num_output_vars == 1:
      return r"L(y, p) = (y - p)^2"
    return r"L(\mathbf{y}, \mathbf{p}) = \sum_{i=1}^{k} (y_i - p_i)^2"

  @classmethod
  def _create_data_table(cls, context, loss_answers: List[ca.Answer]) -> ca.Element:
    """Create table with input features, true values, predictions, and loss fields."""
    headers = ["x"]

    if context.num_output_vars == 1:
      headers.extend(["y", "p", "loss"])
    else:
      # Multiple outputs
      for i in range(context.num_output_vars):
        headers.append(f"y_{i}")
      for i in range(context.num_output_vars):
        headers.append(f"p_{i}")
      headers.append("loss")

    rows = []
    for i, sample in enumerate(context["data"]):
      row = {}

      # Input features as vector
      x_vector = "[" + ", ".join([f"{x:.2f}" for x in sample['inputs']]) + "]"
      row["x"] = x_vector

      # True values
      if context.num_output_vars == 1:
        row["y"] = f"{sample['true_values']:.2f}"
      else:
        for j in range(context.num_output_vars):
          row[f"y_{j}"] = f"{sample['true_values'][j]:.2f}"

      # Predictions
      if context.num_output_vars == 1:
        row["p"] = f"{sample['predictions']:.2f}"
      else:
        for j in range(context.num_output_vars):
          row[f"p_{j}"] = f"{sample['predictions'][j]:.2f}"

      # Loss answer field
      row["loss"] = loss_answers[i]

      rows.append(row)

    return cls.create_answer_table(headers, rows, answer_columns=["loss"])

  @classmethod
  def _create_calculation_steps(cls, context) -> ca.Element:
    """Show step-by-step MSE calculations."""
    steps = ca.Section()

    for i, sample in enumerate(context["data"]):
      steps.add_element(ca.Paragraph([f"Sample {i+1}:"]))

      if context.num_output_vars == 1:
        y = sample['true_values']
        p = sample['predictions']
        loss = context.individual_losses[i]
        diff = y - p

        # Format the subtraction nicely to avoid double negatives
        if p >= 0:
          calculation = f"L = ({y:.2f} - {p:.2f})^2 = ({diff:.2f})^2 = {loss:.4f}"
        else:
          calculation = f"L = ({y:.2f} - ({p:.2f}))^2 = ({diff:.2f})^2 = {loss:.4f}"
        steps.add_element(ca.Equation(calculation, inline=False))
      else:
        # Multi-output calculation
        y_vals = sample['true_values']
        p_vals = sample['predictions']
        loss = context.individual_losses[i]

        terms = []
        for y, p in zip(y_vals, p_vals):
          # Format the subtraction nicely to avoid double negatives
          if p >= 0:
            terms.append(f"({y:.2f} - {p:.2f})^2")
          else:
            terms.append(f"({y:.2f} - ({p:.2f}))^2")

        calculation = f"L = {' + '.join(terms)} = {loss:.4f}"
        steps.add_element(ca.Equation(calculation, inline=False))

    return steps

  @classmethod
  def _create_completed_table(cls, context) -> ca.Element:
    """Create table with all values including calculated losses."""
    headers = ["x_0", "x_1"]

    if context.num_output_vars == 1:
      headers.extend(["y", "p", "loss"])
    else:
      for i in range(context.num_output_vars):
        headers.append(f"y_{i}")
      for i in range(context.num_output_vars):
        headers.append(f"p_{i}")
      headers.append("loss")

    rows = []
    for i, sample in enumerate(context["data"]):
      row = []

      # Input features
      for x in sample['inputs']:
        row.append(f"{x:.2f}")

      # True values
      if context.num_output_vars == 1:
        row.append(f"{sample['true_values']:.2f}")
      else:
        for y in sample['true_values']:
          row.append(f"{y:.2f}")

      # Predictions
      if context.num_output_vars == 1:
        row.append(f"{sample['predictions']:.2f}")
      else:
        for p in sample['predictions']:
          row.append(f"{p:.2f}")

      # Calculated loss
      row.append(f"{context.individual_losses[i]:.4f}")

      rows.append(row)

    return ca.Table(headers=headers, data=rows)

  @classmethod
  def _create_overall_loss_explanation(cls, context) -> ca.Element:
    """Explain overall MSE calculation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph([
      "The overall MSE is the average of individual losses:"
    ]))

    losses_str = " + ".join([f"{loss:.4f}" for loss in context.individual_losses])
    calculation = f"MSE = \\frac{{{losses_str}}}{{{context.num_samples}}} = {context.overall_loss:.4f}"

    explanation.add_element(ca.Equation(calculation, inline=False))

    return explanation


@QuestionRegistry.register("LossQuestion_Logistic")
class LossQuestion_Logistic(LossQuestion):
  """Binary logistic regression with log-loss."""

  @classmethod
  def _generate_data(cls, context):
    """Generate binary classification data."""
    context.data = {}
    context["data"] = []

    for _ in range(context.num_samples):
      sample = {}

      # Generate input features (rounded to 2 decimal places)
      sample['inputs'] = [
        round(context.rng.uniform(-100, 100), 2)
        for _ in range(context.num_input_features)
      ]

      # Generate binary true values (0 or 1)
      sample['true_values'] = context.rng.choice([0, 1])

      # Generate predicted probabilities (between 0 and 1, rounded to 3 decimal places)
      sample['predictions'] = round(context.rng.uniform(0.1, 0.9), 3)  # Avoid extreme values

      context["data"].append(sample)

  @classmethod
  def _calculate_losses(cls, context):
    """Calculate log-loss for each sample and overall."""
    context.individual_losses = []
    total_loss = 0.0

    for sample in context["data"]:
      y = sample['true_values']
      p = sample['predictions']

      # Log-loss: -[y * log(p) + (1-y) * log(1-p)]
      if y == 1:
        loss = -math.log(p)
      else:
        loss = -math.log(1 - p)

      context.individual_losses.append(loss)
      total_loss += loss

    # Overall log-loss is average of individual losses
    context.overall_loss = total_loss / context.num_samples

  @classmethod
  def _get_loss_function_name(cls, context) -> str:
    return "Log-Loss (Binary Cross-Entropy)"

  @classmethod
  def _get_loss_function_short_name(cls, context) -> str:
    return "log-loss"

  @classmethod
  def _get_loss_function_formula(cls, context) -> str:
    return r"L(y, p) = -[y \ln(p) + (1-y) \ln(1-p)]"

  @classmethod
  def _create_data_table(cls, context, loss_answers: List[ca.Answer]) -> ca.Element:
    """Create table with features, true labels, predicted probabilities, and loss fields."""
    headers = ["x", "y", "p", "loss"]

    rows = []
    for i, sample in enumerate(context["data"]):
      row = {}

      # Input features as vector
      x_vector = "[" + ", ".join([f"{x:.2f}" for x in sample['inputs']]) + "]"
      row["x"] = x_vector

      # True label
      row["y"] = str(sample['true_values'])

      # Predicted probability
      row["p"] = f"{sample['predictions']:.3f}"

      # Loss answer field
      row["loss"] = loss_answers[i]

      rows.append(row)

    return cls.create_answer_table(headers, rows, answer_columns=["loss"])

  @classmethod
  def _create_calculation_steps(cls, context) -> ca.Element:
    """Show step-by-step log-loss calculations."""
    steps = ca.Section()

    for i, sample in enumerate(context["data"]):
      y = sample['true_values']
      p = sample['predictions']
      loss = context.individual_losses[i]

      steps.add_element(ca.Paragraph([f"Sample {i+1}:"]))

      if y == 1:
        calculation = f"L = -[1 \\cdot \\ln({p:.3f}) + 0 \\cdot \\ln(1-{p:.3f})] = -\\ln({p:.3f}) = {loss:.4f}"
      else:
        calculation = f"L = -[0 \\cdot \\ln({p:.3f}) + 1 \\cdot \\ln(1-{p:.3f})] = -\\ln({1-p:.3f}) = {loss:.4f}"

      steps.add_element(ca.Equation(calculation, inline=False))

    return steps

  @classmethod
  def _create_completed_table(cls, context) -> ca.Element:
    """Create table with all values including calculated losses."""
    headers = ["x_0", "x_1", "y", "p", "loss"]

    rows = []
    for i, sample in enumerate(context["data"]):
      row = []

      # Input features
      for x in sample['inputs']:
        row.append(f"{x:.2f}")

      # True label
      row.append(str(sample['true_values']))

      # Predicted probability
      row.append(f"{sample['predictions']:.3f}")

      # Calculated loss
      row.append(f"{context.individual_losses[i]:.4f}")

      rows.append(row)

    return ca.Table(headers=headers, data=rows)

  @classmethod
  def _create_overall_loss_explanation(cls, context) -> ca.Element:
    """Explain overall log-loss calculation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph([
      "The overall log-loss is the average of individual losses:"
    ]))

    losses_str = " + ".join([f"{loss:.4f}" for loss in context.individual_losses])
    calculation = f"\\text{{Log-Loss}} = \\frac{{{losses_str}}}{{{context.num_samples}}} = {context.overall_loss:.4f}"

    explanation.add_element(ca.Equation(calculation, inline=False))

    return explanation


@QuestionRegistry.register("LossQuestion_MulticlassLogistic")
class LossQuestion_MulticlassLogistic(LossQuestion):
  """Multi-class logistic regression with cross-entropy loss."""

  DEFAULT_NUM_CLASSES = 3

  def __init__(self, *args, **kwargs):
    self.num_classes = kwargs.get("num_classes", self.DEFAULT_NUM_CLASSES)
    self.num_classes = max(3, min(5, self.num_classes))  # Constrain to 3-5 classes
    super().__init__(*args, **kwargs)

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    return super()._build_context(rng_seed=rng_seed, **kwargs)

  @classmethod
  def _populate_context(cls, context, **kwargs):
    context.num_classes = max(
      3,
      min(5, kwargs.get("num_classes", cls.DEFAULT_NUM_CLASSES))
    )
    return context

  @classmethod
  def _generate_data(cls, context):
    """Generate multi-class classification data."""
    context.data = {}
    context["data"] = []

    for _ in range(context.num_samples):
      sample = {}

      # Generate input features (rounded to 2 decimal places)
      sample['inputs'] = [
        round(context.rng.uniform(-100, 100), 2)
        for _ in range(context.num_input_features)
      ]

      # Generate true class (one-hot encoded) - ensure exactly one class is 1
      true_class_idx = context.rng.randint(0, context.num_classes - 1)
      sample['true_values'] = [0] * context.num_classes  # Start with all zeros
      sample['true_values'][true_class_idx] = 1          # Set exactly one to 1

      # Generate predicted probabilities (softmax-like, sum to 1, rounded to 3 decimal places)
      raw_probs = [context.rng.uniform(0.1, 2.0) for _ in range(context.num_classes)]
      prob_sum = sum(raw_probs)
      sample['predictions'] = [round(p / prob_sum, 3) for p in raw_probs]

      context["data"].append(sample)

  @classmethod
  def _calculate_losses(cls, context):
    """Calculate cross-entropy loss for each sample and overall."""
    context.individual_losses = []
    total_loss = 0.0

    for sample in context["data"]:
      y_vec = sample['true_values']
      p_vec = sample['predictions']

      # Cross-entropy: -sum(y_i * log(p_i))
      loss = -sum(y * math.log(max(p, 1e-15)) for y, p in zip(y_vec, p_vec) if y > 0)

      context.individual_losses.append(loss)
      total_loss += loss

    # Overall cross-entropy is average of individual losses
    context.overall_loss = total_loss / context.num_samples

  @classmethod
  def _get_loss_function_name(cls, context) -> str:
    return "Cross-Entropy Loss"

  @classmethod
  def _get_loss_function_short_name(cls, context) -> str:
    return "cross-entropy loss"

  @classmethod
  def _get_loss_function_formula(cls, context) -> str:
    return r"L(\mathbf{y}, \mathbf{p}) = -\sum_{i=1}^{K} y_i \ln(p_i)"

  @classmethod
  def _create_data_table(cls, context, loss_answers: List[ca.Answer]) -> ca.Element:
    """Create table with features, true class vectors, predicted probabilities, and loss fields."""
    headers = ["x", "y", "p", "loss"]

    rows = []
    for i, sample in enumerate(context["data"]):
      row = {}

      # Input features as vector
      x_vector = "[" + ", ".join([f"{x:.2f}" for x in sample['inputs']]) + "]"
      row["x"] = x_vector

      # True values (one-hot vector)
      y_vector = "[" + ", ".join([str(y) for y in sample['true_values']]) + "]"
      row["y"] = y_vector

      # Predicted probabilities (vector)
      p_vector = "[" + ", ".join([f"{p:.3f}" for p in sample['predictions']]) + "]"
      row["p"] = p_vector

      # Loss answer field
      row["loss"] = loss_answers[i]

      rows.append(row)

    return cls.create_answer_table(headers, rows, answer_columns=["loss"])

  @classmethod
  def _create_calculation_steps(cls, context) -> ca.Element:
    """Show step-by-step cross-entropy calculations."""
    steps = ca.Section()

    for i, sample in enumerate(context["data"]):
      y_vec = sample['true_values']
      p_vec = sample['predictions']
      loss = context.individual_losses[i]

      steps.add_element(ca.Paragraph([f"Sample {i+1}:"]))

      # Show vector dot product calculation
      y_str = "[" + ", ".join([str(y) for y in y_vec]) + "]"
      p_str = "[" + ", ".join([f"{p:.3f}" for p in p_vec]) + "]"

      steps.add_element(ca.Paragraph([f"\\mathbf{{y}} = {y_str}, \\mathbf{{p}} = {p_str}"]))

      # Find the true class (where y_i = 1)
      try:
        true_class_idx = y_vec.index(1)
        p_true = p_vec[true_class_idx]

        # Show the vector multiplication more explicitly
        terms = []
        for y, p in zip(y_vec, p_vec):
          terms.append(f"{y} \\cdot \\ln({p:.3f})")

        calculation = f"L = -\\mathbf{{y}} \\cdot \\ln(\\mathbf{{p}}) = -({' + '.join(terms)}) = -{y_vec[true_class_idx]} \\cdot \\ln({p_true:.3f}) = {loss:.4f}"
      except ValueError:
        # Fallback in case no class is set to 1 (shouldn't happen, but safety check)
        calculation = f"L = -\\mathbf{{y}} \\cdot \\ln(\\mathbf{{p}}) = {loss:.4f}"

      steps.add_element(ca.Equation(calculation, inline=False))

    return steps

  @classmethod
  def _create_completed_table(cls, context) -> ca.Element:
    """Create table with all values including calculated losses."""
    headers = ["x_0", "x_1", "y", "p", "loss"]

    rows = []
    for i, sample in enumerate(context["data"]):
      row = []

      # Input features
      for x in sample['inputs']:
        row.append(f"{x:.2f}")

      # True values (one-hot vector)
      y_vector = "[" + ", ".join([str(y) for y in sample['true_values']]) + "]"
      row.append(y_vector)

      # Predicted probabilities (vector)
      p_vector = "[" + ", ".join([f"{p:.3f}" for p in sample['predictions']]) + "]"
      row.append(p_vector)

      # Calculated loss
      row.append(f"{context.individual_losses[i]:.4f}")

      rows.append(row)

    return ca.Table(headers=headers, data=rows)

  @classmethod
  def _create_overall_loss_explanation(cls, context) -> ca.Element:
    """Explain overall cross-entropy loss calculation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph([
      "The overall cross-entropy loss is the average of individual losses:"
    ]))

    losses_str = " + ".join([f"{loss:.4f}" for loss in context.individual_losses])
    calculation = f"\\text{{Cross-Entropy}} = \\frac{{{losses_str}}}{{{context.num_samples}}} = {context.overall_loss:.4f}"

    explanation.add_element(ca.Equation(calculation, inline=False))

    return explanation
