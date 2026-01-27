from __future__ import annotations

import abc
import logging
import math
import numpy as np
from typing import List, Tuple, Dict, Any

import QuizGenerator.contentast as ca
from QuizGenerator.question import Question, QuestionRegistry
from QuizGenerator.mixins import TableQuestionMixin, BodyTemplatesMixin

log = logging.getLogger(__name__)


# Note: This file migrates to the _get_body()/_get_explanation() pattern


class LossQuestion(Question, TableQuestionMixin, BodyTemplatesMixin, abc.ABC):
  """Base class for loss function calculation questions."""

  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.ML_OPTIMIZATION)
    super().__init__(*args, **kwargs)

    self.num_samples = kwargs.get("num_samples", 5)
    self.num_samples = max(3, min(10, self.num_samples))  # Constrain to 3-10 range

    self.num_input_features = kwargs.get("num_input_features", 2)
    self.num_input_features = max(1, min(5, self.num_input_features))  # Constrain to 1-5 features
    self.vector_inputs = kwargs.get("vector_inputs", False)  # Whether to show inputs as vectors

    # Generate sample data
    self.data = []
    self.individual_losses = []
    self.overall_loss = 0.0

  def refresh(self, rng_seed=None, *args, **kwargs):
    """Generate new random data and calculate losses."""
    super().refresh(rng_seed=rng_seed, *args, **kwargs)
    self._generate_data()
    self._calculate_losses()
    self._create_answers()

  @abc.abstractmethod
  def _generate_data(self):
    """Generate sample data appropriate for this loss function type."""
    pass

  @abc.abstractmethod
  def _calculate_losses(self):
    """Calculate individual and overall losses."""
    pass

  @abc.abstractmethod
  def _get_loss_function_name(self) -> str:
    """Return the name of the loss function."""
    pass

  @abc.abstractmethod
  def _get_loss_function_formula(self) -> str:
    """Return the LaTeX formula for the loss function."""
    pass

  @abc.abstractmethod
  def _get_loss_function_short_name(self) -> str:
    """Return the short name of the loss function (used in question body)."""
    pass

  def _create_answers(self):
    """Create answer objects for individual losses and overall loss."""
    self.answers = {}

    # Individual loss answers
    for i in range(self.num_samples):
      self.answers[f"loss_{i}"] = ca.AnswerTypes.Float(self.individual_losses[i], label=f"Sample {i + 1} loss")

    # Overall loss answer
    self.answers["overall_loss"] = ca.AnswerTypes.Float(self.overall_loss, label="Overall loss")

  def _get_body(self, **kwargs) -> Tuple[ca.Element, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    # Question description
    body.add_element(ca.Paragraph([
      f"Given the dataset below, calculate the {self._get_loss_function_short_name()} for each sample "
      f"and the overall {self._get_loss_function_short_name()}."
    ]))

    # Data table (contains individual loss answers)
    body.add_element(self._create_data_table())

    # Collect individual loss answers
    for i in range(self.num_samples):
      answers.append(self.answers[f"loss_{i}"])

    # Overall loss question
    body.add_element(ca.Paragraph([
      f"Overall {self._get_loss_function_short_name()}: "
    ]))
    answers.append(self.answers["overall_loss"])
    body.add_element(self.answers["overall_loss"])

    return body, answers

  def get_body(self, **kwargs) -> ca.Element:
    """Build question body (backward compatible interface)."""
    body, _ = self._get_body(**kwargs)
    return body

  @abc.abstractmethod
  def _create_data_table(self) -> ca.Element:
    """Create the data table with answer fields."""
    pass

  def _get_explanation(self, **kwargs) -> Tuple[ca.Element, List[ca.Answer]]:
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph([
      f"To calculate the {self._get_loss_function_name()}, we apply the formula to each sample:"
    ]))

    explanation.add_element(ca.Equation(self._get_loss_function_formula(), inline=False))

    # Step-by-step calculations
    explanation.add_element(self._create_calculation_steps())

    # Completed table
    explanation.add_element(ca.Paragraph(["Completed table:"]))
    explanation.add_element(self._create_completed_table())

    # Overall loss calculation
    explanation.add_element(self._create_overall_loss_explanation())

    return explanation, []

  def get_explanation(self, **kwargs) -> ca.Element:
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation(**kwargs)
    return explanation

  @abc.abstractmethod
  def _create_calculation_steps(self) -> ca.Element:
    """Create step-by-step calculation explanations."""
    pass

  @abc.abstractmethod
  def _create_completed_table(self) -> ca.Element:
    """Create the completed table with all values filled in."""
    pass

  @abc.abstractmethod
  def _create_overall_loss_explanation(self) -> ca.Element:
    """Create explanation for overall loss calculation."""
    pass


@QuestionRegistry.register("LossQuestion_Linear")
class LossQuestion_Linear(LossQuestion):
  """Linear regression with Mean Squared Error (MSE) loss."""

  def __init__(self, *args, **kwargs):
    self.num_output_vars = kwargs.get("num_output_vars", 1)
    self.num_output_vars = max(1, min(5, self.num_output_vars))  # Constrain to 1-5 range
    super().__init__(*args, **kwargs)

  def _generate_data(self):
    """Generate regression data with continuous target values."""
    self.data = []

    for i in range(self.num_samples):
      sample = {}

      # Generate input features (rounded to 2 decimal places)
      sample['inputs'] = [round(self.rng.uniform(-100, 100), 2) for _ in range(self.num_input_features)]

      # Generate true values (y) - multiple outputs if specified (rounded to 2 decimal places)
      if self.num_output_vars == 1:
        sample['true_values'] = round(self.rng.uniform(-100, 100), 2)
      else:
        sample['true_values'] = [round(self.rng.uniform(-100, 100), 2) for _ in range(self.num_output_vars)]

      # Generate predictions (p) - multiple outputs if specified (rounded to 2 decimal places)
      if self.num_output_vars == 1:
        sample['predictions'] = round(self.rng.uniform(-100, 100), 2)
      else:
        sample['predictions'] = [round(self.rng.uniform(-100, 100), 2) for _ in range(self.num_output_vars)]

      self.data.append(sample)

  def _calculate_losses(self):
    """Calculate MSE for each sample and overall."""
    self.individual_losses = []
    total_loss = 0.0

    for sample in self.data:
      if self.num_output_vars == 1:
        # Single output MSE: (y - p)^2
        loss = (sample['true_values'] - sample['predictions']) ** 2
      else:
        # Multi-output MSE: sum of (y_i - p_i)^2
        loss = sum(
          (y - p) ** 2
          for y, p in zip(sample['true_values'], sample['predictions'])
        )

      self.individual_losses.append(loss)
      total_loss += loss

    # Overall MSE is average of individual losses
    self.overall_loss = total_loss / self.num_samples

  def _get_loss_function_name(self) -> str:
    return "Mean Squared Error (MSE)"

  def _get_loss_function_short_name(self) -> str:
    return "MSE"

  def _get_loss_function_formula(self) -> str:
    if self.num_output_vars == 1:
      return r"L(y, p) = (y - p)^2"
    else:
      return r"L(\mathbf{y}, \mathbf{p}) = \sum_{i=1}^{k} (y_i - p_i)^2"

  def _create_data_table(self) -> ca.Element:
    """Create table with input features, true values, predictions, and loss fields."""
    headers = ["x"]

    if self.num_output_vars == 1:
      headers.extend(["y", "p", "loss"])
    else:
      # Multiple outputs
      for i in range(self.num_output_vars):
        headers.append(f"y_{i}")
      for i in range(self.num_output_vars):
        headers.append(f"p_{i}")
      headers.append("loss")

    rows = []
    for i, sample in enumerate(self.data):
      row = {}

      # Input features as vector
      x_vector = "[" + ", ".join([f"{x:.2f}" for x in sample['inputs']]) + "]"
      row["x"] = x_vector

      # True values
      if self.num_output_vars == 1:
        row["y"] = f"{sample['true_values']:.2f}"
      else:
        for j in range(self.num_output_vars):
          row[f"y_{j}"] = f"{sample['true_values'][j]:.2f}"

      # Predictions
      if self.num_output_vars == 1:
        row["p"] = f"{sample['predictions']:.2f}"
      else:
        for j in range(self.num_output_vars):
          row[f"p_{j}"] = f"{sample['predictions'][j]:.2f}"

      # Loss answer field
      row["loss"] = self.answers[f"loss_{i}"]

      rows.append(row)

    return self.create_answer_table(headers, rows, answer_columns=["loss"])

  def _create_calculation_steps(self) -> ca.Element:
    """Show step-by-step MSE calculations."""
    steps = ca.Section()

    for i, sample in enumerate(self.data):
      steps.add_element(ca.Paragraph([f"Sample {i+1}:"]))

      if self.num_output_vars == 1:
        y = sample['true_values']
        p = sample['predictions']
        loss = self.individual_losses[i]
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
        loss = self.individual_losses[i]

        terms = []
        for j, (y, p) in enumerate(zip(y_vals, p_vals)):
          # Format the subtraction nicely to avoid double negatives
          if p >= 0:
            terms.append(f"({y:.2f} - {p:.2f})^2")
          else:
            terms.append(f"({y:.2f} - ({p:.2f}))^2")

        calculation = f"L = {' + '.join(terms)} = {loss:.4f}"
        steps.add_element(ca.Equation(calculation, inline=False))

    return steps

  def _create_completed_table(self) -> ca.Element:
    """Create table with all values including calculated losses."""
    headers = ["x_0", "x_1"]

    if self.num_output_vars == 1:
      headers.extend(["y", "p", "loss"])
    else:
      for i in range(self.num_output_vars):
        headers.append(f"y_{i}")
      for i in range(self.num_output_vars):
        headers.append(f"p_{i}")
      headers.append("loss")

    rows = []
    for i, sample in enumerate(self.data):
      row = []

      # Input features
      for x in sample['inputs']:
        row.append(f"{x:.2f}")

      # True values
      if self.num_output_vars == 1:
        row.append(f"{sample['true_values']:.2f}")
      else:
        for y in sample['true_values']:
          row.append(f"{y:.2f}")

      # Predictions
      if self.num_output_vars == 1:
        row.append(f"{sample['predictions']:.2f}")
      else:
        for p in sample['predictions']:
          row.append(f"{p:.2f}")

      # Calculated loss
      row.append(f"{self.individual_losses[i]:.4f}")

      rows.append(row)

    return ca.Table(headers=headers, data=rows)

  def _create_overall_loss_explanation(self) -> ca.Element:
    """Explain overall MSE calculation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph([
      "The overall MSE is the average of individual losses:"
    ]))

    losses_str = " + ".join([f"{loss:.4f}" for loss in self.individual_losses])
    calculation = f"MSE = \\frac{{{losses_str}}}{{{self.num_samples}}} = {self.overall_loss:.4f}"

    explanation.add_element(ca.Equation(calculation, inline=False))

    return explanation


@QuestionRegistry.register("LossQuestion_Logistic")
class LossQuestion_Logistic(LossQuestion):
  """Binary logistic regression with log-loss."""

  def _generate_data(self):
    """Generate binary classification data."""
    self.data = []

    for i in range(self.num_samples):
      sample = {}

      # Generate input features (rounded to 2 decimal places)
      sample['inputs'] = [round(self.rng.uniform(-100, 100), 2) for _ in range(self.num_input_features)]

      # Generate binary true values (0 or 1)
      sample['true_values'] = self.rng.choice([0, 1])

      # Generate predicted probabilities (between 0 and 1, rounded to 3 decimal places)
      sample['predictions'] = round(self.rng.uniform(0.1, 0.9), 3)  # Avoid extreme values

      self.data.append(sample)

  def _calculate_losses(self):
    """Calculate log-loss for each sample and overall."""
    self.individual_losses = []
    total_loss = 0.0

    for sample in self.data:
      y = sample['true_values']
      p = sample['predictions']

      # Log-loss: -[y * log(p) + (1-y) * log(1-p)]
      if y == 1:
        loss = -math.log(p)
      else:
        loss = -math.log(1 - p)

      self.individual_losses.append(loss)
      total_loss += loss

    # Overall log-loss is average of individual losses
    self.overall_loss = total_loss / self.num_samples

  def _get_loss_function_name(self) -> str:
    return "Log-Loss (Binary Cross-Entropy)"

  def _get_loss_function_short_name(self) -> str:
    return "log-loss"

  def _get_loss_function_formula(self) -> str:
    return r"L(y, p) = -[y \ln(p) + (1-y) \ln(1-p)]"

  def _create_data_table(self) -> ca.Element:
    """Create table with features, true labels, predicted probabilities, and loss fields."""
    headers = ["x", "y", "p", "loss"]

    rows = []
    for i, sample in enumerate(self.data):
      row = {}

      # Input features as vector
      x_vector = "[" + ", ".join([f"{x:.2f}" for x in sample['inputs']]) + "]"
      row["x"] = x_vector

      # True label
      row["y"] = str(sample['true_values'])

      # Predicted probability
      row["p"] = f"{sample['predictions']:.3f}"

      # Loss answer field
      row["loss"] = self.answers[f"loss_{i}"]

      rows.append(row)

    return self.create_answer_table(headers, rows, answer_columns=["loss"])

  def _create_calculation_steps(self) -> ca.Element:
    """Show step-by-step log-loss calculations."""
    steps = ca.Section()

    for i, sample in enumerate(self.data):
      y = sample['true_values']
      p = sample['predictions']
      loss = self.individual_losses[i]

      steps.add_element(ca.Paragraph([f"Sample {i+1}:"]))

      if y == 1:
        calculation = f"L = -[1 \\cdot \\ln({p:.3f}) + 0 \\cdot \\ln(1-{p:.3f})] = -\\ln({p:.3f}) = {loss:.4f}"
      else:
        calculation = f"L = -[0 \\cdot \\ln({p:.3f}) + 1 \\cdot \\ln(1-{p:.3f})] = -\\ln({1-p:.3f}) = {loss:.4f}"

      steps.add_element(ca.Equation(calculation, inline=False))

    return steps

  def _create_completed_table(self) -> ca.Element:
    """Create table with all values including calculated losses."""
    headers = ["x_0", "x_1", "y", "p", "loss"]

    rows = []
    for i, sample in enumerate(self.data):
      row = []

      # Input features
      for x in sample['inputs']:
        row.append(f"{x:.2f}")

      # True label
      row.append(str(sample['true_values']))

      # Predicted probability
      row.append(f"{sample['predictions']:.3f}")

      # Calculated loss
      row.append(f"{self.individual_losses[i]:.4f}")

      rows.append(row)

    return ca.Table(headers=headers, data=rows)

  def _create_overall_loss_explanation(self) -> ca.Element:
    """Explain overall log-loss calculation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph([
      "The overall log-loss is the average of individual losses:"
    ]))

    losses_str = " + ".join([f"{loss:.4f}" for loss in self.individual_losses])
    calculation = f"\\text{{Log-Loss}} = \\frac{{{losses_str}}}{{{self.num_samples}}} = {self.overall_loss:.4f}"

    explanation.add_element(ca.Equation(calculation, inline=False))

    return explanation


@QuestionRegistry.register("LossQuestion_MulticlassLogistic")
class LossQuestion_MulticlassLogistic(LossQuestion):
  """Multi-class logistic regression with cross-entropy loss."""

  def __init__(self, *args, **kwargs):
    self.num_classes = kwargs.get("num_classes", 3)
    self.num_classes = max(3, min(5, self.num_classes))  # Constrain to 3-5 classes
    super().__init__(*args, **kwargs)

  def _generate_data(self):
    """Generate multi-class classification data."""
    self.data = []

    for i in range(self.num_samples):
      sample = {}

      # Generate input features (rounded to 2 decimal places)
      sample['inputs'] = [round(self.rng.uniform(-100, 100), 2) for _ in range(self.num_input_features)]

      # Generate true class (one-hot encoded) - ensure exactly one class is 1
      true_class_idx = self.rng.randint(0, self.num_classes - 1)
      sample['true_values'] = [0] * self.num_classes  # Start with all zeros
      sample['true_values'][true_class_idx] = 1        # Set exactly one to 1

      # Generate predicted probabilities (softmax-like, sum to 1, rounded to 3 decimal places)
      raw_probs = [self.rng.uniform(0.1, 2.0) for _ in range(self.num_classes)]
      prob_sum = sum(raw_probs)
      sample['predictions'] = [round(p / prob_sum, 3) for p in raw_probs]

      self.data.append(sample)

  def _calculate_losses(self):
    """Calculate cross-entropy loss for each sample and overall."""
    self.individual_losses = []
    total_loss = 0.0

    for sample in self.data:
      y_vec = sample['true_values']
      p_vec = sample['predictions']

      # Cross-entropy: -sum(y_i * log(p_i))
      loss = -sum(y * math.log(max(p, 1e-15)) for y, p in zip(y_vec, p_vec) if y > 0)

      self.individual_losses.append(loss)
      total_loss += loss

    # Overall cross-entropy is average of individual losses
    self.overall_loss = total_loss / self.num_samples

  def _get_loss_function_name(self) -> str:
    return "Cross-Entropy Loss"

  def _get_loss_function_short_name(self) -> str:
    return "cross-entropy loss"

  def _get_loss_function_formula(self) -> str:
    return r"L(\mathbf{y}, \mathbf{p}) = -\sum_{i=1}^{K} y_i \ln(p_i)"

  def _create_data_table(self) -> ca.Element:
    """Create table with features, true class vectors, predicted probabilities, and loss fields."""
    headers = ["x", "y", "p", "loss"]

    rows = []
    for i, sample in enumerate(self.data):
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
      row["loss"] = self.answers[f"loss_{i}"]

      rows.append(row)

    return self.create_answer_table(headers, rows, answer_columns=["loss"])

  def _create_calculation_steps(self) -> ca.Element:
    """Show step-by-step cross-entropy calculations."""
    steps = ca.Section()

    for i, sample in enumerate(self.data):
      y_vec = sample['true_values']
      p_vec = sample['predictions']
      loss = self.individual_losses[i]

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
        for j, (y, p) in enumerate(zip(y_vec, p_vec)):
          if y == 1:
            terms.append(f"{y} \\cdot \\ln({p:.3f})")
          else:
            terms.append(f"{y} \\cdot \\ln({p:.3f})")

        calculation = f"L = -\\mathbf{{y}} \\cdot \\ln(\\mathbf{{p}}) = -({' + '.join(terms)}) = -{y_vec[true_class_idx]} \\cdot \\ln({p_true:.3f}) = {loss:.4f}"
      except ValueError:
        # Fallback in case no class is set to 1 (shouldn't happen, but safety check)
        calculation = f"L = -\\mathbf{{y}} \\cdot \\ln(\\mathbf{{p}}) = {loss:.4f}"

      steps.add_element(ca.Equation(calculation, inline=False))

    return steps

  def _create_completed_table(self) -> ca.Element:
    """Create table with all values including calculated losses."""
    headers = ["x_0", "x_1", "y", "p", "loss"]

    rows = []
    for i, sample in enumerate(self.data):
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
      row.append(f"{self.individual_losses[i]:.4f}")

      rows.append(row)

    return ca.Table(headers=headers, data=rows)

  def _create_overall_loss_explanation(self) -> ca.Element:
    """Explain overall cross-entropy loss calculation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph([
      "The overall cross-entropy loss is the average of individual losses:"
    ]))

    losses_str = " + ".join([f"{loss:.4f}" for loss in self.individual_losses])
    calculation = f"\\text{{Cross-Entropy}} = \\frac{{{losses_str}}}{{{self.num_samples}}} = {self.overall_loss:.4f}"

    explanation.add_element(ca.Equation(calculation, inline=False))

    return explanation