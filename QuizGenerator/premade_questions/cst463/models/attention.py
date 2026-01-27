import abc
import logging
import math
import keras
import numpy as np
from typing import List, Tuple

from QuizGenerator.question import Question, QuestionRegistry
import QuizGenerator.contentast as ca
from QuizGenerator.constants import MathRanges
from QuizGenerator.mixins import TableQuestionMixin

from .matrices import MatrixQuestion

log = logging.getLogger(__name__)


@QuestionRegistry.register("cst463.attention.forward-pass")
class AttentionForwardPass(MatrixQuestion, TableQuestionMixin):
  
  @staticmethod
  def simple_attention(Q, K, V):
    """
    Q: (seq_len, d_k) - queries
    K: (seq_len, d_k) - keys
    V: (seq_len, d_v) - values

    Returns: (seq_len, d_v) - attended output
    """
    d_k = Q.shape[1]
    
    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_k)
    
    # Softmax to get weights
    attention_weights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    
    # Weighted sum of values
    output = attention_weights @ V
    
    return output, attention_weights
  
  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)
    
    seq_len = kwargs.get("seq_len", 3)
    d_k = kwargs.get("key_dimension", 1)  # key/query dimension
    d_v = kwargs.get("value_dimension", 1)  # value dimension
    
    # Small integer matrices
    self.Q = self.rng.randint(0, 3, size=(seq_len, d_k))
    self.K = self.rng.randint(0, 3, size=(seq_len, d_k))
    self.V = self.rng.randint(0, 3, size=(seq_len, d_v))

    self.Q = self.get_rounded_matrix((seq_len, d_k), 0, 3)
    self.K = self.get_rounded_matrix((seq_len, d_k), 0, 3)
    self.V = self.get_rounded_matrix((seq_len, d_v), 0, 3)
    
    self.output, self.weights = self.simple_attention(self.Q, self.K, self.V)
    
    ## Answers:
    # Q, K, V, output, weights

    self.answers["weights"] = ca.AnswerTypes.Matrix(self.weights, label="Weights")
    self.answers["output"] = ca.AnswerTypes.Matrix(self.output, label="Output")
    
    return True
  
  def _get_body(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    body.add_element(
      ca.Text("Given the below information about a self attention layer, please calculate the output sequence.")
    )
    body.add_element(
      self.create_info_table(
        {
          "Q": ca.Matrix(self.Q),
          "K": ca.Matrix(self.K),
          "V": ca.Matrix(self.V),
        }
      )
    )

    answers.append(self.answers["weights"])
    answers.append(self.answers["output"])
    body.add_elements([
      ca.LineBreak(),
      self.answers["weights"],
      ca.LineBreak(),
      self.answers["output"],
    ])

    return body, answers

  def get_body(self, **kwargs) -> ca.Section:
    """Build question body (backward compatible interface)."""
    body, _ = self._get_body(**kwargs)
    return body
  
  def _get_explanation(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question explanation."""
    explanation = ca.Section()
    digits = ca.Answer.DEFAULT_ROUNDING_DIGITS

    explanation.add_element(
      ca.Paragraph([
        "Self-attention uses scaled dot-product attention to compute a weighted combination of values based on query-key similarity."
      ])
    )

    # Step 1: Compute attention scores
    explanation.add_element(
      ca.Paragraph([
        ca.Text("Step 1: Compute attention scores", emphasis=True)
      ])
    )

    d_k = self.Q.shape[1]
    explanation.add_element(
      ca.Equation(f"\\text{{scores}} = \\frac{{Q K^T}}{{\\sqrt{{d_k}}}} = \\frac{{Q K^T}}{{\\sqrt{{{d_k}}}}}")
    )

    scores = self.Q @ self.K.T / np.sqrt(d_k)

    explanation.add_element(
      ca.Paragraph([
        "Raw scores (scaling by ",
        ca.Equation(f'\\sqrt{{{d_k}}}', inline=True),
        " prevents extremely large values):"
      ])
    )
    explanation.add_element(ca.Matrix(np.round(scores, digits)))

    # Step 2: Apply softmax
    explanation.add_element(
      ca.Paragraph([
        ca.Text("Step 2: Apply softmax to get attention weights", emphasis=True)
      ])
    )

    explanation.add_element(
      ca.Equation(r"\alpha_{ij} = \frac{\exp(\text{score}_{ij})}{\sum_k \exp(\text{score}_{ik})}")
    )

    # Show ONE example row
    explanation.add_element(
      ca.Paragraph([
        "Example: Row 0 softmax computation"
      ])
    )

    row_scores = scores[0]
    exp_scores = np.exp(row_scores)
    sum_exp = exp_scores.sum()
    weights_row = exp_scores / sum_exp

    exp_terms = " + ".join([f"e^{{{s:.{digits}f}}}" for s in row_scores])

    explanation.add_element(
      ca.Paragraph([
        f"Denominator = {exp_terms} = {sum_exp:.{digits}f}"
      ])
    )

    # Format array with proper rounding
    weights_str = "[" + ", ".join([f"{w:.{digits}f}" for w in weights_row]) + "]"
    explanation.add_element(
      ca.Paragraph([
        f"Resulting weights: {weights_str}"
      ])
    )

    explanation.add_element(
      ca.Paragraph([
        "Complete attention weight matrix:"
      ])
    )
    explanation.add_element(ca.Matrix(np.round(self.weights, digits)))

    # Step 3: Weighted sum of values
    explanation.add_element(
      ca.Paragraph([
        ca.Text("Step 3: Compute weighted sum of values", emphasis=True)
      ])
    )

    explanation.add_element(
      ca.Equation(r"\text{output} = \text{weights} \times V")
    )

    explanation.add_element(
      ca.Paragraph([
        "Final output:"
      ])
    )
    explanation.add_element(ca.Matrix(np.round(self.output, digits)))

    return explanation, []

  def get_explanation(self, **kwargs) -> ca.Section:
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation(**kwargs)
    return explanation

