import abc
import logging
import math
import keras
import numpy as np
from typing import List, Tuple

from .matrices import MatrixQuestion
from QuizGenerator.question import Question, QuestionRegistry
import QuizGenerator.contentast as ca
from QuizGenerator.constants import MathRanges
from QuizGenerator.mixins import TableQuestionMixin

log = logging.getLogger(__name__)


@QuestionRegistry.register("cst463.rnn.forward-pass")
class RNNForwardPass(MatrixQuestion, TableQuestionMixin):
  
  @staticmethod
  def rnn_forward(x_seq, W_xh, W_hh, b_h, h_0, activation='tanh'):
    """
    x_seq: (seq_len, input_dim) - input sequence
    W_xh: (input_dim, hidden_dim) - input to hidden weights
    W_hh: (hidden_dim, hidden_dim) - hidden to hidden weights
    b_h: (hidden_dim,) - hidden bias
    h_0: (hidden_dim,) - initial hidden state

    Returns: all hidden states (seq_len, hidden_dim)
    """
    seq_len = len(x_seq)
    hidden_dim = W_hh.shape[0]
    
    h_states = np.zeros((seq_len, hidden_dim))
    h_t = h_0
    
    for t in range(seq_len):
      h_t = x_seq[t] @ W_xh + h_t @ W_hh + b_h
      if activation == 'tanh':
        h_t = np.tanh(h_t)
      elif activation == 'relu':
        h_t = np.maximum(0, h_t)
      h_states[t] = h_t
    
    return h_states
  
  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)
    self.rng = np.random.RandomState(kwargs.get("rng_seed", None))
    
    seq_len = kwargs.get("seq_len", 3)
    input_dim =  kwargs.get("input_dim", 1)
    hidden_dim = kwargs.get("hidden_dim", 1)
    
    # Small integer weights for hand calculation
    self.x_seq = self.get_rounded_matrix((seq_len, input_dim)) # self.rng.randint(0, 3, size=(seq_len, input_dim))
    self.W_xh = self.get_rounded_matrix((input_dim, hidden_dim), -1, 2)
    self.W_hh = self.get_rounded_matrix((hidden_dim, hidden_dim), -1, 2)
    self.b_h = self.get_rounded_matrix((hidden_dim,), -1, 2)
    self.h_0 = np.zeros(hidden_dim)
    
    self.h_states = self.rnn_forward(self.x_seq, self.W_xh, self.W_hh, self.b_h, self.h_0) #.reshape((seq_len,-1))
    
    ## Answers:
    # x_seq, W_xh, W_hh, b_h, h_0, h_states
    
    self.answers["output_sequence"] = ca.AnswerTypes.Matrix(value=self.h_states, label="Hidden states")
    
    return True
  
  def _get_body(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    body.add_element(
      ca.Paragraph([
        ca.Text("Given the below information about an RNN, please calculate the output sequence."),
        "Assume that you are using a tanh activation function."
      ])
    )
    body.add_element(
      self.create_info_table(
        {
          ca.Container(["Input sequence, ", ca.Equation("x_{seq}", inline=True)]) : ca.Matrix(self.x_seq),
          ca.Container(["Input weights, ",  ca.Equation("W_{xh}", inline=True)])  : ca.Matrix(self.W_xh),
          ca.Container(["Hidden weights, ", ca.Equation("W_{hh}", inline=True)])  : ca.Matrix(self.W_hh),
          ca.Container(["Bias, ",           ca.Equation("b_{h}", inline=True)])   : ca.Matrix(self.b_h),
          ca.Container(["Hidden states, ",  ca.Equation("h_{0}", inline=True)])   : ca.Matrix(self.h_0),
        }
      )
    )

    body.add_element(ca.LineBreak())

    answers.append(self.answers["output_sequence"])
    body.add_element(self.answers["output_sequence"])

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
    "For an RNN forward pass, we compute the hidden state at each time step using:"
      ])
    )

    explanation.add_element(
      ca.Equation(r"h_t = \tanh(x_t W_{xh} + h_{t-1} W_{hh} + b_h)")
    )

    explanation.add_element(
      ca.Paragraph([
        "Where the input contributes via ", ca.Equation("W_{xh}", inline=True),
        ", the previous hidden state contributes via ", ca.Equation("W_{hh}", inline=True),
        ", and ", ca.Equation("b_h", inline=True), " is the bias."
      ])
    )

    # Format arrays with proper rounding
    def format_array(arr):
      from QuizGenerator.misc import fix_negative_zero
      if arr.ndim == 0:
        return f"{fix_negative_zero(arr):.{digits}f}"
      return "[" + ", ".join([f"{fix_negative_zero(x):.{digits}f}" for x in arr.flatten()]) + "]"

    # Show detailed examples for first 2 timesteps (or just 1 if seq_len == 1)
    seq_len = len(self.x_seq)
    num_examples = min(2, seq_len)

    explanation.add_element(ca.Paragraph([""]))

    for t in range(num_examples):
      explanation.add_element(
        ca.Paragraph([
          ca.Text(f"Example: Timestep {t}", emphasis=True)
        ])
      )

      # Compute step t
      x_contribution = self.x_seq[t] @ self.W_xh
      if t == 0:
        h_prev = self.h_0
        h_prev_label = 'h_{-1}'
        h_prev_desc = " (initial state)"
      else:
        h_prev = self.h_states[t-1]
        h_prev_label = f'h_{{{t-1}}}'
        h_prev_desc = ""

      h_contribution = h_prev @ self.W_hh
      pre_activation = x_contribution + h_contribution + self.b_h
      h_result = np.tanh(pre_activation)

      explanation.add_element(
        ca.Paragraph([
          "Input contribution: ",
          ca.Equation(f'x_{t} W_{{xh}}', inline=True),
          f" = {format_array(x_contribution)}"
        ])
      )

      explanation.add_element(
        ca.Paragraph([
          "Hidden contribution: ",
          ca.Equation(f'{h_prev_label} W_{{hh}}', inline=True),
          f"{h_prev_desc} = {format_array(h_contribution)}"
        ])
      )

      explanation.add_element(
        ca.Paragraph([
          f"Pre-activation: {format_array(pre_activation)}"
        ])
      )

      explanation.add_element(
        ca.Paragraph([
          "After tanh: ",
          ca.Equation(f'h_{t}', inline=True),
          f" = {format_array(h_result)}"
        ])
      )

      # Add visual separator between timesteps (except after the last one)
      if t < num_examples - 1:
        explanation.add_element(ca.Paragraph([""]))

    # Show complete output sequence (rounded)
    explanation.add_element(
      ca.Paragraph([
        "Complete hidden state sequence (each row is one timestep):"
      ])
    )

    explanation.add_element(
      ca.Matrix(np.round(self.h_states, digits))
    )

    return explanation, []

  def get_explanation(self, **kwargs) -> ca.Section:
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation(**kwargs)
    return explanation

