import logging
from typing import List, Tuple

import numpy as np

import QuizGenerator.contentast as ca
from QuizGenerator.mixins import TableQuestionMixin
from QuizGenerator.question import QuestionRegistry

from .matrices import MatrixQuestion

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
  
  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = cls.get_rng(rng_seed)
    digits = cls.get_digits_to_round(digits_to_round=kwargs.get("digits_to_round"))

    seq_len = kwargs.get("seq_len", 3)
    input_dim = kwargs.get("input_dim", 1)
    hidden_dim = kwargs.get("hidden_dim", 1)

    x_seq = cls.get_rounded_matrix(rng, (seq_len, input_dim), digits_to_round=digits)
    W_xh = cls.get_rounded_matrix(rng, (input_dim, hidden_dim), -1, 2, digits)
    W_hh = cls.get_rounded_matrix(rng, (hidden_dim, hidden_dim), -1, 2, digits)
    b_h = cls.get_rounded_matrix(rng, (hidden_dim,), -1, 2, digits)
    h_0 = np.zeros(hidden_dim)

    h_states = cls.rnn_forward(x_seq, W_xh, W_hh, b_h, h_0)

    return {
      "digits": digits,
      "seq_len": seq_len,
      "input_dim": input_dim,
      "hidden_dim": hidden_dim,
      "x_seq": x_seq,
      "W_xh": W_xh,
      "W_hh": W_hh,
      "b_h": b_h,
      "h_0": h_0,
      "h_states": h_states,
    }

  @classmethod
  def _build_body(cls, context) -> Tuple[ca.Section, List[ca.Answer]]:
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
      cls.create_info_table(
        {
          ca.Container([ca.Text("Input sequence, "), ca.Equation("x_{seq}", inline=True)]) : ca.Matrix(context["x_seq"]),
          ca.Container([ca.Text("Input weights, "),  ca.Equation("W_{xh}", inline=True)])  : ca.Matrix(context["W_xh"]),
          ca.Container([ca.Text("Hidden weights, "), ca.Equation("W_{hh}", inline=True)])  : ca.Matrix(context["W_hh"]),
          ca.Container([ca.Text("Bias, "),           ca.Equation("b_{h}", inline=True)])   : ca.Matrix(context["b_h"]),
          ca.Container([ca.Text("Hidden states, "),  ca.Equation("h_{0}", inline=True)])   : ca.Matrix(context["h_0"]),
        }
      )
    )

    body.add_element(ca.LineBreak())

    output_answer = ca.AnswerTypes.Matrix(value=context["h_states"], label="Hidden states")
    answers.append(output_answer)
    body.add_element(output_answer)

    return body, answers

  @classmethod
  def _build_explanation(cls, context) -> Tuple[ca.Section, List[ca.Answer]]:
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
    seq_len = len(context["x_seq"])
    num_examples = min(2, seq_len)

    explanation.add_element(ca.Paragraph([""]))

    for t in range(num_examples):
      explanation.add_element(
        ca.Paragraph([
          ca.Text(f"Example: Timestep {t}", emphasis=True)
        ])
      )

      # Compute step t
      x_contribution = context["x_seq"][t] @ context["W_xh"]
      if t == 0:
        h_prev = context["h_0"]
        h_prev_label = 'h_{-1}'
        h_prev_desc = " (initial state)"
      else:
        h_prev = context["h_states"][t-1]
        h_prev_label = f'h_{{{t-1}}}'
        h_prev_desc = ""

      h_contribution = h_prev @ context["W_hh"]
      pre_activation = x_contribution + h_contribution + context["b_h"]
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
      ca.Matrix(np.round(context["h_states"], digits))
    )

    return explanation, []
