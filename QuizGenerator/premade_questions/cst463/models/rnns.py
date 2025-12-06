import abc
import logging
import math
import keras
import numpy as np

from .matrices import MatrixQuestion
from QuizGenerator.question import Question, QuestionRegistry, Answer
from QuizGenerator.contentast import ContentAST
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
    
    self.answers["output_sequence"] = Answer.matrix(key="output_sequence", value=self.h_states)
    
    return True
  
  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()
    
    body.add_element(
      ContentAST.Paragraph([
        ContentAST.Text("Given the below information about an RNN, please calculate the output sequence."),
        "Assume that you are using a tanh activation function."
      ])
    )
    body.add_element(
      self.create_info_table(
        {
          ContentAST.Container(["Input sequence, ", ContentAST.Equation("x_{seq}", inline=True)]) : ContentAST.Matrix(self.x_seq),
          ContentAST.Container(["Input weights, ",  ContentAST.Equation("W_{xh}", inline=True)])  : ContentAST.Matrix(self.W_xh),
          ContentAST.Container(["Hidden weights, ", ContentAST.Equation("W_{hh}", inline=True)])  : ContentAST.Matrix(self.W_hh),
          ContentAST.Container(["Bias, ",           ContentAST.Equation("b_{h}", inline=True)])   : ContentAST.Matrix(self.b_h),
          ContentAST.Container(["Hidden states, ",  ContentAST.Equation("h_{0}", inline=True)])   : ContentAST.Matrix(self.h_0),
        }
      )
    )
    
    body.add_element(
      self.answers["output_sequence"].get_ast_element(label=f"Hidden states")
    )
    
    return body
  
  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    return explanation

