import abc
import logging
import math
import keras
import numpy as np

from QuizGenerator.question import Question, QuestionRegistry, Answer
from QuizGenerator.contentast import ContentAST
from QuizGenerator.constants import MathRanges
from QuizGenerator.mixins import TableQuestionMixin

log = logging.getLogger(__name__)


@QuestionRegistry.register("cst463.rnn.forward-pass")
class RNNForwardPass(Question, TableQuestionMixin):
  
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
    
    seq_len = 3
    input_dim = 2
    hidden_dim = 2
    
    # Small integer weights for hand calculation
    self.x_seq = np.random.randint(0, 3, size=(seq_len, input_dim))
    self.W_xh = np.random.randint(-1, 2, size=(input_dim, hidden_dim))
    self.W_hh = np.random.randint(-1, 2, size=(hidden_dim, hidden_dim))
    self.b_h = np.random.randint(-1, 2, size=hidden_dim)
    self.h_0 = np.zeros(hidden_dim)
    
    self.h_states = self.rnn_forward(self.x_seq, self.W_xh, self.W_hh, self.b_h, self.h_0)
    
    ## Answers:
    # x_seq, W_xh, W_hh, b_h, h_0, h_states
    
    return True
  
  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()
    
    body.add_element(
      ContentAST.Text("Given the below information about an RNN, please calculate the output sequence.")
    )
    body.add_element(
      self.create_info_table(
        {
          "x_seq": self.x_seq,
          "W_xh" : self.W_xh,
          "W_hh" : self.W_hh,
          "b_h"  : self.b_h,
          "h_0"  : self.h_0,
        }
      )
    )
    
    log.debug(f"h_states: {self.h_states}")
    
    return body
  
  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    return explanation

