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


@QuestionRegistry.register("cst463.attention.forward-pass")
class AttentionForwardPass(Question, TableQuestionMixin):
  
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
    self.rng = np.random.RandomState(kwargs.get("rng_seed", None))
    
    seq_len = 3
    d_k = 2  # key/query dimension
    d_v = 2  # value dimension
    
    # Small integer matrices
    self.Q = self.rng.randint(0, 3, size=(seq_len, d_k))
    self.K = self.rng.randint(0, 3, size=(seq_len, d_k))
    self.V = self.rng.randint(0, 3, size=(seq_len, d_v))
    
    self.output, self.weights = self.simple_attention(self.Q, self.K, self.V)
    
    ## Answers:
    # Q, K, V, output, weights
    
    return True
  
  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()
    
    body.add_element(
      ContentAST.Text("Given the below information about a self attention layer, please calculate the output sequence.")
    )
    body.add_element(
      self.create_info_table(
        {
          "Q": ContentAST.Matrix(self.Q),
          "K": ContentAST.Matrix(self.K),
          "V": ContentAST.Matrix(self.V),
        }
      )
    )
    
    log.debug(f"output: {self.output}")
    log.debug(f"weights: {self.weights}")
    
    return body
  
  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    return explanation

