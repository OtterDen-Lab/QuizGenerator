#!env python
import abc

import numpy as np

from QuizGenerator.question import Question


class MatrixQuestion(Question, abc.ABC):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.default_digits_to_round = kwargs.get("digits_to_round", 2)
  
  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)
    self.rng = np.random.RandomState(rng_seed)
  
  def get_rounded_matrix(self, shape, low=0, high=1, digits_to_round=None):
    if digits_to_round is None:
      digits_to_round = self.default_digits_to_round
    return np.round(
      (high - low) * self.rng.rand(*shape) + low,
      digits_to_round
    )
