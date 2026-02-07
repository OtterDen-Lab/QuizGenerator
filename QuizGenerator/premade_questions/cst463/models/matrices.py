#!/usr/bin/env python
import abc

import numpy as np

from QuizGenerator.question import Question


class MatrixQuestion(Question, abc.ABC):
  default_digits_to_round = 2

  @staticmethod
  def get_rng(rng_seed):
    return np.random.RandomState(rng_seed)

  @classmethod
  def get_digits_to_round(cls, *, digits_to_round=None):
    if digits_to_round is None:
      digits_to_round = cls.default_digits_to_round
    return digits_to_round

  @staticmethod
  def get_rounded_matrix(*args, **kwargs):
    # Supports (rng, shape, ...) or (self, shape, ...)
    if not args:
      raise ValueError("get_rounded_matrix requires at least a rng or self plus shape.")

    if isinstance(args[0], MatrixQuestion):
      self = args[0]
      shape = args[1]
      remaining = list(args[2:])
      rng = kwargs.pop("rng", getattr(self, "_np_rng", np.random.RandomState()))
      low = kwargs.pop("low", None)
      high = kwargs.pop("high", None)
      digits_to_round = kwargs.pop("digits_to_round", None)

      if remaining:
        low = remaining.pop(0)
      if remaining:
        high = remaining.pop(0)
      if remaining:
        digits_to_round = remaining.pop(0)

      if low is None:
        low = 0
      if high is None:
        high = 1
      digits_to_round = self.get_digits_to_round(digits_to_round=digits_to_round)
    else:
      rng = args[0]
      shape = args[1]
      remaining = list(args[2:])
      low = kwargs.pop("low", None)
      high = kwargs.pop("high", None)
      digits_to_round = kwargs.pop("digits_to_round", None)

      if remaining:
        low = remaining.pop(0)
      if remaining:
        high = remaining.pop(0)
      if remaining:
        digits_to_round = remaining.pop(0)

      if low is None:
        low = 0
      if high is None:
        high = 1
      if digits_to_round is None:
        digits_to_round = 2

    return np.round(
      (high - low) * rng.rand(*shape) + low,
      digits_to_round
    )
