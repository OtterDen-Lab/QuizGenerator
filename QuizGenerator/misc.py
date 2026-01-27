#!env python
from __future__ import annotations

import decimal
import enum
import itertools
import logging
import math
import numpy as np
from typing import List, Dict, Tuple, Any

import fractions

import QuizGenerator.contentast as ca

log = logging.getLogger(__name__)


def fix_negative_zero(value):
  """Convert -0.0 to 0.0 to avoid confusing display."""
  if isinstance(value, (int, float)):
    return 0.0 if value == 0 else value
  return value
