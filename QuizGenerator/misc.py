#!env python
from __future__ import annotations

import logging



log = logging.getLogger(__name__)


def fix_negative_zero(value):
  """Convert -0.0 to 0.0 to avoid confusing display."""
  if isinstance(value, (int, float)):
    return 0.0 if value == 0 else value
  return value
