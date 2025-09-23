#!env python
"""
Common constants used across question types.
Centralizing these values makes it easier to maintain consistency 
and adjust ranges globally.
"""

# Bit-related constants
class BitRanges:
  DEFAULT_MIN_BITS = 3
  DEFAULT_MAX_BITS = 16
  
  # Memory addressing specific (using standardized names)
  DEFAULT_MIN_BITS_VA = 5
  DEFAULT_MAX_BITS_VA = 10
  DEFAULT_MIN_BITS_OFFSET = 3
  DEFAULT_MAX_BITS_OFFSET = 8
  DEFAULT_MIN_BITS_VPN = 3
  DEFAULT_MAX_BITS_VPN = 8
  DEFAULT_MIN_BITS_PFN = 3
  DEFAULT_MAX_BITS_PFN = 16

  # Backward compatibility - deprecated, use standardized names above
  DEFAULT_MIN_VA_BITS = DEFAULT_MIN_BITS_VA
  DEFAULT_MAX_VA_BITS = DEFAULT_MAX_BITS_VA
  DEFAULT_MIN_OFFSET_BITS = DEFAULT_MIN_BITS_OFFSET
  DEFAULT_MAX_OFFSET_BITS = DEFAULT_MAX_BITS_OFFSET
  DEFAULT_MIN_VPN_BITS = DEFAULT_MIN_BITS_VPN
  DEFAULT_MAX_VPN_BITS = DEFAULT_MAX_BITS_VPN
  DEFAULT_MIN_PFN_BITS = DEFAULT_MIN_BITS_PFN
  DEFAULT_MAX_PFN_BITS = DEFAULT_MAX_BITS_PFN
  
  # Base and bounds
  DEFAULT_MAX_ADDRESS_BITS = 32
  DEFAULT_MIN_BOUNDS_BITS = 5
  DEFAULT_MAX_BOUNDS_BITS = 16

# Job/Process constants
class ProcessRanges:
  DEFAULT_MIN_JOBS = 2
  DEFAULT_MAX_JOBS = 5
  DEFAULT_MIN_DURATION = 2
  DEFAULT_MAX_DURATION = 10
  DEFAULT_MIN_ARRIVAL_TIME = 0
  DEFAULT_MAX_ARRIVAL_TIME = 20

# Cache/Memory constants
class CacheRanges:
  DEFAULT_MIN_CACHE_SIZE = 2
  DEFAULT_MAX_CACHE_SIZE = 8
  DEFAULT_MIN_ELEMENTS = 3
  DEFAULT_MAX_ELEMENTS = 10
  DEFAULT_MIN_REQUESTS = 5
  DEFAULT_MAX_REQUESTS = 20

# Disk/IO constants
class IOConstants:
  DEFAULT_MIN_RPM = 3600
  DEFAULT_MAX_RPM = 15000
  DEFAULT_MIN_SEEK_DELAY = 3.0
  DEFAULT_MAX_SEEK_DELAY = 20.0
  DEFAULT_MIN_TRANSFER_RATE = 50
  DEFAULT_MAX_TRANSFER_RATE = 300

# Math question constants
class MathRanges:
  DEFAULT_MIN_MATH_BITS = 3
  DEFAULT_MAX_MATH_BITS = 49


# =============================================================================
# Utility Functions
# =============================================================================

def get_bit_range(bit_type: str) -> tuple[int, int]:
  """
  Get the min/max range for a specific type of bit parameter.

  Args:
      bit_type: One of 'va', 'offset', 'vpn', 'pfn', 'general'

  Returns:
      Tuple of (min_bits, max_bits)

  Raises:
      ValueError: If bit_type is not recognized
  """
  ranges = {
    'va': (BitRanges.DEFAULT_MIN_BITS_VA, BitRanges.DEFAULT_MAX_BITS_VA),
    'offset': (BitRanges.DEFAULT_MIN_BITS_OFFSET, BitRanges.DEFAULT_MAX_BITS_OFFSET),
    'vpn': (BitRanges.DEFAULT_MIN_BITS_VPN, BitRanges.DEFAULT_MAX_BITS_VPN),
    'pfn': (BitRanges.DEFAULT_MIN_BITS_PFN, BitRanges.DEFAULT_MAX_BITS_PFN),
    'general': (BitRanges.DEFAULT_MIN_BITS, BitRanges.DEFAULT_MAX_BITS),
  }
  
  if bit_type not in ranges:
    raise ValueError(f"Unknown bit_type '{bit_type}'. Valid types: {list(ranges.keys())}")
  
  return ranges[bit_type]


def get_process_range(param_type: str) -> tuple[int, int]:
  """
  Get the min/max range for job/process parameters.

  Args:
      param_type: One of 'jobs', 'duration', 'arrival_time'

  Returns:
      Tuple of (min_value, max_value)

  Raises:
      ValueError: If param_type is not recognized
  """
  ranges = {
    'jobs': (ProcessRanges.DEFAULT_MIN_JOBS, ProcessRanges.DEFAULT_MAX_JOBS),
    'duration': (ProcessRanges.DEFAULT_MIN_DURATION, ProcessRanges.DEFAULT_MAX_DURATION),
    'arrival_time': (ProcessRanges.DEFAULT_MIN_ARRIVAL_TIME, ProcessRanges.DEFAULT_MAX_ARRIVAL_TIME),
  }
  
  if param_type not in ranges:
    raise ValueError(f"Unknown param_type '{param_type}'. Valid types: {list(ranges.keys())}")
  
  return ranges[param_type]


def get_cache_range(param_type: str) -> tuple[int, int]:
  """
  Get the min/max range for cache/memory parameters.

  Args:
      param_type: One of 'cache_size', 'elements', 'requests'

  Returns:
      Tuple of (min_value, max_value)

  Raises:
      ValueError: If param_type is not recognized
  """
  ranges = {
    'cache_size': (CacheRanges.DEFAULT_MIN_CACHE_SIZE, CacheRanges.DEFAULT_MAX_CACHE_SIZE),
    'elements': (CacheRanges.DEFAULT_MIN_ELEMENTS, CacheRanges.DEFAULT_MAX_ELEMENTS),
    'requests': (CacheRanges.DEFAULT_MIN_REQUESTS, CacheRanges.DEFAULT_MAX_REQUESTS),
  }
  
  if param_type not in ranges:
    raise ValueError(f"Unknown param_type '{param_type}'. Valid types: {list(ranges.keys())}")
  
  return ranges[param_type]
