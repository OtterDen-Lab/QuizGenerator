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
    
    # Memory addressing specific
    DEFAULT_MIN_VA_BITS = 5
    DEFAULT_MAX_VA_BITS = 10
    DEFAULT_MIN_OFFSET_BITS = 3
    DEFAULT_MAX_OFFSET_BITS = 8
    DEFAULT_MIN_VPN_BITS = 3
    DEFAULT_MAX_VPN_BITS = 8
    DEFAULT_MIN_PFN_BITS = 3
    DEFAULT_MAX_PFN_BITS = 16
    
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