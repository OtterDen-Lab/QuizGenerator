# Parameter Naming Standards

This document establishes consistent parameter naming conventions across all question types.

## Current Inconsistencies Found

### Bit-related Parameters
- ✅ `num_bits` (math_questions.py) - **STANDARD**
- ❌ `num_va_bits`, `num_offset_bits`, `num_vpn_bits`, `num_pfn_bits` (memory_questions.py) - Should be `num_bits_va`, `num_bits_offset`, etc.

### Job/Process Parameters  
- ✅ `num_jobs` - **STANDARD**
- ✅ `duration` - **STANDARD** 
- ❌ `arrival` (should be `arrival_time` for clarity)

### Cache/Memory Parameters
- ✅ `cache_size` - **STANDARD**
- ✅ `num_elements` - **STANDARD** 
- ✅ `num_requests` - **STANDARD**

## Naming Conventions

### 1. Count Parameters
Use `num_<item>` format:
- `num_jobs` (not `job_count`)
- `num_bits` (not `bits_count`, `bit_count`)  
- `num_requests` (not `request_count`)
- `num_elements` (not `element_count`)

### 2. Size Parameters  
Use `<item>_size` format:
- `cache_size` (not `size_cache`)
- `memory_size` (not `size_memory`)

### 3. Time Parameters
Use `<event>_time` format:
- `arrival_time` (not just `arrival`)
- `start_time` (not `start`)
- `completion_time` (not `completion`)
- `response_time` (acceptable as is)
- `turnaround_time` (acceptable as is)

### 4. Bit-specific Parameters
For memory addressing, use descriptive suffixes:
- `num_bits_va` (virtual address bits)
- `num_bits_offset` (offset bits)  
- `num_bits_vpn` (VPN bits)
- `num_bits_pfn` (PFN bits)
- `num_bits_physical` (physical address bits)

### 5. Address Parameters
- `virtual_address` ✅
- `physical_address` ✅  
- `base_address` (not `base`)
- `bounds_value` (not `bounds` to avoid confusion with bounds checking)

## Implementation Plan

### Phase 1: Create Constants File
Create `QuizGenerator/constants.py` with common constants:
```python
# Bit ranges
DEFAULT_MIN_BITS = 3
DEFAULT_MAX_BITS = 16
DEFAULT_MIN_VA_BITS = 5  
DEFAULT_MAX_VA_BITS = 10

# Job/Process ranges
DEFAULT_MIN_JOBS = 2
DEFAULT_MAX_JOBS = 5
DEFAULT_MIN_DURATION = 2
DEFAULT_MAX_DURATION = 10

# Cache ranges
DEFAULT_MIN_CACHE_SIZE = 2
DEFAULT_MAX_CACHE_SIZE = 8
```

### Phase 2: Update Parameter Names
Priority order (most impact first):
1. **memory_questions.py** - Standardize bit naming (`num_va_bits` → `num_bits_va`)
2. **process.py** - Standardize time naming (`arrival` → `arrival_time`)  
3. **All files** - Extract magic numbers to constants

### Phase 3: Update YAML Configs
Update any YAML configuration files that reference the old parameter names.

## Backward Compatibility

During transition:
1. Accept both old and new parameter names in constructors
2. Add deprecation warnings for old names
3. Update documentation to show new names
4. Remove old names in future version

Example transition pattern:
```python
def __init__(self, num_bits_va=None, num_va_bits=None, **kwargs):
    if num_va_bits is not None:
        warnings.warn("num_va_bits is deprecated, use num_bits_va", DeprecationWarning)
        if num_bits_va is None:
            num_bits_va = num_va_bits
    self.num_bits_va = num_bits_va or kwargs.get("num_bits_va", 8)
```

## Files to Update

1. `QuizGenerator/premade_questions/memory_questions.py` - Major bit naming changes
2. `QuizGenerator/premade_questions/process.py` - Time parameter naming  
3. `QuizGenerator/premade_questions/math_questions.py` - Extract constants
4. `example_files/*.yaml` - Update parameter references (if any)