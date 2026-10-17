"""Conversion at the owned Rust grid boundary."""
import numpy as np


def to_nan(value):
    """Copy numeric data as float64, preserving missing cells as NaN."""
    if value is None:
        return None
    arr = np.ma.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr.filled(np.nan))
    return np.array(arr.filled(np.nan), dtype=np.float64, copy=True)


def finish(name, array, scalar_input, *, dtype=None):
    """Export an owned plain array or a Python scalar with the reference dtype."""
    result = np.array(array, copy=True)
    if dtype is not None:
        result = result.astype(dtype)
    elif name in ('fire_type', 'fi_class'):
        result = np.where(np.isnan(result), 0 if name == 'fire_type' else -99, result).astype(np.int64)
    elif name == 'dj' and not np.isnan(result).any():
        result = result.astype(np.int64)
    else:
        result = result.astype(np.float64)
    return result.item(0) if scalar_input else result
