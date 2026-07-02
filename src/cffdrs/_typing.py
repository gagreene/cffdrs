"""Shared type aliases for the cffdrs package.

Kept deliberately small; modules import these to keep signatures readable. Adopted
incrementally (the equation modules currently alias ``MaskedArray`` locally).
"""
from __future__ import annotations

import numpy as np
from numpy import ma as mask
from numpy.typing import NDArray

# A single numeric value.
Scalar = int | float

# Either a scalar or an ndarray input, as accepted by FBP.initialize.
ArrayLike = Scalar | np.ndarray

# Masked float array — the working type flowing through the equation functions.
MaskedArray = mask.MaskedArray

# A dense float64 ndarray (e.g. raster block).
FloatArray = NDArray[np.float64]
