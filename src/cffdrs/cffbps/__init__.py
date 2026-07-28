"""Canadian Forest Fire Behavior Prediction System (CFFBPS).

Public API for the ``cffbps`` package. The implementation is decomposed into a
facade (:class:`~cffbps.facade.FBP`) over focused constant/validation/equation
modules; this ``__init__`` aggregates the historical
import surface (``from cffbps import FBP``).
"""
from .constants import (
    fbpFTCode_AlphaToNum_LUT,
    fbpFTCode_NumToAlpha_LUT,
    valid_outputs,
)
from .facade import FBP, getSeasonGrassCuring
from .inputs import convert_grid_codes
from .parallel import fbpMultiprocessArray

__author__ = ["Gregory A. Greene, map.n.trowel@gmail.com"]

__all__ = [
    "FBP",
    "convert_grid_codes",
    "fbpFTCode_AlphaToNum_LUT",
    "fbpFTCode_NumToAlpha_LUT",
    "fbpMultiprocessArray",
    "getSeasonGrassCuring",
    "valid_outputs",
]
