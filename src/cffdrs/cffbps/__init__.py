"""Canadian Forest Fire Behavior Prediction System, with a lazy Python API."""
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .python import (
        FBP,
        convert_grid_codes,
        fbpFTCode_AlphaToNum_LUT,
        fbpFTCode_NumToAlpha_LUT,
        fbpMultiprocessArray,
        getSeasonGrassCuring,
        valid_outputs,
    )

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


def __getattr__(name: str) -> Any:
    """Load the explicit Python backend when a public name is requested."""
    if name in __all__:
        from . import python

        return getattr(python, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
