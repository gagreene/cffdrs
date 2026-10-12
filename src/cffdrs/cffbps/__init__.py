"""Canadian Forest Fire Behavior Prediction System, with lazy automatic backend selection."""
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .auto import AutoFBP as FBP
    from .auto import available_backends
    from .python import (
        convert_grid_codes,
        fbpFTCode_AlphaToNum_LUT,
        fbpFTCode_NumToAlpha_LUT,
        fbpMultiprocessArray,
        getSeasonGrassCuring,
        valid_outputs,
    )

__author__ = ["Gregory A. Greene"]

__all__ = [
    "FBP",
    "available_backends",
    "convert_grid_codes",
    "fbpFTCode_AlphaToNum_LUT",
    "fbpFTCode_NumToAlpha_LUT",
    "fbpMultiprocessArray",
    "getSeasonGrassCuring",
    "valid_outputs",
]


def __getattr__(name: str) -> Any:
    """Load the dispatcher or Python helper only when requested."""
    if name in ('FBP', 'available_backends'):
        from .auto import AutoFBP, available_backends

        return AutoFBP if name == 'FBP' else available_backends
    if name in __all__:
        from . import python

        return getattr(python, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
