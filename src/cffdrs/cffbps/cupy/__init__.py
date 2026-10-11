"""Experimental CuPy backend; requires CuPy and a CUDA environment."""
try:
    from .core import FBP
except ImportError as exc:
    raise ImportError(
        'cffdrs.cffbps.cupy needs CuPy and CUDA; install with pip install "cffdrs[gpu]"'
    ) from exc

__all__ = ["FBP"]
