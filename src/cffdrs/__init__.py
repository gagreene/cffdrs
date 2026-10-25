"""cffdrs — Canadian Forest Fire Danger Rating System tools.

Subpackages / modules:
- ``cffdrs.cffbps``          Fire Behavior Prediction System (FBP)
- ``cffdrs.cffbps.python``   Explicit Python reference backend
- ``cffdrs.cffbps.common``   Shared FBP constants and input helpers
- ``cffdrs.cffbps.cupy``     Experimental CuPy backend
- ``cffdrs.cffwis``          Fire Weather Index System (FWI)
- ``cffdrs.diurnal_ffmc_lawson``  Diurnal FFMC (Lawson) interpolation

Import the pieces you need, e.g. ``from cffdrs.cffbps import FBP``.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cffdrs-fire")
except PackageNotFoundError:  # fresh checkout that has never been installed
    __version__ = "0.0.0+unknown"
