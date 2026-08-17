"""cffdrs — Canadian Forest Fire Danger Rating System tools.

Subpackages / modules:
- ``cffdrs.cffbps``          Fire Behavior Prediction System (FBP)
- ``cffdrs.cffwis``          Fire Weather Index System (FWI)
- ``cffdrs.diurnal_ffmc_lawson``  Diurnal FFMC (Lawson) interpolation

Import the pieces you need, e.g. ``from cffdrs.cffbps import FBP``.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cffdrs")
except PackageNotFoundError:  # fresh checkout that has never been installed
    __version__ = "0.0.0+unknown"
