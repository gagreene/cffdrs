"""cffdrs — Canadian Forest Fire Danger Rating System tools.

Subpackages / modules:
- ``cffdrs.cffbps``          Fire Behavior Prediction System (FBP)
- ``cffdrs.cffwis``          Fire Weather Index System (FWI)
- ``cffdrs.diurnal_ffmc_lawson``  Diurnal FFMC (Lawson) interpolation

Import the pieces you need, e.g. ``from cffdrs.cffbps import FBP``.
"""
try:
    # Written by the hatch-vcs build hook (from git tags) at build/install time.
    from cffdrs._version import __version__
except ImportError:  # fresh checkout that has never been built/installed
    __version__ = "0.0.0+unknown"
