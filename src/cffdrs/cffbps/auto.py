"""Choose the FBP backend once, when an instance is initialized.

Rust is preferred when its extension imports; otherwise auto silently uses Python.
Explicit Python never imports Rust. Threads apply only to Rust and are ignored by
Python. Rust float results agree with Python to rtol 1e-9, and integer results
agree exactly, including the documented conditional missing-value exports.
"""
import importlib
from typing import Any

from .python import FBP as PythonFBP

_TABLES = {
    'rosParams', 'open_fuel_types', 'non_crowning_fuels', 'fbpCBH_CFL_HT_LUT',
    'csfiVarList', 'rsoVarList', 'cfbVarList', 'cfcVarList', 'cfiVarList', 'ftypes',
}
_RUST_INSTALL = (
    'Rust FBP requires the cffdrs extension: install with pip install cffdrs '
    'or build the checkout with uv sync --extra test --extra dev'
)


def available_backends() -> list[str]:
    """Return importable backends; this explicit query may import Rust and CuPy."""
    backends = ['python']
    # probe the backend modules themselves: cupy alone can be present while the CuPy
    # facade still fails to import (for example without rasterio)
    for name, module in [('rust', 'cffdrs.cffbps.rust'), ('cupy', 'cffdrs.cffbps.cupy')]:
        try:
            importlib.import_module(module)
        except ImportError:
            continue
        backends.append(name)
    return backends


class AutoFBP:
    """Forward operations to the backend selected at the first initialize.

    Reading or assigning a Python calibration table before initialize pins the
    instance to Python, preserving its mutable tables. After initialize the
    backend never changes, including on repeated initialize calls. A conflicting
    explicit backend request raises ValueError. Unsupported Rust operations fail
    with advice to create an instance using backend='python'; there is no replay
    or computation fallback.
    """

    def __getattr__(self, name: str) -> Any:
        if name in _TABLES:
            if self._backend_in_use == 'rust':
                raise AttributeError(f"{name} is a Python table; use backend='python'")
            if self._backend_in_use is None:
                object.__setattr__(self, '_python_pinned', True)
        return getattr(self._backend, name)

    def __init__(self) -> None:
        object.__setattr__(self, '_backend', PythonFBP())
        object.__setattr__(self, '_backend_in_use', None)
        object.__setattr__(self, '_python_pinned', False)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'backend_in_use':
            raise AttributeError('backend_in_use is read-only')
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return
        if name in _TABLES:
            if self._backend_in_use == 'rust':
                raise AttributeError(f"{name} is a Python table; use backend='python'")
            if self._backend_in_use is None:
                object.__setattr__(self, '_python_pinned', True)
        setattr(self._backend, name, value)

    @property
    def backend_in_use(self) -> str | None:
        """None before successful initialization, then 'python' or 'rust'."""
        return self._backend_in_use

    def getCBH_CFL(self, ftype=None, cbh=None, cfl=None):
        """Run the canopy lookup; Rust overrides use setParams after this stage."""
        if self._backend_in_use == 'rust' and any(value is not None for value in (ftype, cbh, cfl)):
            raise NotImplementedError(
                "getCBH_CFL overrides require backend='python'; "
                "on Rust use setParams({'cbh': ..., 'cfl': ...}) after this stage"
            )
        return self._backend.getCBH_CFL(ftype=ftype, cbh=cbh, cfl=cfl)

    def initialize(self, fuel_type=None, wx_date=None, lat=None, long=None, elevation=None,
                   slope=None, aspect=None, ws=None, wd=None, ffmc=None, bui=None,
                   pc=50, pdf=35, gfl=0.35, gcf=80, d0=None, dj=None, out_request=None,
                   convert_fuel_type_codes=False, percentile_growth=50, fmc=None, *, backend='auto', threads=1):
        """Initialize with Python's inputs plus a fixed backend and Rust threads.

        Auto silently uses Python when Rust is unavailable or a calibration table
        was accessed. Forcing Rust in either case raises an informative error.
        Reinitialization resets model inputs while retaining the chosen backend.
        ``threads`` is used by Rust only: 1 (default) uses the calling thread, 0 uses every
        available core, and Python ignores it.
        """
        if backend not in ('auto', 'python', 'rust'):
            raise ValueError("backend must be 'auto', 'python', or 'rust'")
        chosen = self._backend_in_use
        if chosen is not None and backend not in ('auto', chosen):
            raise ValueError(
                f"Backend was chosen once as {chosen!r}; cannot switch. "
                "Create a new FBP instance; use backend='python' for table calibration."
            )
        candidate = self._backend
        if chosen is None:
            chosen = 'python'
            if backend == 'rust' and self._python_pinned:
                raise ValueError("Calibration tables pin this instance to backend='python'")
            if backend != 'python' and not self._python_pinned:
                try:
                    importlib.import_module('cffdrs._rust')
                except ImportError as error:
                    if backend == 'rust':
                        raise ImportError(_RUST_INSTALL) from error
                else:
                    # Import errors from the facade and errors during calculation
                    # propagate; only an unavailable extension permits fallback.
                    from .rust import FBP as RustFBP
                    candidate = RustFBP()
                    chosen = 'rust'
        kwargs = dict(
            fuel_type=fuel_type, wx_date=wx_date, lat=lat, long=long, elevation=elevation,
            slope=slope, aspect=aspect, ws=ws, wd=wd, ffmc=ffmc, bui=bui, pc=pc, pdf=pdf,
            gfl=gfl, gcf=gcf, d0=d0, dj=dj, fmc=fmc, out_request=out_request,
            convert_fuel_type_codes=convert_fuel_type_codes, percentile_growth=percentile_growth,
        )
        if chosen == 'rust':
            kwargs['threads'] = threads
        candidate.initialize(**kwargs)
        object.__setattr__(self, '_backend', candidate)
        object.__setattr__(self, '_backend_in_use', chosen)

    def runFBP(self, block=None):
        """Run the chosen backend; Rust does not support Python block processing."""
        if self._backend_in_use == 'rust' and block is not None:
            raise NotImplementedError("block processing requires backend='python'; Rust uses threads")
        return self._backend.runFBP(block=block)
