# cffdrs

Python tools for the **Canadian Forest Fire Danger Rating System (CFFDRS)**: the
Fire Behavior Prediction (FBP) System and the Fire Weather Index (FWI) System.

Calculations accept scalars or NumPy arrays (e.g. raster grids) through the same
API, propagate NoData/NaN cells via masked arrays, and include a block-based
multiprocessing driver for large rasters.

## Installation

```bash
pip install cffdrs
```

A tested prebuilt wheel includes Rust and needs no Rust compiler on the consumer's
machine. Building a source distribution requires Python 3.11+, a Rust toolchain
and the platform linker/build tools. Version 0.2.0 is local development work;
this change does not publish a release. Use a tested wheel artifact for this version.

Optional extras:

```bash
pip install "cffdrs[io]"    # rasterio + pandas, for raster/tabular workflows
```

Requires Python 3.11+. Core dependencies: `numpy`, `scipy`, `psutil`.

## Package layout

| Module | Contents |
|---|---|
| `cffdrs.cffbps` | Automatic FBP dispatch: Rust if importable, otherwise Python |
| `cffdrs.cffbps.python` | Explicit Python FBP backend; `python/core.py` orchestrates `python/equations/` |
| `cffdrs.cffbps.common` | Shared constants, fuel-code conversion, and input validation |
| `cffdrs.cffbps.rust` | Explicit Rust FBP backend over the owned native grid |
| `cffdrs.cffbps.cupy` | Experimental CuPy GPU backend |
| `cffdrs.cffwis` | FWI System: FFMC/DMC/DC/ISI/BUI/FWI/DSR functions (hourly + daily) |
| `cffdrs.diurnal_ffmc_lawson` | Diurnal (hourly) FFMC interpolation per Lawson et al. |
| `cffdrs._rust` | Compiled, low-level FBP grid pass used as the performance backend |

The repository is a mixed Python/Rust project. Python package sources live under
`src/cffdrs`; the top-level `rust/` directory is a Cargo workspace containing:

- `cffbps-rust`: dependency-free Rust FBP equations and the grid pass.
- `cffdrs-py`: thin PyO3 bindings that install as `cffdrs._rust` in the same
  `cffdrs` wheel as the Python package.

The Python implementation is the behavioral reference. The Python distribution
(`cffdrs-py`) is **0.2.0**; the independent `cffbps-rust` core is **0.3.0**.
`cffdrs._rust.FbpGrid` is a low-level stateful grid in a private extension module;
normal Python callers use `FBP`, which prepares inputs and exports reference-compatible
results. Keeping `rust/` beside `src/` separates the Cargo workspace from Python imports.

Wheels use `cp311-abi3` (one per platform). The release workflow tests Python
3.11–3.14. Future CPython ABI compatibility does not certify future NumPy/SciPy
support or untested dependency combinations. Free-threaded Python is unsupported.

## Fire Behavior Prediction (`cffdrs.cffbps`)

The `FBP` class is a facade: initialize it with fuel, terrain, and weather inputs,
run the model, and request any of 50+ output variables. The underlying equations
live in `cffdrs.cffbps.python.equations` as pure, typed functions if you need them directly.

### Breaking changes in 0.2.0

The public `from cffdrs.cffbps import FBP` import remains supported and now
selects a backend at initialization. Import from `cffdrs.cffbps.python` for
the explicit reference. The following previous paths are historical migration
references, not current import targets.

Deep imports move without compatibility shims:

| Previous import | New import |
|---|---|
| `cffdrs.cffbps.facade` | `cffdrs.cffbps.python.core` |
| `cffdrs.cffbps.equations` | `cffdrs.cffbps.python.equations` |
| `cffdrs.cffbps.parallel` | `cffdrs.cffbps.python.parallel` |
| `cffdrs.cffbps.constants` | `cffdrs.cffbps.common.constants` |
| `cffdrs.cffbps.inputs` | `cffdrs.cffbps.common.inputs` |
| `cffdrs.cffbps.cupy_backend` | `cffdrs.cffbps.cupy` (implementation: `cupy/core.py`) |

### Choosing a backend

```python
from cffdrs.cffbps import FBP                  # automatic CPU selection
from cffdrs.cffbps.python import FBP as PythonFBP
from cffdrs.cffbps.rust import FBP as RustFBP  # requires the extension
```

`FBP.initialize(..., backend='auto', threads=4)` chooses Rust when
`cffdrs._rust` imports, otherwise Python. `backend='python'` forces the
reference; `backend='rust'` raises `ImportError` when the extension is missing.
`backend_in_use` is `None` before initialization and then `'python'` or `'rust'`.
The backend stays fixed for that instance, including reinitialization. Create a
new instance to change it. Calculation errors do not trigger fallback or replay.
Reading or changing calibration tables before initialization pins auto to Python.
Explicit imports never dispatch and do not take a `backend=` selector.

Rust threads default to **1**; **0** uses all available cores. Keep 1 for many
simultaneous fires. Python ignores the automatic interface's `threads` argument.
Rust floating results agree with Python to **rtol 1e-9**, with matching missing
values; integer results agree exactly. Bit identity with NumPy is not promised.

On Rust, access computed results through `getParams(['hros', ...])`. Per-instance
lookup-table calibration (`rosParams`, `open_fuel_types`, etc.), result attribute
reads, `runFBP(block=...)`, `getCBH_CFL(ftype=..., cbh=..., cfl=...)` overrides,
and `fbpMultiprocessArray` require Python. Choose `backend='python'` before
initialization for those workflows. For Rust canopy injection, run
`getCBH_CFL()` and then `setParams({'cbh': ..., 'cfl': ...})` in a staged pipeline.
Rust uses threads for parallel work. See [the lifecycle and stage reference](docs/CODEBASE.md)
for memory retention and supported update routes.

### Scalar example

```python
from cffdrs.cffbps import FBP

fbp = FBP()
fbp.initialize(
    fuel_type=2,          # C-2 (or pass the string code 'C2')
    wx_date=20240516,     # YYYYMMDD, used for foliar moisture
    lat=62.245533, long=-133.840363, elevation=1180,
    slope=8, aspect=60,   # slope %, aspect degrees
    ws=24, wd=266,        # wind speed km/h, direction degrees
    ffmc=92, bui=31,      # CFFWIS codes
    out_request=['fire_type', 'hros', 'hfi'],
    backend='auto', threads=4,  # omit for auto selection and one thread
)
fire_type, hros, hfi = fbp.runFBP()
```

### Array / raster example

Any spatial input may be a NumPy array (all arrays must share one shape); outputs
come back as plain arrays. Missing floating values are NaN; integer outputs
have field-specific missing values (see below):

```python
import numpy as np
from cffdrs.cffbps import FBP

fuel_type = np.array([[2, 3, 7], [8, 14, 19]], dtype=np.int8)  # 19 = non-fuel
shape = fuel_type.shape

fbp = FBP()
fbp.initialize(
    fuel_type=fuel_type, wx_date=20240516,
    lat=np.full(shape, 62.2455), long=np.full(shape, -133.8404),
    elevation=np.full(shape, 1180.0), slope=np.full(shape, 8.0),
    aspect=np.full(shape, 60.0), ws=np.full(shape, 24.0),
    wd=np.full(shape, 266.0), ffmc=np.full(shape, 92.0),
    bui=np.full(shape, 31.0),
    out_request=['hros', 'hfi', 'fire_type'],
)
hros, hfi, fire_type = fbp.runFBP()
```

For large rasters, use Rust `threads=`. Python-only
`cffdrs.cffbps.python.fbpMultiprocessArray(...)` splits inputs into worker blocks.

### Available outputs

`out_request` accepts any of the names in `cffdrs.cffbps.valid_outputs` — including
final outputs (`hros`, `hfi`, `fire_type`, `cfb`, `tfc`, `fi_class`, …) and
intermediates (`isi`, `wsv`, `raz`, `sfc`, `fmc`, `csfi`, `rso`, `be`, …).

### Output contract

`runFBP()` and `getParams(names)` return a list in requested order. Array exports
are plain `numpy.ndarray`, never masked arrays. Most computed floats are
`float64`; retained integer inputs (`ws`, `wd`, `ffmc`, `bui`), integer `d0`/`nd`
and injected results can preserve their dtype. Prepared array fuel is normally
`int8`; scalar fuel mixed with arrays can retain `int64`. `fire_type` and
`fi_class` arrays are `int64` (masked values 0 and -99 respectively). `dj` is
`int64` when complete, `float64` with NaN when an override is missing. Fuel
exports retain the prepared code beneath a mask. Missing values follow each
field's dependencies, not a blanket mask of every output.

All-scalar inputs export Python scalars. Legacy masked/non-modeled cases can
export floating zero even for nominal integer fields; missing explicit dates
export NaN. Rust expands every result to the full grid shape; Python can keep
scalar-derived results at `(1,)` and has legacy 0-d export differences.
See [the detailed output contract](docs/CODEBASE.md#outputs-and-intentional-differences)
for these measured exceptions.

### Experimental GPU backend

An experimental CuPy (GPU) variant of the FBP model lives at
`cffdrs.cffbps.cupy`. Install the optional `gpu` extra (needs a
CUDA-capable environment):

```bash
pip install "cffdrs[gpu]"
# On CUDA, prefer the version-matched wheel instead, e.g.:
#   pip install cupy-cuda12x
```

```python
from cffdrs.cffbps.cupy import FBP   # requires cupy + CUDA
```

> **Experimental — not held to reference parity.** This backend is frozen at the
> pre-refactor monolith, defaults to `float32`, and predates the reference
> package's masked-array NaN-propagation fixes. It does **not** reproduce the
> golden snapshots and has no parity test. Use `from cffdrs.cffbps import FBP`
> (the validated CPU backends) for validated results.

## Fire Weather Index System (`cffdrs.cffwis`)

Functions for the daily and hourly FWI codes; scalar or array inputs.

```python
from cffdrs import cffwis

ffmc = cffwis.dailyFFMC(ffmc0=85, temp=15, rh=50, wind=10, precip=0)
dmc  = cffwis.dailyDMC(dmc0=6, temp=15, rh=50, precip=0, month=6)
dc   = cffwis.dailyDC(dc0=15, temp=15, precip=0, month=6)
isi  = cffwis.dailyISI(wind=10, ffmc=ffmc)
bui  = cffwis.dailyBUI(dmc=dmc, dc=dc)
fwi  = cffwis.dailyFWI(isi=isi, bui=bui)
```

Hourly FFMC (Van Wagner 1977 / Alexander et al. 1984) is available via
`cffwis.hourlyFFMC`, and the Lawson diurnal interpolation via
`cffwis.diurnalFFMC_lawson` / `cffdrs.diurnal_ffmc_lawson`.

## Validation

FBP outputs are validated against the published test cases in Wotton, Alexander &
Taylor (2009), *Updates and revisions to the 1992 Canadian Forest Fire Behavior
Prediction System* (all 20 reference cases; fire type 20/20, core metrics within
cross-implementation rounding). The test suite locks all 54 outputs to
full-precision golden snapshots on both the scalar and array code paths.

## Development

Development requires Python 3.11+, `uv`, and a stable Rust toolchain. The main
project uses Maturin, so syncing the environment builds `cffdrs._rust` alongside
the editable Python package:

```bash
git clone https://github.com/gagreene/cffdrs.git
cd cffdrs
uv sync --extra test --extra dev   # editable mixed Python/Rust install + tooling
uv run --no-sync python -m pytest tests -q -p no:cacheprovider  # both backends
uv run --no-sync ruff check src/ tests/cffbps  # lint
uv run --no-sync mypy               # type-check the equation core
cargo test --manifest-path rust/Cargo.toml -p cffbps-rust
```

The Python distribution version comes from `rust/crates/cffdrs-py/Cargo.toml`;
update that crate version when preparing a release tag. The pure Rust core has
its own version in `rust/crates/cffbps-rust/Cargo.toml` because it may be consumed
independently by Rust applications.

Existing golden fixtures are the behavior-preservation oracle. Do not regenerate
them to accommodate a refactor. Release publishing remains disabled in the
workflow; version tags and publication require owner authorization.

## License

MIT — see [LICENSE](LICENSE). © 2024 Gregory A. Greene.
