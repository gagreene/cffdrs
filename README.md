# cffdrs

Python tools for the **Canadian Forest Fire Danger Rating System (CFFDRS)**: the
Fire Behavior Prediction (FBP) System and the Fire Weather Index (FWI) System.

Calculations accept scalars or NumPy arrays (e.g. raster grids) through the same
API, propagate NoData/NaN cells via masked arrays, and include a block-based
multiprocessing driver for large rasters.

## Installation

Install from PyPI. The prebuilt wheel already contains the compiled Rust extension,
so no Rust compiler is needed:

```bash
pip install cffdrs-fire
```

The PyPI distribution is named `cffdrs-fire`; the import name is still `cffdrs`
(`from cffdrs.cffbps import FBP`).

Or build a **source checkout**, which builds the extension and needs the Rust
toolchain described below: `pip install .` or `uv sync` from the repository root.

Optional extras:

```bash
pip install "cffdrs-fire[io]"    # rasterio + pandas, for raster/tabular workflows
```

Core dependencies: `numpy>=1.24`, `scipy>=1.10`, `psutil>=5.9` (pip chooses newer
releases where an interpreter needs them).

### Supported versions

| Component | Supported | Checked by |
|---|---|---|
| CPython | 3.11, 3.12, 3.13, 3.14 (`requires-python >= 3.11`; Python 3.10 is no longer supported) | CI runs the full suite on 3.12 for every PR; the release workflow smoke-tests the wheel on 3.11-3.14 on each platform |
| Wheels | one `cp311-abi3` wheel per platform: Linux x86-64 (manylinux 2_17), macOS x86-64 and arm64, Windows x86-64 | release workflow (manual dispatch or tag) |
| Rust, building from source | **1.78 or newer** (the MSRV) | CI builds with the MSRV toolchain on every PR; consumers of the wheel need no Rust |
| Rust crate `cffdrs-fire` | 1.78 or newer, no runtime dependencies | CI |
| PyO3 / rust-numpy | 0.27 | `Cargo.lock` |
| NumPy / SciPy | tested with NumPy 2.4-2.5 and SciPy 1.17-1.18 (see `uv.lock`); older NumPy 1.24+ is allowed by the metadata but not tested here | CI uses `uv.lock` |
| Free-threaded Python (3.14t) | not supported | n/a |

The abi3 wheel keeps working on later CPython versions, but that does not certify
future NumPy/SciPy releases or untested dependency combinations.

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

- `cffdrs-fire`: dependency-free Rust FBP equations and the grid pass.
- `cffdrs-py`: thin PyO3 bindings that install as `cffdrs._rust` in the same
  `cffdrs` wheel as the Python package.

The Python implementation is the behavioral reference. The Python distribution
(`cffdrs-py`) is **0.3.0**; the independent `cffdrs-fire` core is **0.4.0**.
`cffdrs._rust.FbpGrid` is a low-level stateful grid in a private extension module;
normal Python callers use `FBP`, which prepares inputs and exports reference-compatible
results. Keeping `rust/` beside `src/` separates the Cargo workspace from Python imports.

Wheels use `cp311-abi3` (one per platform). The release workflow tests Python
3.11–3.14. Future CPython ABI compatibility does not certify future NumPy/SciPy
support or untested dependency combinations. Free-threaded Python is unsupported.

## Fire Behavior Prediction (`cffdrs.cffbps`)

The `FBP` class is a facade: initialize it with fuel, terrain, and weather inputs,
run the model, and request any of the 54 output variables. The underlying equations
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

Inside `python.equations`, the module `fmc` is now `foliar_moisture`: `calc_fmc` no longer computes everything
(it gives the foliar moisture content from `nd`); `calc_foliar_moisture()` chains `calc_latn`, `calc_d0`,
`calc_dj`, `calc_nd`, `calc_fmc` and `calc_fme`, and returns a `FoliarMoistureResult` (was `FMCResult`).

Inside `python.equations`, `surface.py` is gone: `calc_sfc` and `calc_cfc` live in `consumption.py` (all fuel
consumption), and `calc_fire_type`, `calc_hfi` and `calc_fire_intensity_class` live in the new `behavior.py`
(`crown.py` keeps only the crown determination). The Rust crate has the same modules (`consumption.rs`,
`behavior.rs`).

The historical `calcC6hros` stage no longer exists. A custom stage sequence replaces it with `calcC6BlendCFB`,
`calcC6BlendCFC`, `calcC6CROS` and `calcC6HROS` (in that order), computes `calcPercentileCFB` before
`calcRosPercentileGrowth`, calls `calcCFB` again after growth, and runs `calcAccelParam` after the final CFB (see
[Upgrading PyroCell](https://github.com/gagreene/cffdrs/blob/master/docs/CODEBASE.md#upgrading-pyrocell)). Replacing only the old method name is not enough.

**C6 acceleration changes.** Acceleration is now computed from the final (post-blend) CFB, so for C-6 the head ROS is
unchanged but `cfb`, `accel` and `hfi` change. For C-6 at ws 20, flat ground, 49.48 N, 115.69 W, 100 m,
date 2023-07-15, ffmc 90, bui 80 (float32 1x1 grid): `cfb` 0.1286 -> 0.3676, `accel` 0.0751 -> 0.0336,
`hfi` 5752 -> 6843 (the earlier values are those of the previous stage order, reported from a PyroCell run; `hros` 8.448
is unchanged). Reference rasters or goldens of a model that accelerates C-6 fires need to be regenerated, after checking
the new values against `alpha = 0.115 - 18.8 * CFB^2.5 * exp(-8 * CFB)`.

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
Rust uses threads for parallel work. See [the lifecycle and stage reference](https://github.com/gagreene/cffdrs/blob/master/docs/CODEBASE.md)
for memory retention and supported update routes.

Every difference between the two backends that is known today is tracked in
[docs/BACKEND_DIFFERENCES.md](https://github.com/gagreene/cffdrs/blob/master/docs/BACKEND_DIFFERENCES.md); `tools/compare_backends.py`
re-checks them.

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

### Running stage by stage (and growth percentiles)

Both backends expose the 23 pipeline steps (`invertWindAspect`, `calcSF`, ...,
`calcFireIntensityClass`) in `runFBP()` order, plus `setParams` to inject a value
between steps and `getParams` to read results. `reset()` discards computed and injected
values and returns to the initialised inputs; `runFBP()` always starts from there.
On Rust, intermediate arrays that no later step needs are freed, so request what you
want to read afterwards in `out_request`:

```python
fbp = FBP()
fbp.initialize(fuel_type=2, wx_date=20240516, lat=62.2, long=-133.8, elevation=1180,
               slope=8, aspect=60, ws=24, wd=266, ffmc=92, bui=31,
               out_request=['fmc', 'hros'])
fbp.invertWindAspect(); fbp.calcSF(); fbp.calcISZ(); fbp.calcFMC()
fmc = fbp.getParams(['fmc'])[0]          # inspect, or override with fbp.setParams({'fmc': ...})
fbp.calcISI_RSI_BE(); fbp.calcROS()
hros = fbp.getParams(['hros'])[0]
```

`initialize(..., d0=..., dj=..., fmc=...)` (or `set_override`, `setParams`, `calcFMC(...)`) override the
calculated foliar moisture quantities: a value you give is used by every later `runFBP` until you replace
it, clear it with `None`, or call `initialize` again; anything you do not give is calculated. A given `fmc`
also gives the matching foliar moisture effect (`fme`), and NaN or masked cells are missing values.

In an override **array**, a negative value (please use -1) asks for that cell to be calculated while the other cells
take your values: `fmc=[95, -1, 110]` gives the cell 0 and 2 values and calculates cell 1 when `calcFMC` (or `runFBP`)
runs. A negative scalar is the same as `None`. Until `calcFMC` runs, calculated cells of `fmc` and `fme` are NaN (nothing
is calculated for you when you change a value: run the stages yourself). Other inputs are not given a calculate value.

**Missing data must be masked.** The package only knows masked cells (`np.ma` arrays) and NaN as missing; a nodata value
such as -999 is an ordinary number. Mask it before `initialize`, for example `np.ma.masked_equal(arr, -999)` or
rasterio's `src.read(1, masked=True)`. Negative `ws`, `ffmc`, `bui`, `pc`, `pdf` and `gfl` are set to 0, so an unmasked
-999 gives a plausible-looking wrong result.

**The fuel grid mask applies to every dataset.** A cell whose fuel is masked or invalid (not 1-20) is missing in every
input and override, so its outputs are as for an invalid fuel with all other inputs missing (floats NaN, `fire_type` 0,
`fi_class` -99, `dj` 0, and `fuel_type` 0). Non-fuel (19) and water (20) are valid codes and are not masked.

`initialize(..., percentile_growth=p)` takes a percentile (0-100) of the rate-of-spread
distribution of the Han & Braun (2014) model, not a percent change: 50 (the default)
is the unadjusted ROS, and for example 95 scales surface-fire ROS by about 4.6x.

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
field's dependencies, not a blanket mask of every output, with one exception: fuel
consumption (`ffc`, `wfc`, `sfc`, `cfc`, `tfc`) follows the spread, so it is 0 where the head
rate of spread is 0 (for example D-2 with BUI below 70) and NaN where it is NaN.

All-scalar inputs export Python scalars. Legacy masked/non-modeled cases can
export floating zero even for nominal integer fields; missing explicit dates
export NaN. Rust expands every result to the full grid shape; Python can keep
scalar-derived results at `(1,)` and has legacy 0-d export differences.
See [the detailed output contract](https://github.com/gagreene/cffdrs/blob/master/docs/CODEBASE.md#outputs-and-intentional-differences)
for these measured exceptions.

### Experimental GPU backend

An experimental CuPy (GPU) variant of the FBP model lives at
`cffdrs.cffbps.cupy`. Install the optional `gpu` extra (needs a
CUDA-capable environment):

```bash
pip install "cffdrs-fire[gpu]"
# On CUDA, prefer the version-matched wheel instead, e.g.:
#   pip install cupy-cuda12x
#   pip install cupy-cuda13x
# If CuPy fails with "Failed to find CUDA headers", add the CUDA component extra:
#   pip install "cupy-cuda13x[ctk]"
```

The CuPy backend was last run on one NVIDIA RTX 5090 (CUDA 13.3, Python 3.14, `cupy-cuda13x[ctk]`
14.2); no other GPU, CUDA version or platform has been tried.

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
cross-implementation rounding). The test suite locks all 54 outputs to golden
snapshots on both the scalar and array code paths, compared at float32 precision so
that last-bit libm/BLAS differences between platforms do not fail the tests.
The golden, missing-input and unit suites also run against the Rust backend, and a
bit-level oracle pins the scalar Rust core.

## Development

Development requires Python 3.11+ (3.11-3.14), `uv`, and Rust 1.78 or newer
(`rustup` stable works; CI also builds with 1.78). The main
project uses Maturin, so syncing the environment builds `cffdrs._rust` alongside
the editable Python package:

```bash
git clone https://github.com/gagreene/cffdrs.git
cd cffdrs
uv sync --extra test --extra dev   # editable mixed Python/Rust install + tooling
uv run --no-sync python -m pytest tests -q -p no:cacheprovider  # both backends
uv run --no-sync ruff check src/ tests/cffbps  # lint
uv run --no-sync mypy               # type-check the equation core
cargo test --manifest-path rust/Cargo.toml -p cffdrs-fire
uv run --no-sync python tools/compare_backends.py   # Python-vs-Rust differences
```

After a Rust-only change, rebuild the extension with
`uv sync --reinstall-package cffdrs --extra test --extra dev`.

The Python distribution version comes from `rust/crates/cffdrs-py/Cargo.toml`;
update that crate version when preparing a release tag. The pure Rust core has
its own version in `rust/crates/cffdrs-fire/Cargo.toml` because it may be consumed
independently by Rust applications.

Existing golden fixtures are the behavior-preservation oracle. Do not regenerate
them to accommodate a refactor. Releases are published by the
`release` workflow (a `v*` tag publishes to PyPI); version tags and publication require owner authorization.

### Contributing

Read [`docs/CONTRIBUTION_PROCEDURES.md`](https://github.com/gagreene/cffdrs/blob/master/docs/CONTRIBUTION_PROCEDURES.md) before opening a pull request: it
covers branch routing (major and minor changes go to `pre_release`, patches to `master`), commit rules, the
local checks and large tests that are recorded in every pull request, and the rules for numerical changes.
Tests are run locally and never automatically on a schedule.

## License

MIT — see [LICENSE](LICENSE). © 2024 Gregory A. Greene.
