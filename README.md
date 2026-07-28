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

Optional extras:

```bash
pip install "cffdrs[io]"    # rasterio + pandas, for raster/tabular workflows
```

Requires Python 3.10+. Core dependencies: `numpy`, `scipy`, `psutil`.

## Package layout

| Module | Contents |
|---|---|
| `cffdrs.cffbps` | FBP System: the `FBP` class (facade) over pure equation modules |
| `cffdrs.cffwis` | FWI System: FFMC/DMC/DC/ISI/BUI/FWI/DSR functions (hourly + daily) |
| `cffdrs.diurnal_ffmc_lawson` | Diurnal (hourly) FFMC interpolation per Lawson et al. |

## Fire Behavior Prediction (`cffdrs.cffbps`)

The `FBP` class is a facade: initialize it with fuel, terrain, and weather inputs,
run the model, and request any of 50+ output variables. The underlying equations
live in `cffdrs.cffbps.equations` as pure, typed functions if you need them directly.

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
)
fire_type, hros, hfi = fbp.runFBP()
```

### Array / raster example

Any spatial input may be a NumPy array (all arrays must share one shape); outputs
come back as arrays with NaN at invalid/NoData cells:

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

For very large rasters, `cffdrs.cffbps.fbpMultiprocessArray(...)` splits the inputs
into blocks and runs them across a worker pool with the same semantics.

### Available outputs

`out_request` accepts any of the names in `cffdrs.cffbps.valid_outputs` — including
final outputs (`hros`, `hfi`, `fire_type`, `cfb`, `tfc`, `fi_class`, …) and
intermediates (`isi`, `wsv`, `raz`, `sfc`, `fmc`, `csfi`, `rso`, `be`, …).

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

```bash
git clone https://github.com/gagreene/cffdrs.git
cd cffdrs
uv sync --extra test --extra dev   # editable install + tooling
uv run pytest                      # full suite
uv run ruff check src/             # lint
uv run mypy                        # type-check the equation core
```

Golden regression fixtures are regenerated with `uv run python tools/gen_fbp_goldens.py`
— only do this deliberately from a known-good state; the snapshots are the
behavior-preservation oracle for refactors.

## License

MIT — see [LICENSE](LICENSE). © 2024 Gregory A. Greene.
