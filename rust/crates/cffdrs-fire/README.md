# cffdrs-fire

Pure-Rust implementation of the **Fire Behaviour Prediction (FBP) system** of the Canadian Forest Fire Danger
Rating System (CFFDRS): a scalar core for one cell, a grid pass, and the 23 pipeline stages that can be run one at a
time over a whole grid. It has no dependencies and forbids `unsafe` code. The minimum supported Rust version is 1.78.

It mirrors the Python package [`cffdrs-fire`](https://pypi.org/project/cffdrs-fire/) function by function (that
package is the reference and bundles this crate as its optional Rust backend) and is validated against the same
golden fixtures. Floating-point results agree with the Python reference to a relative tolerance of 1e-9, and integer
results agree exactly. Only the FBP system is implemented here: the Fire Weather Index (FWI) system is Python-only.

```toml
[dependencies]
cffdrs-fire = "0.4"
```

## One cell

```rust
use cffdrs_fire::fbp::{run, FbpInput};

let mut input = FbpInput::new(2, 20230615); // fuel C-2, weather date YYYYMMDD
input.lat = 55.0;
input.long = -115.0;
input.elevation = 500.0;
input.slope_pct = 0.0;
input.aspect_deg = 0.0;
input.ws = 20.0; // km/h
input.wd = 270.0; // compass degrees
input.ffmc = 90.0;
input.bui = 60.0;

let result = run(&input);
assert!(result.hros > 0.0);
println!("head ROS {:.2} m/min, HFI {:.0} kW/m", result.hros, result.hfi);
```

`FbpResult` holds all 54 outputs, named like the Python package's `getParams` names (`fF` and `fW` are spelled `f_f`
and `f_w`); `FbpResult::named_values` gives the name table.

## A grid of cells

`grid::run_grid_fields` computes any set of fields for many cells at once. Each input is a `PerCell`: one constant for
every cell (`52.0.into()`) or one value per cell (`(&[10.0, 0.0]).into()`).

```rust
use cffdrs_fire::field::Field;
use cffdrs_fire::grid::{run_grid_fields, GridInput, NonModeled};

let input = GridInput {
    fuel_type: &[2, 19], // C-2 and a non-fuel cell
    lat: 52.0.into(),
    long: (-115.0).into(),
    elevation: 800.0.into(),
    slope_pct: (&[10.0, 0.0]).into(),
    aspect_deg: 180.0.into(),
    pct_conifer: 0.0.into(),
    pct_dead_fir: 0.0.into(),
    grass_fuel_load: 0.0.into(),
    grass_curing: 0.0.into(),
    ws: 20.0.into(),
    wd: 270.0.into(),
    ffmc: 90.0.into(),
    bui: 80.0.into(),
    wx_date: 20_230_615,
    percentile_growth: 50.0,
    d0: None,
    dj: None,
    fmc: None,
};
let out = run_grid_fields(&input, &[Field::Hros, Field::Hfi, Field::FireType], 1, NonModeled::PythonCompat)
    .unwrap();
assert!(out.get(Field::Hros).unwrap()[0] > 0.0);
assert!(out.get(Field::Hros).unwrap()[1].is_nan()); // the non-fuel cell has no behaviour
```

- **Outputs:** pick them with the `field::Field` enum (54 outputs, 13 input-only quantities, 9 internal ones);
  `Field::name` is the Python attribute name. `grid::run_grid` returns the usual set of behaviour grids at once.
- **Overrides:** `d0`, `dj` and `fmc` (a `PerCell`, or `d0_override`, `dj_override`, `fmc_override` and
  `hros_override` on a single `FbpInput`) replace the calculated value. In an array a negative cell means "calculate
  this cell", and `NaN` is a missing value. `fmc` replaces only the foliar moisture content; `fme` follows it.
- **Threads:** the `threads` argument is `1` for the calling thread (the default way to use it), `0` for every core, or
  a number of workers. The result does not depend on it, and the library never starts threads unless asked.
- **Stage by stage:** `fbp_grid::FbpGrid` runs the 23 pipeline stages one at a time over a grid, with injection of
  values between stages, and frees arrays once no later stage needs them.
- **Units:** wind speed in km/h; wind direction, aspect and latitude/longitude in degrees (compass degrees for wind);
  slope in percent; rate of spread in m/min; fuel consumption in kg/m^2; intensity in kW/m; `wx_date` as `YYYYMMDD`.
- **Missing values and fuel codes:** a `NaN` input is a missing cell and propagates to the outputs that depend on it;
  nothing panics. Fuel codes 1 to 18 are the modeled fuels (C-1 to S-3), 19 is non-fuel and 20 is water; other codes
  give `NaN` behaviour, or the Python package's values with `NonModeled::PythonCompat`.

## Documentation

The crate documentation on [docs.rs](https://docs.rs/cffdrs-fire) has the module map, the conventions (including
`percentile_growth` as a percentile of the ROS distribution, not a percent change) and the API stability policy. The
equations, the stage order and the known differences from the reference are described in the
[Python package README](https://pypi.org/project/cffdrs-fire/) and in
[`docs/CODEBASE.md`](https://github.com/gagreene/cffdrs/blob/master/docs/CODEBASE.md).

Repository: <https://github.com/gagreene/cffdrs>. Licensed under the MIT license.
