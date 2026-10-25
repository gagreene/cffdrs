# cffdrs-fire

Pure-Rust implementation of the **Fire Behaviour Prediction (FBP) system** of the Canadian Forest Fire Danger
Rating System (CFFDRS): a scalar core for one cell, a grid pass, and the 23 pipeline stages that can be run one at a
time over a whole grid. It has no dependencies and forbids `unsafe` code. The minimum supported Rust version is 1.78.

It mirrors the Python package [`cffdrs-fire`](https://pypi.org/project/cffdrs-fire/) function by function (that
package is the reference and bundles this crate as its optional Rust backend) and is validated against the same
golden fixtures. Floating-point results agree with the Python reference to a relative tolerance of 1e-9, and integer
results agree exactly. Only the FBP system is implemented here: the Fire Weather Index (FWI) system is Python-only.

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

For a grid, see `grid::run_grid_fields` (any requested set of fields) and `fbp_grid::FbpGrid` (the stages over a whole
grid, freeing arrays once no later stage needs them). A `NaN` input is a missing cell and propagates to the outputs
that depend on it; nothing panics. Fuel codes 1 to 18 are the modeled fuels (C-1 to S-3), 19 is non-fuel and 20 is
water. The crate documentation on [docs.rs](https://docs.rs/cffdrs-fire) has the module map, the conventions
(units, missing values, `percentile_growth` as a percentile of the ROS distribution) and the API stability policy.

Repository: <https://github.com/gagreene/cffdrs>. Licensed under the MIT license.
