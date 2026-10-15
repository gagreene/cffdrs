//! Pure-Rust CFFDRS core.
//!
//! This crate computes the Canadian Forest Fire Behaviour Prediction System
//! (FBP) outputs for one cell ([`fbp::run`]) or a grid of cells
//! ([`grid::run_grid`]), without a Python runtime in the hot path. The Python
//! package in this repository (`src/cffdrs/cffbps/`) is the reference
//! implementation; this crate mirrors it function by function and is validated
//! against the same golden fixtures as the Python test suite
//! (`tests/cffbps/data/golden/`). It has no dependencies.
//!
//! # Module map
//!
//! | Module | Responsibility | Python counterpart |
//! |---|---|---|
//! | [`fbp`] | [`fbp::FbpInput`], [`fbp::FbpResult`], [`fbp::run`] composing the stages | `FBP.runFBP` (`python/core.py`) |
//! | [`field`] | [`field::Field`]: the 75 per-cell quantities (54 outputs, 12 input-only, 9 internal) | `FBP` attributes |
//! | [`stage`] | [`stage::Stage`]: the 23 pipeline stages, their order and the fields each reads and writes | `FBP.calcX` methods |
//! | [`state`] | [`state::CellState`], [`state::StageParams`]: one cell's fields and the stage bodies | an initialised `FBP` |
//! | [`grid`] | [`grid::GridInput`] of [`grid::PerCell`] inputs, [`grid::run_grid_fields`] (any requested [`field::Field`]s, [`grid::NonModeled`] choice), [`grid::GridError`], [`grid::BehaviourGrids`], [`grid::run_grid`] | `cffdrs._rust.run_fbp_grid` (compiled grid pass); `fbpMultiprocessArray` in `python/parallel.py` |
//! | [`fbp_grid`] | [`fbp_grid::FbpGrid`]: the stages (or the fused pass) over a whole grid, storing only the arrays still needed; [`fbp_grid::StageError`], [`fbp_grid::ParamName`] | a Python `FBP` object stepped with `calcX`, `setParams`, `getParams`, `runFBP` and `reset` |
//! | `liveness` (private) | which computed fields the planned stages still read | |
//! | [`fuel`] | [`fuel::FuelType`] | `common/constants.py` |
//! | [`percentile`] | percentile growth model | `python/equations/growth.py` |
//! | `quantile` (private) | internal Student-t / normal quantile used by the percentile model | scipy `t.ppf` |
//!
//! The remaining modules are private implementation stages: input
//! normalisation (`common/inputs.py`), foliar moisture (`python/equations/foliar_moisture.py`), slope and
//! wind effects (`python/equations/slope_wind.py`), surface fuel consumption
//! (`python/equations/surface.py`), rate of spread (`python/equations/ros.py`), crown fire
//! (`python/equations/crown.py`), fire growth (`python/equations/growth.py`) and total
//! consumption / intensity (`python/equations/consumption.py`).
//!
//! # Conventions
//!
//! - All floating-point values are `f64`, including code-like outputs such as
//!   `fire_type` and `fi_class`.
//! - Fuel codes: 1..=18 are the modeled fuels (C-1 to S-3), 19 is non-fuel and
//!   20 is water. Non-modeled codes give NaN behaviour, not an error
//!   ([`grid::NonModeled::PythonCompat`] gives the Python package's values
//!   instead).
//! - A NaN input is a missing/masked cell and propagates to the outputs that
//!   depend on it. Neither [`fbp::run`] nor [`grid::run_grid`] panics.
//! - `percentile_growth` is a percentile (0-100) of the ROS distribution, not
//!   a percent change: 50 is the unadjusted ROS. See [`percentile`].
//! - Units: wind in km/h, angles in compass degrees, ROS in m/min, fuel
//!   consumption in kg/m^2, intensity in kW/m, `wx_date` as `YYYYMMDD`.
//! - [`VERSION`] is this crate's version string; the Python binding exposes it
//!   as `cffdrs._rust.__core_version__`.
//!
//! # API stability
//!
//! The output types [`fuel::FuelType`], [`fbp::FbpResult`] and
//! [`grid::BehaviourGrids`], and the enums [`field::Field`],
//! [`stage::Stage`], [`grid::GridError`], [`fbp_grid::StageError`] and
//! [`fbp_grid::ParamName`], are `#[non_exhaustive]`, so variants or fields can
//! be added without a breaking release: match `FuelType` with a wildcard arm
//! and read result fields rather than destructuring them exhaustively
//! (`FbpResult::default()` still works). The input types [`fbp::FbpInput`] and
//! [`grid::GridInput`] stay constructible by struct literal on purpose;
//! [`fbp::FbpInput::new`] is the forward-compatible constructor.
//!
//! # Changes in 0.3.0
//!
//! Every [`grid::GridInput`] input except `fuel_type` is a [`grid::PerCell`]
//! (one constant or one value per cell; `ffmc`, `bui`, `pct_dead_fir` and
//! `grass_fuel_load` were scalars, the rest slices), and the input gains
//! explicit `d0` / `dj` overrides. [`grid::run_grid_fields`] returns any
//! requested fields. [`grid::GridError::LengthMismatch`] fires only for a
//! [`grid::PerCell::Cells`] input of the wrong length;
//! [`grid::GridError::InfiniteDj`] and [`grid::GridError::NotAnOverride`] are
//! new. [`fbp_grid::FbpGrid`] is new: the pipeline stages over whole grids.
//!
//! # Historical changes in core 0.2.0
//!
//! `fbp::run_grid` and `fbp::BehaviourGrids` are replaced by
//! [`grid::run_grid`], [`grid::GridInput`], [`grid::GridError`] and
//! [`grid::BehaviourGrids`]; `run_grid` now returns a `Result` instead of
//! panicking on bad input.
//!
//! Current versions are core 0.3.0 and `cffdrs-py` 0.2.0 (the Python
//! distribution). The historical core 0.2.0 release had an independent binding version.
//!
//! # Developing
//!
//! Two local tools support refactoring without changing numbers. Neither runs
//! in CI.
//!
//! - **Bit-level oracle**: `tests/characterization.rs` runs [`fbp::run`] over
//!   7,200 deterministic inputs and compares every output bit for bit (NaN
//!   canonicalised). It is `#[ignore]`d. Record a baseline on the same machine
//!   and toolchain, then compare after each change:
//!
//!   ```text
//!   CFFDRS_BASELINE_RECORD=1 cargo test --manifest-path rust/Cargo.toml -p cffbps-rust --test characterization -- --ignored
//!   cargo test --manifest-path rust/Cargo.toml -p cffbps-rust --test characterization -- --ignored
//!   ```
//!
//!   The baseline lives in `rust/target/` and is never committed.
//! - **Benchmarks**: `examples/bench_run.rs` times [`fbp::run`] in the release
//!   profile (`cargo run --release --manifest-path rust/Cargo.toml -p cffbps-rust --example bench_run`);
//!   `tools/bench_rust_grid.py` times the compiled grid pass from Python,
//!   including array copies (`uv run --no-sync python tools/bench_rust_grid.py`).

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]

/// This crate's version (`CARGO_PKG_VERSION` of `cffbps-rust`), for example
/// to report which core a binding or engine was built against.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// Lets unit tests share the integration tests' input generator
// (`tests/common/cases.rs`), which names this crate `cffbps_rust`.
#[cfg(test)]
extern crate self as cffbps_rust;

mod consumption;
mod crown;
pub mod fbp;
pub mod fbp_grid;
pub mod field;
mod foliar_moisture;
pub mod fuel;
pub mod grid;
mod growth;
mod liveness;
mod normalize;
pub mod percentile;
mod quantile;
mod ros;
mod slope_wind;
pub mod stage;
pub mod state;
mod surface;

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_the_core_crate_version() {
        assert_ne!(super::VERSION, "");
        assert!(super::VERSION.starts_with("0."), "{}", super::VERSION);
    }
}
