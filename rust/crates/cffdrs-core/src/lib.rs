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
//! | [`fbp`] | [`fbp::FbpInput`], [`fbp::FbpResult`], [`fbp::run`] composing the stages | `facade.runFBP` |
//! | [`grid`] | [`grid::GridInput`], [`grid::GridError`], [`grid::BehaviourGrids`], [`grid::run_grid`] | grid driver |
//! | [`fuel`] | [`fuel::FuelType`] | `constants.py` |
//! | [`percentile`] | percentile growth model | `equations/growth.py` |
//! | [`quantile`] | Student-t / normal quantile used by the percentile model | scipy `t.ppf` |
//!
//! The remaining modules are private implementation stages: input
//! normalisation (`inputs.py`), foliar moisture (`equations/fmc.py`), slope and
//! wind effects (`equations/slope_wind.py`), surface fuel consumption
//! (`equations/surface.py`), rate of spread (`equations/ros.py`), crown fire
//! (`equations/crown.py`), fire growth (`equations/growth.py`) and total
//! consumption / intensity (`equations/consumption.py`).
//!
//! # Conventions
//!
//! - All floating-point values are `f64`, including code-like outputs such as
//!   `fire_type` and `fi_class`.
//! - Fuel codes: 1..=18 are the modeled fuels (C-1 to S-3), 19 is non-fuel and
//!   20 is water. Non-modeled codes give NaN behaviour, not an error.
//! - A NaN input is a missing/masked cell and propagates to the outputs that
//!   depend on it. Neither [`fbp::run`] nor [`grid::run_grid`] panics.
//! - `percentile_growth` is a percentile (0-100) of the ROS distribution, not
//!   a percent change: 50 is the unadjusted ROS. See [`percentile`].
//! - Units: wind in km/h, angles in compass degrees, ROS in m/min, fuel
//!   consumption in kg/m^2, intensity in kW/m, `wx_date` as `YYYYMMDD`.
//!
//! # Changes in 0.2.0
//!
//! `fbp::run_grid` and `fbp::BehaviourGrids` are replaced by
//! [`grid::run_grid`], [`grid::GridInput`], [`grid::GridError`] and
//! [`grid::BehaviourGrids`]; `run_grid` now returns a `Result` instead of
//! panicking on bad input.
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
//!   CFFDRS_BASELINE_RECORD=1 cargo test --manifest-path rust/Cargo.toml -p cffdrs-core --test characterization -- --ignored
//!   cargo test --manifest-path rust/Cargo.toml -p cffdrs-core --test characterization -- --ignored
//!   ```
//!
//!   The baseline lives in `rust/target/` and is never committed.
//! - **Benchmarks**: `examples/bench_run.rs` times [`fbp::run`] in the release
//!   profile (`cargo run --release --manifest-path rust/Cargo.toml -p cffdrs-core --example bench_run`);
//!   `tools/bench_rust_grid.py` times the compiled grid pass from Python,
//!   including array copies (`uv run --no-sync python tools/bench_rust_grid.py`).

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]

mod consumption;
mod crown;
pub mod fbp;
mod fmc;
pub mod fuel;
pub mod grid;
mod growth;
mod normalize;
pub mod percentile;
pub mod quantile;
mod ros;
mod slope_wind;
mod surface;
