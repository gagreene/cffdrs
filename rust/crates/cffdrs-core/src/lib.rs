//! Pure-Rust CFFDRS core.
//!
//! The Python package in this repository (`src/cffdrs/`) is the reference
//! implementation; this crate mirrors it for consumers that need the FBP
//! equations without a Python runtime in the hot path. Validated against the
//! same golden fixtures as the Python test suite
//! (`tests/cffbps/data/golden/`).
//!
//! # Changes in 0.2.0
//!
//! `fbp::run_grid` and `fbp::BehaviourGrids` are replaced by
//! [`grid::run_grid`], [`grid::GridInput`], [`grid::GridError`] and
//! [`grid::BehaviourGrids`]; `run_grid` now returns a `Result` instead of
//! panicking on bad input.

#![forbid(unsafe_code)]

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
