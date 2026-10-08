//! Pure-Rust CFFDRS core.
//!
//! The Python package in this repository (`src/cffdrs/`) is the reference
//! implementation; this crate mirrors it for consumers that need the FBP
//! equations without a Python runtime in the hot path. Validated against the
//! same golden fixtures as the Python test suite
//! (`tests/cffbps/data/golden/`).

#![forbid(unsafe_code)]

pub mod fbp;
pub mod percentile;
pub mod quantile;
