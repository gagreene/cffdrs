//! Python bindings over cffdrs-core. Kept intentionally thin: the science
//! lives in cffdrs-core; the cffdrs Python package remains the reference
//! implementation and the spec.

// pyo3 0.22's `#[pyfunction]` expansion converts `PyErr` into `PyErr`, which clippy
// reports as a useless conversion at the signature of every pyfunction. The lint
// fires in macro-generated code, so an attribute on the function does not reach
// it; drop this allow when pyo3 is upgraded past the affected releases.
#![allow(clippy::useless_conversion)]
// pyo3 0.22 macro expansion also trips `unsafe_op_in_unsafe_fn` (E0133) on newer
// rustc once the workspace lint is enabled; same removal condition as above.
#![allow(unsafe_op_in_unsafe_fn)]

use cffdrs_core::grid::{run_grid, GridInput};
use numpy::{PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Copy a 2-D array into an owned `Vec` in logical row-major order, whatever
/// its memory layout (C, Fortran or strided).
fn row_major<T: numpy::Element + Copy>(array: &PyReadonlyArray2<'_, T>) -> Vec<T> {
    array.as_array().iter().copied().collect()
}

/// The CFFBPS grid pass for one weather step: per-cell fuel/terrain/wind
/// grids plus scalar ffmc/bui/date. Returns a dict of 2-D float64 arrays
/// (`hros, bros, raz, wsv, hfi, rso, sros, sfc, fmc, accel`); cells with
/// fuel codes outside 1..18 are NaN.
///
/// `percentile_growth` is a percentile (0-100) of the ROS distribution, not a
/// percent change: 50 is the unadjusted ROS, values outside (0.001, 99.999) are
/// capped, and NaN propagates, matching the Python package.
///
/// Input contract: arrays may have any memory layout (C, Fortran or strided);
/// they are copied to row-major buffers before the GIL is released, so other
/// threads may safely mutate the caller's arrays during the run. `wx_date` must
/// be a real `YYYYMMDD` calendar date or a `ValueError` is raised. A NaN
/// latitude, longitude or elevation masks that cell's foliar moisture (and any
/// crown behaviour that depends on it), as in the Python package.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_fbp_grid<'py>(
    py: Python<'py>,
    fuel_type: PyReadonlyArray2<'py, i32>,
    lat: PyReadonlyArray2<'py, f64>,
    long: PyReadonlyArray2<'py, f64>,
    elevation: PyReadonlyArray2<'py, f64>,
    slope_pct: PyReadonlyArray2<'py, f64>,
    aspect_deg: PyReadonlyArray2<'py, f64>,
    pct_conifer: PyReadonlyArray2<'py, f64>,
    grass_curing: PyReadonlyArray2<'py, f64>,
    ws: PyReadonlyArray2<'py, f64>,
    wd: PyReadonlyArray2<'py, f64>,
    wx_date: i64,
    ffmc: f64,
    bui: f64,
    pct_dead_fir: f64,
    grass_fuel_load: f64,
    percentile_growth: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let (nrows, ncols) = (fuel_type.shape()[0], fuel_type.shape()[1]);
    let expect = |name: &str, s: &[usize]| -> PyResult<()> {
        if s != [nrows, ncols] {
            return Err(PyValueError::new_err(format!(
                "{name} shape {s:?} != fuel_type shape ({nrows}, {ncols})"
            )));
        }
        Ok(())
    };
    for (name, arr) in [
        ("lat", &lat),
        ("long", &long),
        ("elevation", &elevation),
        ("slope_pct", &slope_pct),
        ("aspect_deg", &aspect_deg),
        ("pct_conifer", &pct_conifer),
        ("grass_curing", &grass_curing),
        ("ws", &ws),
        ("wd", &wd),
    ] {
        expect(name, arr.shape())?;
    }

    // Snapshot every input into an owned, row-major buffer before the GIL is
    // released. Reading `as_slice()` would hand Rust the array's physical memory
    // order (wrong for Fortran-ordered inputs, an error for strided views), and
    // would leave caller-owned memory borrowed while other Python threads can
    // still mutate it; rust-numpy's borrow tracking does not cover that.
    let fuel = row_major(&fuel_type);
    let (lat_s, long_s) = (row_major(&lat), row_major(&long));
    let (elev_s, slope_s, aspect_s) = (
        row_major(&elevation),
        row_major(&slope_pct),
        row_major(&aspect_deg),
    );
    let (pc_s, gc_s) = (row_major(&pct_conifer), row_major(&grass_curing));
    let (ws_s, wd_s) = (row_major(&ws), row_major(&wd));

    let input = GridInput {
        fuel_type: &fuel,
        lat: &lat_s,
        long: &long_s,
        elevation: &elev_s,
        slope_pct: &slope_s,
        aspect_deg: &aspect_s,
        pct_conifer: &pc_s,
        grass_curing: &gc_s,
        ws: &ws_s,
        wd: &wd_s,
        wx_date,
        ffmc,
        bui,
        pct_dead_fir,
        grass_fuel_load,
        percentile_growth,
    };
    let grids = py
        .allow_threads(|| run_grid(&input))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let out = PyDict::new_bound(py);
    for (name, v) in [
        ("hros", grids.hros),
        ("bros", grids.bros),
        ("raz", grids.raz),
        ("wsv", grids.wsv),
        ("hfi", grids.hfi),
        ("rso", grids.rso),
        ("sros", grids.sros),
        ("sfc", grids.sfc),
        ("fmc", grids.fmc),
        ("accel", grids.accel),
    ] {
        out.set_item(
            name,
            numpy::PyArray1::from_vec_bound(py, v).reshape([nrows, ncols])?,
        )?;
    }
    Ok(out)
}

/// The compiled backend exposed as `cffdrs._rust` by the mixed Python/Rust wheel.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__core_version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(run_fbp_grid, m)?)?;
    Ok(())
}
