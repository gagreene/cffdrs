//! Python bindings over cffdrs-core. Kept intentionally thin: the science
//! lives in cffdrs-core; the cffdrs Python package remains the reference
//! implementation and the spec.

// pyo3 0.22's `#[pyfunction]` expansion converts `PyErr` into `PyErr`, which clippy
// reports as a useless conversion at the signature of every pyfunction. The lint
// fires in macro-generated code, so an attribute on the function does not reach
// it; drop this allow when pyo3 is upgraded past the affected releases.
#![allow(clippy::useless_conversion)]

use numpy::{PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The CFFBPS grid pass for one weather step: per-cell fuel/terrain/wind
/// grids plus scalar ffmc/bui/date. Returns a dict of 2-D float64 arrays
/// (`hros, bros, raz, wsv, hfi, rso, sros, sfc, fmc, accel`); cells with
/// fuel codes outside 1..18 are NaN.
///
/// `percentile_growth` is a percentile (0-100) of the ROS distribution, not a
/// percent change: 50 is the unadjusted ROS, values outside (0.001, 99.999) are
/// capped, and NaN propagates, matching the Python package.
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

    let fuel = fuel_type.as_slice()?;
    let (lat_s, long_s) = (lat.as_slice()?, long.as_slice()?);
    let (elev_s, slope_s, aspect_s) = (
        elevation.as_slice()?,
        slope_pct.as_slice()?,
        aspect_deg.as_slice()?,
    );
    let (pc_s, gc_s) = (pct_conifer.as_slice()?, grass_curing.as_slice()?);
    let (ws_s, wd_s) = (ws.as_slice()?, wd.as_slice()?);

    let grids = py.allow_threads(|| {
        cffdrs_core::fbp::run_grid(
            fuel,
            lat_s,
            long_s,
            elev_s,
            slope_s,
            aspect_s,
            pc_s,
            gc_s,
            ws_s,
            wd_s,
            wx_date,
            ffmc,
            bui,
            pct_dead_fir,
            grass_fuel_load,
            percentile_growth,
        )
    });

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
