//! Python bindings over cffbps-rust. Kept intentionally thin: the science
//! lives in cffbps-rust; the cffdrs Python package remains the reference
//! implementation and the spec.

mod fbp_grid;

use cffbps_rust::grid::{run_grid_with_threads, GridInput, PerCell};
use numpy::{PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Default percent dead balsam fir for a NaN scalar, as in the Python package.
const PCT_DEAD_FIR_DEFAULT: f64 = 35.0;
/// Default grass fuel load (kg/m^2) for a NaN scalar, as in the Python package.
const GRASS_FUEL_LOAD_DEFAULT: f64 = 0.35;

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
///
/// Threads: `threads=1` (the default) computes on the calling thread, so a
/// caller that runs many grids at once (for example one fire per worker) is not
/// surprised by extra threads. `threads=0` uses every available core and any
/// other positive value uses that many; results are identical for every value.
/// Grids under 4,096 cells always run serially.
///
/// Scalar defaults: a NaN `pct_dead_fir` is replaced by 35.0 and a NaN
/// `grass_fuel_load` by 0.35 before the grid is built, exactly as the Python
/// package's `initialize` treats a NaN scalar `pdf` / `gfl`. The per-cell
/// arrays `pct_conifer` and `grass_curing` get no default: a NaN cell there is
/// a missing input and propagates as NaN.
// `PyReadonlyArray2` is pyo3's by-value extraction type for numpy arguments; it
// is a cheap borrow guard, not a payload, so taking it by value is intentional.
#[pyfunction]
#[pyo3(signature = (
    fuel_type, lat, long, elevation, slope_pct, aspect_deg, pct_conifer,
    grass_curing, ws, wd, wx_date, ffmc, bui, pct_dead_fir, grass_fuel_load,
    percentile_growth, threads = 1
))]
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
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
    threads: i64,
) -> PyResult<Bound<'py, PyDict>> {
    let threads = usize::try_from(threads).map_err(|_| {
        PyValueError::new_err(format!(
            "threads must be 0 (all cores) or a positive count, got {threads}"
        ))
    })?;
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

    // Python's `_coerce` defaults for a NaN scalar pdf / gfl (inputs.py).
    let pct_dead_fir = if pct_dead_fir.is_nan() {
        PCT_DEAD_FIR_DEFAULT
    } else {
        pct_dead_fir
    };
    let grass_fuel_load = if grass_fuel_load.is_nan() {
        GRASS_FUEL_LOAD_DEFAULT
    } else {
        grass_fuel_load
    };

    // Arrays are per cell, scalars constant for every cell.
    let input = GridInput {
        fuel_type: &fuel,
        lat: PerCell::Cells(&lat_s),
        long: PerCell::Cells(&long_s),
        elevation: PerCell::Cells(&elev_s),
        slope_pct: PerCell::Cells(&slope_s),
        aspect_deg: PerCell::Cells(&aspect_s),
        pct_conifer: PerCell::Cells(&pc_s),
        pct_dead_fir: PerCell::Const(pct_dead_fir),
        grass_fuel_load: PerCell::Const(grass_fuel_load),
        grass_curing: PerCell::Cells(&gc_s),
        ws: PerCell::Cells(&ws_s),
        wd: PerCell::Cells(&wd_s),
        ffmc: PerCell::Const(ffmc),
        bui: PerCell::Const(bui),
        wx_date,
        percentile_growth,
        d0: None,
        dj: None,
    };
    let grids = py
        .detach(|| run_grid_with_threads(&input, threads))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let out = PyDict::new(py);
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
            numpy::PyArray1::from_vec(py, v).reshape([nrows, ncols])?,
        )?;
    }
    Ok(out)
}

/// The compiled backend exposed as `cffdrs._rust` by the mixed Python/Rust wheel.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__core_version__", cffbps_rust::VERSION)?;
    m.add_function(wrap_pyfunction!(run_fbp_grid, m)?)?;
    m.add_class::<fbp_grid::FbpGrid>()?;
    Ok(())
}
