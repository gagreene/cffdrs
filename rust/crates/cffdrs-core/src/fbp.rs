//! The CFFBPS scalar core and grid pass.
//!
//! Mirrors `src/cffdrs/cffbps/` (the reference implementation): same
//! equations, same fuel-code conventions (1..18 modeled, 19/20 non-fuel),
//! same intermediate quantities. `run` computes one cell; `run_grid` is the
//! per-window vectorization consumed by fire-growth engines.
//!
//! Every quantity in [`FbpResult`] is asserted against the shared golden
//! snapshot (`tests/cffbps/data/golden/wotton2009_scalar_snapshot.json`) —
//! the same file the Python suite validates against.

/// Scalar inputs, matching `FBP.initialize` in the Python package.
#[derive(Debug, Clone)]
pub struct FbpInput {
    /// CFFBPS numeric fuel code (1..18 modeled; >= 19 non-fuel).
    pub fuel_type: i32,
    /// Date as YYYYMMDD (drives day-of-year and foliar moisture).
    pub wx_date: i64,
    pub lat: f64,
    pub long: f64,
    pub elevation: f64,
    /// Ground slope in percent.
    pub slope_pct: f64,
    /// Aspect (downhill direction), compass degrees.
    pub aspect_deg: f64,
    /// 10-m open wind speed, km/h.
    pub ws: f64,
    /// Wind direction, compass degrees.
    pub wd: f64,
    pub ffmc: f64,
    pub bui: f64,
    /// Percent conifer (M-1/M-2).
    pub pc: f64,
    /// Percent dead fir (M-3/M-4).
    pub pdf: f64,
    /// Grass fuel load, kg/m^2 (O-1a/b).
    pub gfl: f64,
    /// Grass curing factor, percent (O-1a/b).
    pub gcf: f64,
    /// Percentile growth (50 = median behaviour).
    pub percentile_growth: f64,
}

/// Everything the scalar pass computes, mirroring the Python snapshot's 54
/// quantities plus `lb_ratio` (computed but absent from the snapshot; it is
/// covered by the differential tests in cffdrs-py instead).
///
/// All quantities are f64, including code-like values (`fire_type`,
/// `fi_class`) so golden comparison is uniform; consumers cast as needed.
#[derive(Debug, Clone, Default)]
pub struct FbpResult {
    // wind/slope vectoring
    pub ws: f64,
    pub wd: f64,
    pub wse: f64,
    pub wse1: f64,
    pub wse2: f64,
    pub wsx: f64,
    pub wsy: f64,
    pub wsv: f64,
    pub raz: f64,
    // moisture / ISI chain
    pub m: f64,
    pub f_f: f64,
    pub f_w: f64,
    pub ffmc: f64,
    pub isi: f64,
    pub bui: f64,
    // fuel-type ROS parameterization
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub q: f64,
    pub bui0: f64,
    pub be: f64,
    pub be_max: f64,
    // slope-adjusted spread chain
    pub sf: f64,
    pub rsz: f64,
    pub rsf: f64,
    pub isf: f64,
    pub rsi: f64,
    // foliar moisture
    pub latn: f64,
    pub dj: f64,
    pub d0: f64,
    pub nd: f64,
    pub fmc: f64,
    pub fme: f64,
    // consumption
    pub ffc: f64,
    pub wfc: f64,
    pub sfc: f64,
    pub cfl: f64,
    pub cfc: f64,
    pub tfc: f64,
    pub cbh: f64,
    // crowning
    pub csfi: f64,
    pub rso: f64,
    pub cfb: f64,
    pub fire_type: f64,
    // spread rates
    pub hros: f64,
    pub sros: f64,
    pub cros: f64,
    pub bfw: f64,
    pub bisi: f64,
    pub bros: f64,
    // intensity / class / growth
    pub hfi: f64,
    pub fi_class: f64,
    pub accel: f64,
    pub fuel_type: f64,
    /// Length-to-breadth ratio (not in the golden snapshot; differential-tested).
    pub lb_ratio: f64,
}

/// Run the scalar FBP chain for one cell. Mirror of the Python package's
/// `FBP.initialize(...)` + `runFBP()` for a single-point input.
pub fn run(input: &FbpInput) -> FbpResult {
    let _ = input;
    todo!("port the cffbps scalar chain (src/cffdrs/cffbps/equations)")
}

/// Per-window behaviour grids for a fire-growth engine, one weather step.
pub struct BehaviourGrids {
    pub hros: Vec<f64>,
    pub bros: Vec<f64>,
    pub raz: Vec<f64>,
    pub lb_ratio: Vec<f64>,
    pub wsv: Vec<f64>,
    pub hfi: Vec<f64>,
    pub rso: Vec<f64>,
    pub sros: Vec<f64>,
    pub sfc: Vec<f64>,
    pub fmc: Vec<f64>,
    pub accel: Vec<f64>,
}

/// The grid pass: per-cell fuel/terrain plus per-cell wind, scalar
/// ffmc/bui/date. Mirror of the Python package's array path as consumed by
/// fire-growth engines (one call per weather step).
#[allow(clippy::too_many_arguments)]
pub fn run_grid(
    fuel_type: &[i32],
    lat: &[f64],
    long: &[f64],
    elevation: &[f64],
    slope_pct: &[f64],
    aspect_deg: &[f64],
    pct_conifer: &[f64],
    grass_curing: &[f64],
    ws: &[f64],
    wd: &[f64],
    wx_date: i64,
    ffmc: f64,
    bui: f64,
    pct_dead_fir: f64,
    grass_fuel_load: f64,
    percentile_growth: f64,
) -> BehaviourGrids {
    let _ = (
        fuel_type, lat, long, elevation, slope_pct, aspect_deg, pct_conifer, grass_curing, ws, wd,
        wx_date, ffmc, bui, pct_dead_fir, grass_fuel_load, percentile_growth,
    );
    todo!("vectorize run() over the window")
}
