//! The grid pass: per-cell fuel/terrain/wind plus the scalars of one weather
//! step, producing the behaviour grids a fire-growth engine consumes.

use std::fmt;

use crate::fbp::{run, FbpInput};
use crate::fmc::is_valid_wx_date;
use crate::fuel::FuelType;

/// Per-cell input grids (all the same length) plus the scalars of one weather step.
#[derive(Debug, Clone, Copy)]
pub struct GridInput<'a> {
    pub fuel_type: &'a [i32],
    pub lat: &'a [f64],
    pub long: &'a [f64],
    pub elevation: &'a [f64],
    pub slope_pct: &'a [f64],
    pub aspect_deg: &'a [f64],
    pub pct_conifer: &'a [f64],
    pub grass_curing: &'a [f64],
    pub ws: &'a [f64],
    pub wd: &'a [f64],
    pub wx_date: i64,
    pub ffmc: f64,
    pub bui: f64,
    pub pct_dead_fir: f64,
    pub grass_fuel_load: f64,
    pub percentile_growth: f64,
}

/// Why a grid run was rejected before any cell was computed.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GridError {
    /// A per-cell slice has a different length than `fuel_type`.
    LengthMismatch {
        name: &'static str,
        expected: usize,
        found: usize,
    },
    /// `wx_date` is not a real `YYYYMMDD` calendar date.
    InvalidDate(i64),
}

impl fmt::Display for GridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GridError::LengthMismatch {
                name,
                expected,
                found,
            } => write!(f, "{name} has {found} values but fuel_type has {expected}"),
            GridError::InvalidDate(d) => {
                write!(f, "wx_date {d} is not a valid YYYYMMDD calendar date")
            }
        }
    }
}

impl std::error::Error for GridError {}

/// Per-window behaviour grids for a fire-growth engine, one weather step.
/// `lb_ratio` is deliberately absent: length-to-breadth is an engine-side
/// quantity (derived from `wsv`), not part of this package's spec.
#[derive(Debug, Clone, PartialEq)]
pub struct BehaviourGrids {
    pub hros: Vec<f64>,
    pub bros: Vec<f64>,
    pub raz: Vec<f64>,
    pub wsv: Vec<f64>,
    pub hfi: Vec<f64>,
    pub rso: Vec<f64>,
    pub sros: Vec<f64>,
    pub sfc: Vec<f64>,
    pub fmc: Vec<f64>,
    pub accel: Vec<f64>,
}

impl GridInput<'_> {
    /// The scalar input for cell `i`.
    fn cell(&self, i: usize) -> FbpInput {
        FbpInput {
            fuel_type: self.fuel_type[i],
            wx_date: self.wx_date,
            lat: self.lat[i],
            long: self.long[i],
            elevation: self.elevation[i],
            slope_pct: self.slope_pct[i],
            aspect_deg: self.aspect_deg[i],
            ws: self.ws[i],
            wd: self.wd[i],
            ffmc: self.ffmc,
            bui: self.bui,
            pc: self.pct_conifer[i],
            pdf: self.pct_dead_fir,
            gfl: self.grass_fuel_load,
            gcf: self.grass_curing[i],
            percentile_growth: self.percentile_growth,
            d0_override: None,
            dj_override: None,
            fmc_override: None,
            hros_override: None,
        }
    }

    fn validate(&self) -> Result<(), GridError> {
        if !is_valid_wx_date(self.wx_date) {
            return Err(GridError::InvalidDate(self.wx_date));
        }
        let expected = self.fuel_type.len();
        for (name, found) in [
            ("lat", self.lat.len()),
            ("long", self.long.len()),
            ("elevation", self.elevation.len()),
            ("slope_pct", self.slope_pct.len()),
            ("aspect_deg", self.aspect_deg.len()),
            ("pct_conifer", self.pct_conifer.len()),
            ("grass_curing", self.grass_curing.len()),
            ("ws", self.ws.len()),
            ("wd", self.wd.len()),
        ] {
            if found != expected {
                return Err(GridError::LengthMismatch {
                    name,
                    expected,
                    found,
                });
            }
        }
        Ok(())
    }
}

/// The grid pass: per-cell fuel/terrain plus per-cell wind, scalar ffmc/bui/
/// date. One call per weather step; non-modeled cells (codes outside 1..18)
/// yield NaN behaviour, which engines exclude via their burnable masks.
pub fn run_grid(input: &GridInput<'_>) -> Result<BehaviourGrids, GridError> {
    input.validate()?;
    let n = input.fuel_type.len();
    let mut out = BehaviourGrids {
        hros: vec![f64::NAN; n],
        bros: vec![f64::NAN; n],
        raz: vec![f64::NAN; n],
        wsv: vec![f64::NAN; n],
        hfi: vec![f64::NAN; n],
        rso: vec![f64::NAN; n],
        sros: vec![f64::NAN; n],
        sfc: vec![f64::NAN; n],
        fmc: vec![f64::NAN; n],
        accel: vec![f64::NAN; n],
    };
    for i in 0..n {
        if !FuelType::from_code(input.fuel_type[i]).is_modeled() {
            continue;
        }
        let r = run(&input.cell(i));
        out.hros[i] = r.hros;
        out.bros[i] = r.bros;
        out.raz[i] = r.raz;
        out.wsv[i] = r.wsv;
        out.hfi[i] = r.hfi;
        out.rso[i] = r.rso;
        out.sros[i] = r.sros;
        out.sfc[i] = r.sfc;
        out.fmc[i] = r.fmc;
        out.accel[i] = r.accel;
    }
    Ok(out)
}
