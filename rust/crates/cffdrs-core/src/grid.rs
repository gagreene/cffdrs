//! The grid pass: per-cell fuel/terrain/wind plus the scalars of one weather
//! step, producing the behaviour grids a fire-growth engine consumes.

use std::fmt;

use crate::fbp::{run, FbpInput};
use crate::fmc::is_valid_wx_date;
use crate::fuel::FuelType;

/// Per-cell input grids plus the scalars of one weather step.
///
/// All slice fields are per-cell and must have the same length as `fuel_type`
/// (checked by [`run_grid`]). Any memory layout works as long as every slice
/// uses the same one. A NaN value is a missing/masked cell input and
/// propagates to that cell's outputs.
///
/// Note that `PartialEq` follows `f64`: an input holding NaN is not equal to
/// itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridInput<'a> {
    /// CFFBPS fuel codes per cell (1..=18 modeled, 19 non-fuel, 20 water). Its length defines the grid size.
    pub fuel_type: &'a [i32],
    /// Latitude per cell, decimal degrees (north positive).
    pub lat: &'a [f64],
    /// Longitude per cell, decimal degrees (west negative).
    pub long: &'a [f64],
    /// Elevation per cell, metres.
    pub elevation: &'a [f64],
    /// Ground slope per cell, percent.
    pub slope_pct: &'a [f64],
    /// Aspect (downhill direction) per cell, compass degrees.
    pub aspect_deg: &'a [f64],
    /// Percent conifer per cell (0-100; used by M-1/M-2).
    pub pct_conifer: &'a [f64],
    /// Grass curing factor per cell, percent (0-100; used by O-1a/O-1b).
    pub grass_curing: &'a [f64],
    /// 10-m open wind speed per cell, km/h.
    pub ws: &'a [f64],
    /// Wind direction per cell, compass degrees.
    pub wd: &'a [f64],
    /// Weather date as `YYYYMMDD`, shared by all cells; must be a real calendar date.
    pub wx_date: i64,
    /// Fine Fuel Moisture Code shared by all cells (0-101).
    pub ffmc: f64,
    /// Buildup Index shared by all cells (>= 0).
    pub bui: f64,
    /// Percent dead balsam fir shared by all cells (0-100; used by M-3/M-4).
    pub pct_dead_fir: f64,
    /// Grass fuel load shared by all cells, kg/m^2 (O-1a/O-1b).
    pub grass_fuel_load: f64,
    /// Percentile (0-100) of the ROS distribution, not a percent change; 50 is the unadjusted ROS. Values outside (0.001, 99.999) are capped; NaN propagates.
    pub percentile_growth: f64,
}

/// Why a grid run was rejected before any cell was computed (returned by
/// [`run_grid`]; the type is `#[non_exhaustive]`, so match with a wildcard arm).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GridError {
    /// A per-cell slice has a different length than `fuel_type`.
    LengthMismatch {
        /// Name of the offending [`GridInput`] field (for example `"lat"`).
        name: &'static str,
        /// The length required, i.e. that of `fuel_type`.
        expected: usize,
        /// The length actually supplied for `name`.
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
///
/// The struct is `#[non_exhaustive]`: read its fields rather than building or
/// destructuring it exhaustively.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BehaviourGrids {
    /// Head fire rate of spread, m/min. NaN where the cell is not a modeled fuel.
    pub hros: Vec<f64>,
    /// Backing fire rate of spread, m/min. NaN where the cell is not a modeled fuel.
    pub bros: Vec<f64>,
    /// Net spread direction, compass degrees. NaN where the cell is not a modeled fuel.
    pub raz: Vec<f64>,
    /// Net effective wind speed, km/h. NaN where the cell is not a modeled fuel.
    pub wsv: Vec<f64>,
    /// Head fire intensity, kW/m. NaN where the cell is not a modeled fuel.
    pub hfi: Vec<f64>,
    /// Critical surface ROS for crowning, m/min. NaN where the cell is not a modeled fuel.
    pub rso: Vec<f64>,
    /// Surface head fire rate of spread, m/min. NaN where the cell is not a modeled fuel.
    pub sros: Vec<f64>,
    /// Surface fuel consumption, kg/m^2. NaN where the cell is not a modeled fuel.
    pub sfc: Vec<f64>,
    /// Foliar moisture content, percent. NaN where the cell is not a modeled fuel.
    pub fmc: Vec<f64>,
    /// Acceleration parameter for fire growth. NaN where the cell is not a modeled fuel.
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
/// yield NaN behaviour, which engines exclude via their burnable masks. A NaN
/// input in a modeled cell is a missing/masked value and propagates to that
/// cell's outputs only.
///
/// Computes each cell with [`crate::fbp::run`] and keeps the ten outputs of
/// [`BehaviourGrids`]; the Python counterpart is `cffdrs._rust.run_fbp_grid`.
///
/// # Errors
///
/// Returns [`GridError::InvalidDate`] if `wx_date` is not a real `YYYYMMDD`
/// calendar date, and [`GridError::LengthMismatch`] if any per-cell slice
/// differs in length from `fuel_type`. Both checks run before any cell is
/// computed.
///
/// # Panics
///
/// Never panics: bad input is reported through the `Err` variants above.
///
/// # Examples
///
/// ```
/// use cffdrs_core::grid::{run_grid, GridError, GridInput};
///
/// let input = GridInput {
///     fuel_type: &[2, 19], // C-2 and a non-fuel cell
///     lat: &[52.0, 52.0],
///     long: &[-115.0, -115.0],
///     elevation: &[800.0, 800.0],
///     slope_pct: &[10.0, 0.0],
///     aspect_deg: &[180.0, 0.0],
///     pct_conifer: &[0.0, 0.0],
///     grass_curing: &[0.0, 0.0],
///     ws: &[20.0, 20.0],
///     wd: &[270.0, 270.0],
///     wx_date: 20_230_615,
///     ffmc: 90.0,
///     bui: 80.0,
///     pct_dead_fir: 0.0,
///     grass_fuel_load: 0.0,
///     percentile_growth: 50.0,
/// };
/// let grids = run_grid(&input).unwrap();
/// assert_eq!(grids.hros.len(), 2);
/// assert!(grids.hros[0] > 0.0);
/// assert!(grids.hros[1].is_nan()); // non-fuel cell: NaN behaviour
///
/// // An impossible date is rejected up front.
/// let bad = GridInput { wx_date: 20_231_345, ..input };
/// assert_eq!(run_grid(&bad).unwrap_err(), GridError::InvalidDate(20_231_345));
/// ```
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
