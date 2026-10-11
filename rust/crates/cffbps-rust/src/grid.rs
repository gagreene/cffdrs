//! The grid pass: one value per cell, or one constant, for every input of one
//! weather step, producing any of the per-cell fields ([`run_grid_fields`]) or
//! the behaviour grids a fire-growth engine consumes ([`run_grid`]).

use std::fmt;
use std::sync::Mutex;

use crate::fbp::{run, FbpInput, FbpResult};
use crate::field::Field;
use crate::fmc::{is_leap_year, is_valid_wx_date};
use crate::fuel::FuelType;
use crate::stage::Stage;
use crate::state::CellState;

/// One grid input: a constant shared by every cell, or one value per cell.
///
/// A NaN value is a missing/masked cell input and propagates to that cell's
/// outputs. `Const` never has a length problem; `Cells` must have one value per
/// cell (checked by [`run_grid_fields`]).
///
/// ```
/// use cffbps_rust::grid::PerCell;
///
/// let constant: PerCell<'_> = 90.0.into();
/// let per_cell: PerCell<'_> = (&[88.0, 91.5][..]).into();
/// assert_eq!(constant, PerCell::Const(90.0));
/// assert_eq!(per_cell, PerCell::Cells(&[88.0, 91.5]));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PerCell<'a> {
    /// The same value for every cell.
    Const(f64),
    /// One value per cell, in the order of [`GridInput::fuel_type`].
    Cells(&'a [f64]),
}

impl From<f64> for PerCell<'_> {
    fn from(value: f64) -> Self {
        PerCell::Const(value)
    }
}

impl<'a> From<&'a [f64]> for PerCell<'a> {
    fn from(values: &'a [f64]) -> Self {
        PerCell::Cells(values)
    }
}

impl<'a, const N: usize> From<&'a [f64; N]> for PerCell<'a> {
    fn from(values: &'a [f64; N]) -> Self {
        PerCell::Cells(values)
    }
}

impl<'a> From<&'a Vec<f64>> for PerCell<'a> {
    fn from(values: &'a Vec<f64>) -> Self {
        PerCell::Cells(values)
    }
}

impl PerCell<'_> {
    /// The value for cell `i` (`Cells` must have been length-checked).
    #[inline]
    pub(crate) fn at(&self, i: usize) -> f64 {
        match self {
            PerCell::Const(v) => *v,
            PerCell::Cells(values) => values[i],
        }
    }

    /// Whether this input fits a grid of `n` cells (`Const` always fits).
    fn check_len(&self, n: usize) -> bool {
        match self {
            PerCell::Const(_) => true,
            PerCell::Cells(values) => values.len() == n,
        }
    }

    /// The number of values supplied (`Const` is one value).
    fn supplied_len(&self) -> usize {
        match self {
            PerCell::Const(_) => 1,
            PerCell::Cells(values) => values.len(),
        }
    }
}

/// An owned [`PerCell`]: a constant, or one value per cell.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum OwnedPerCell {
    /// The same value for every cell.
    Const(f64),
    /// One value per cell.
    Cells(Vec<f64>),
}

impl OwnedPerCell {
    /// The value for cell `i`.
    #[inline]
    pub(crate) fn at(&self, i: usize) -> f64 {
        match self {
            OwnedPerCell::Const(v) => *v,
            OwnedPerCell::Cells(values) => values[i],
        }
    }

    /// A borrowed view.
    pub(crate) fn as_per_cell(&self) -> PerCell<'_> {
        match self {
            OwnedPerCell::Const(v) => PerCell::Const(*v),
            OwnedPerCell::Cells(values) => PerCell::Cells(values),
        }
    }

    /// `f` applied to every value.
    pub(crate) fn map(&self, f: impl Fn(f64) -> f64) -> OwnedPerCell {
        match self {
            OwnedPerCell::Const(v) => OwnedPerCell::Const(f(*v)),
            OwnedPerCell::Cells(values) => {
                OwnedPerCell::Cells(values.iter().map(|&v| f(v)).collect())
            }
        }
    }

    /// Bytes held (a constant counts as one value).
    pub(crate) fn bytes(&self) -> usize {
        match self {
            OwnedPerCell::Const(_) => std::mem::size_of::<f64>(),
            OwnedPerCell::Cells(values) => std::mem::size_of_val(values.as_slice()),
        }
    }
}

impl From<PerCell<'_>> for OwnedPerCell {
    fn from(p: PerCell<'_>) -> Self {
        match p {
            PerCell::Const(v) => OwnedPerCell::Const(v),
            PerCell::Cells(values) => OwnedPerCell::Cells(values.to_vec()),
        }
    }
}

/// Every input of one weather step, each a constant or one value per cell.
///
/// The grid size is the length of `fuel_type`; every [`PerCell::Cells`] input
/// must have that length (checked by [`run_grid_fields`] and [`run_grid`]).
/// Any memory layout works as long as every slice uses the same one. A NaN
/// value is a missing/masked cell input and propagates to that cell's outputs.
///
/// Missing values are never defaulted here: a NaN `pct_dead_fir` or
/// `grass_fuel_load` (even a NaN `Const`) is a missing input in every cell.
/// The Python package replaces a NaN *scalar* `pdf` / `gfl` / `pc` / `gcf` by
/// its default (35 / 0.35 / 50 / 80) in `initialize`; callers that mirror it
/// apply those defaults before building the grid, as the `cffdrs._rust`
/// binding does.
///
/// Note that `PartialEq` follows `f64`: an input holding NaN is not equal to
/// itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridInput<'a> {
    /// CFFBPS fuel codes per cell (1..=18 modeled, 19 non-fuel, 20 water; anything else is invalid). Its length defines the grid size.
    pub fuel_type: &'a [i32],
    /// Latitude, decimal degrees (north positive).
    pub lat: PerCell<'a>,
    /// Longitude, decimal degrees (west negative); only its magnitude is used.
    pub long: PerCell<'a>,
    /// Elevation, metres.
    pub elevation: PerCell<'a>,
    /// Ground slope, percent.
    pub slope_pct: PerCell<'a>,
    /// Aspect (downhill direction), compass degrees.
    pub aspect_deg: PerCell<'a>,
    /// Percent conifer (0-100; used by M-1/M-2).
    pub pct_conifer: PerCell<'a>,
    /// Percent dead balsam fir (0-100; used by M-3/M-4). NaN is not defaulted.
    pub pct_dead_fir: PerCell<'a>,
    /// Grass fuel load, kg/m^2 (O-1a/O-1b). NaN is not defaulted.
    pub grass_fuel_load: PerCell<'a>,
    /// Grass curing factor, percent (0-100; used by O-1a/O-1b).
    pub grass_curing: PerCell<'a>,
    /// 10-m open wind speed, km/h.
    pub ws: PerCell<'a>,
    /// Wind direction, compass degrees.
    pub wd: PerCell<'a>,
    /// Fine Fuel Moisture Code (0-101).
    pub ffmc: PerCell<'a>,
    /// Buildup Index (>= 0).
    pub bui: PerCell<'a>,
    /// Weather date as `YYYYMMDD`, shared by all cells; must be a real calendar date.
    pub wx_date: i64,
    /// Percentile (0-100) of the ROS distribution, not a percent change; 50 is the unadjusted ROS. Values outside (0.001, 99.999) are capped; NaN propagates.
    pub percentile_growth: f64,
    /// Explicit date of minimum foliar moisture (`d0`), used as given (no
    /// range limit); `None` derives it from latitude and elevation. A NaN value
    /// is a masked override: that cell's `d0`, `nd`, `fmc` and `fme` are NaN.
    pub d0: Option<PerCell<'a>>,
    /// Explicit day of year of the fire (`dj`); `None` derives it from
    /// `wx_date`. A supplied value is floored and then clipped to 1..365
    /// (1..366 in a leap year of `wx_date`); infinity is rejected
    /// ([`GridError::InfiniteDj`]). A NaN value is a masked override: that
    /// cell's `dj`, `nd`, `fmc` and `fme` are NaN.
    pub dj: Option<PerCell<'a>>,
}

/// Why a grid run was rejected before any cell was computed (returned by
/// [`run_grid_fields`] and [`run_grid`]; the type is `#[non_exhaustive]`, so
/// match with a wildcard arm).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GridError {
    /// A per-cell slice ([`PerCell::Cells`]) has a different length than `fuel_type`.
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
    /// A supplied `dj` override contains an unmasked infinity, which has no
    /// day of year (it is rejected, not clipped).
    InfiniteDj,
    /// [`crate::fbp_grid::FbpGrid::set_override`] was given a field other
    /// than [`Field::D0Override`] or [`Field::DjOverride`].
    NotAnOverride(Field),
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
            GridError::InfiniteDj => {
                write!(f, "dj override must be finite (it contains infinity)")
            }
            GridError::NotAnOverride(field) => write!(
                f,
                "{} is not an override: use d0_override or dj_override",
                field.name()
            ),
        }
    }
}

impl std::error::Error for GridError {}

/// What a grid pass gives a cell whose fuel code is not modeled (19 non-fuel,
/// 20 water, or an invalid code such as 0, 99 or a negative code).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonModeled {
    /// Every requested field is NaN: the engine contract of [`run_grid`]
    /// (engines exclude such cells through their burnable masks).
    Nan,
    /// Every output equals what the Python package's `FBP.getParams` returns
    /// for that cell (for example `hros` NaN, `fire_type` 0, `fi_class` -99,
    /// `cfb` 0 for codes 19/20 and NaN for invalid codes, `fmc` as computed,
    /// `fuel_type` 0 for invalid codes). Input-only and internal fields keep
    /// their computed values.
    PythonCompat,
}

/// The fields a [`run_grid_fields`] call asked for, in the order first asked,
/// each with one value per cell.
///
/// The struct is `#[non_exhaustive]`: read its fields rather than building or
/// destructuring it exhaustively.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GridFields {
    /// The requested fields (each once) and their values, one per cell.
    pub fields: Vec<(Field, Vec<f64>)>,
    /// How many supplied `dj` override values were changed by clipping to the
    /// days of the year of `wx_date` (below 1, or above 365/366); flooring a
    /// fraction does not count, nor do NaN values. A [`PerCell::Const`]
    /// override is one supplied value, so it counts at most 1. 0 without a
    /// `dj` override. Callers use it to announce clipping.
    pub dj_override_clipped: usize,
}

impl GridFields {
    /// The values of `field`, or `None` when it was not requested.
    #[must_use]
    pub fn get(&self, field: Field) -> Option<&[f64]> {
        self.fields
            .iter()
            .find(|(f, _)| *f == field)
            .map(|(_, v)| v.as_slice())
    }
}

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

/// The ten [`BehaviourGrids`] fields, in its field order.
const BEHAVIOUR_FIELDS: [Field; 10] = [
    Field::Hros,
    Field::Bros,
    Field::Raz,
    Field::Wsv,
    Field::Hfi,
    Field::Rso,
    Field::Sros,
    Field::Sfc,
    Field::Fmc,
    Field::Accel,
];

/// A supplied `dj` override resolved to the days of the year of `wx_date`
/// (D15.6 of the FMC contract): each value is floored, then clipped to 1..365,
/// or 1..366 in a leap year; NaN stays NaN (a masked override). Returns the
/// resolved values and how many supplied values clipping changed (flooring a
/// fraction does not count; a `Const` is one value).
///
/// # Errors
///
/// [`GridError::InfiniteDj`] if any supplied value is infinite.
pub(crate) fn resolve_dj(
    supplied: PerCell<'_>,
    wx_date: i64,
) -> Result<(OwnedPerCell, usize), GridError> {
    let last_day = if is_leap_year(wx_date / 10_000) {
        366.0
    } else {
        365.0
    };
    // (resolved, changed by clipping); NaN stays NaN and is not counted
    let one = |v: f64| -> Result<(f64, bool), GridError> {
        if v.is_infinite() {
            return Err(GridError::InfiniteDj);
        }
        let day = v.floor();
        let clipped = day.clamp(1.0, last_day);
        Ok((clipped, clipped != day && !v.is_nan()))
    };
    match supplied {
        PerCell::Const(v) => {
            let (day, changed) = one(v)?;
            Ok((OwnedPerCell::Const(day), usize::from(changed)))
        }
        PerCell::Cells(values) => {
            let mut days = Vec::with_capacity(supplied.supplied_len());
            let mut changed = 0;
            for &v in values {
                let (day, c) = one(v)?;
                days.push(day);
                changed += usize::from(c);
            }
            Ok((OwnedPerCell::Cells(days), changed))
        }
    }
}

impl GridInput<'_> {
    /// The checks of [`run_grid_fields`], in its documented order; on success
    /// the resolved `dj` override and its clipped count.
    pub(crate) fn validate(&self) -> Result<(Option<OwnedPerCell>, usize), GridError> {
        if !is_valid_wx_date(self.wx_date) {
            return Err(GridError::InvalidDate(self.wx_date));
        }
        let expected = self.fuel_type.len();
        for (name, input) in [
            ("lat", Some(self.lat)),
            ("long", Some(self.long)),
            ("elevation", Some(self.elevation)),
            ("slope_pct", Some(self.slope_pct)),
            ("aspect_deg", Some(self.aspect_deg)),
            ("pct_conifer", Some(self.pct_conifer)),
            ("pct_dead_fir", Some(self.pct_dead_fir)),
            ("grass_fuel_load", Some(self.grass_fuel_load)),
            ("grass_curing", Some(self.grass_curing)),
            ("ws", Some(self.ws)),
            ("wd", Some(self.wd)),
            ("ffmc", Some(self.ffmc)),
            ("bui", Some(self.bui)),
            ("d0", self.d0),
            ("dj", self.dj),
        ] {
            if let Some(p) = input {
                if !p.check_len(expected) {
                    return Err(GridError::LengthMismatch {
                        name,
                        expected,
                        found: p.supplied_len(),
                    });
                }
            }
        }
        match self.dj {
            None => Ok((None, 0)),
            Some(dj) => {
                let (days, clipped) = resolve_dj(dj, self.wx_date)?;
                Ok((Some(days), clipped))
            }
        }
    }
}

/// One validated grid pass: the input, the resolved `dj` override and what to
/// compute.
struct Pass<'i, 'a> {
    input: &'i GridInput<'a>,
    dj: Option<OwnedPerCell>,
    fields: Vec<Field>,
    /// A requested field is not an output, so cells run stage by stage.
    staged: bool,
    non_modeled: NonModeled,
}

impl Pass<'_, '_> {
    /// The scalar input for cell `i`.
    #[inline]
    fn cell(&self, i: usize) -> FbpInput {
        let g = self.input;
        FbpInput {
            fuel_type: g.fuel_type[i],
            wx_date: g.wx_date,
            lat: g.lat.at(i),
            long: g.long.at(i),
            elevation: g.elevation.at(i),
            slope_pct: g.slope_pct.at(i),
            aspect_deg: g.aspect_deg.at(i),
            ws: g.ws.at(i),
            wd: g.wd.at(i),
            ffmc: g.ffmc.at(i),
            bui: g.bui.at(i),
            pc: g.pct_conifer.at(i),
            pdf: g.pct_dead_fir.at(i),
            gfl: g.grass_fuel_load.at(i),
            gcf: g.grass_curing.at(i),
            percentile_growth: g.percentile_growth,
            d0_override: g.d0.map(|d0| d0.at(i)),
            dj_override: self.dj.as_ref().map(|dj| dj.at(i)),
            fmc_override: None,
            hros_override: None,
        }
    }

    /// Computes cells `start..start + len` into `part` (one slice per
    /// requested field). Cells skipped here keep the NaN they were
    /// initialised with.
    fn fill(&self, start: usize, len: usize, part: &mut [&mut [f64]]) {
        // One loop per path keeps the hot fused loop small.
        if self.staged {
            self.fill_with(start, len, part, |cell, fields, out, j| {
                let mut s = CellState::from_input(cell);
                for stage in Stage::ALL {
                    s.apply(stage);
                }
                for (o, &f) in out.iter_mut().zip(fields) {
                    o[j] = s.get(f);
                }
            });
        } else if self.fields == BEHAVIOUR_FIELDS {
            // The engine's ten fields with a fixed layout: after inlining `run`
            // the compiler skips work no stored field needs (measured faster).
            self.fill_with(start, len, part, |cell, _, out, j| {
                let r = run(cell);
                let values = [
                    r.hros, r.bros, r.raz, r.wsv, r.hfi, r.rso, r.sros, r.sfc, r.fmc, r.accel,
                ];
                for (o, v) in out.iter_mut().zip(values) {
                    o[j] = v;
                }
            });
        } else {
            self.fill_with(start, len, part, |cell, fields, out, j| {
                let r = run(cell);
                for (o, &f) in out.iter_mut().zip(fields) {
                    o[j] = output_value(&r, f);
                }
            });
        }
    }

    /// The cell loop of [`Pass::fill`]; `compute` writes cell `j` of every
    /// requested field.
    #[inline]
    fn fill_with(
        &self,
        start: usize,
        len: usize,
        part: &mut [&mut [f64]],
        compute: impl Fn(&FbpInput, &[Field], &mut [&mut [f64]], usize),
    ) {
        for j in 0..len {
            let i = start + j;
            let ft = FuelType::from_code(self.input.fuel_type[i]);
            let modeled = ft.is_modeled();
            if !modeled && self.non_modeled == NonModeled::Nan {
                continue;
            }
            let cell = self.cell(i);
            compute(&cell, &self.fields, part, j);
            if !modeled {
                for (o, &f) in part.iter_mut().zip(&self.fields) {
                    o[j] = python_non_modeled(f, ft, &cell, o[j]);
                }
            }
        }
    }
}

/// The value of output field `f` in `r` (the order of
/// [`FbpResult::named_values`]); NaN for a field that is not an output.
#[inline]
fn output_value(r: &FbpResult, f: Field) -> f64 {
    match f {
        Field::Ws => r.ws,
        Field::Wd => r.wd,
        Field::Wse => r.wse,
        Field::Wse1 => r.wse1,
        Field::Wse2 => r.wse2,
        Field::Wsx => r.wsx,
        Field::Wsy => r.wsy,
        Field::Wsv => r.wsv,
        Field::Raz => r.raz,
        Field::M => r.m,
        Field::FF => r.f_f,
        Field::FW => r.f_w,
        Field::Ffmc => r.ffmc,
        Field::Isi => r.isi,
        Field::Bui => r.bui,
        Field::A => r.a,
        Field::B => r.b,
        Field::C => r.c,
        Field::Q => r.q,
        Field::Bui0 => r.bui0,
        Field::Be => r.be,
        Field::BeMax => r.be_max,
        Field::Sf => r.sf,
        Field::Rsz => r.rsz,
        Field::Rsf => r.rsf,
        Field::Isf => r.isf,
        Field::Rsi => r.rsi,
        Field::Latn => r.latn,
        Field::Dj => r.dj,
        Field::D0 => r.d0,
        Field::Nd => r.nd,
        Field::Fmc => r.fmc,
        Field::Fme => r.fme,
        Field::Ffc => r.ffc,
        Field::Wfc => r.wfc,
        Field::Sfc => r.sfc,
        Field::Cfl => r.cfl,
        Field::Cfc => r.cfc,
        Field::Tfc => r.tfc,
        Field::Cbh => r.cbh,
        Field::Csfi => r.csfi,
        Field::Rso => r.rso,
        Field::Cfb => r.cfb,
        Field::FireType => r.fire_type,
        Field::Hros => r.hros,
        Field::Sros => r.sros,
        Field::Cros => r.cros,
        Field::Bfw => r.bfw,
        Field::Bisi => r.bisi,
        Field::Bros => r.bros,
        Field::Hfi => r.hfi,
        Field::FiClass => r.fi_class,
        Field::Accel => r.accel,
        Field::FuelType => r.fuel_type,
        _ => f64::NAN,
    }
}

/// [`NonModeled::PythonCompat`]: the value the Python package's `getParams`
/// gives output `f` of a cell whose fuel `ft` is not modeled, given the cell's
/// inputs and the `computed` value. Derived from the golden snapshot
/// `non_modeled_snapshot.json`, not from [`run`]: Python leaves most fields of
/// such a cell at their initial template, masks the rest, and computes only
/// the fuel-independent chains (wind and moisture inputs, slope factor,
/// foliar moisture). Codes 19 (non-fuel) and 20 (water) are valid, unmasked
/// codes and differ from invalid codes, which the fuel validation masks (and
/// exports as code 0). Input-only and internal fields keep `computed`.
#[allow(clippy::match_same_arms)] // one arm per rule of the table, kept apart on purpose
pub(crate) fn python_non_modeled(f: Field, ft: FuelType, cell: &FbpInput, computed: f64) -> f64 {
    const NAN: f64 = f64::NAN;
    // NaN where any of `inputs` is missing, else `value`
    let unless_missing = |inputs: &[f64], value: f64| {
        if inputs.iter().any(|v| v.is_nan()) {
            NAN
        } else {
            value
        }
    };
    // 19 and 20; everything else reaching here is an invalid code
    let valid = matches!(ft, FuelType::NonFuel | FuelType::Water);
    match f {
        // computed from inputs that do not depend on the fuel
        Field::Ws
        | Field::Wd
        | Field::M
        | Field::FF
        | Field::Ffmc
        | Field::Bui
        | Field::Sf
        | Field::Latn
        | Field::D0
        | Field::Nd
        | Field::Fmc
        | Field::Fme => computed,
        // an integer output: a derived day of year that is missing (NaN
        // geography) exports as 0; a masked override stays NaN
        Field::Dj => {
            if computed.is_nan() && cell.dj_override.is_none() {
                0.0
            } else {
                computed
            }
        }
        // the zero template
        Field::FireType | Field::A | Field::B | Field::C | Field::Q | Field::Accel => 0.0,
        Field::FiClass => -99.0,
        Field::FuelType => {
            if valid {
                f64::from(ft.code())
            } else {
                0.0
            }
        }
        // masked for every non-modeled code
        Field::Hros
        | Field::Hfi
        | Field::FW
        | Field::Isi
        | Field::Isf
        | Field::Rsi
        | Field::Wse1
        | Field::Wse2
        | Field::Wse
        | Field::Wsx
        | Field::Wsy
        | Field::Wsv
        | Field::Raz
        | Field::Ffc
        | Field::Wfc
        | Field::Sfc
        | Field::Rso
        | Field::Bfw
        | Field::Bisi
        | Field::Bros
        | Field::Cfc
        | Field::Tfc => NAN,
        // 19/20: zero unless an input of the equation is missing; invalid: masked
        Field::Rsz if valid => unless_missing(&[cell.ffmc], 0.0),
        Field::Rsf if valid => unless_missing(&[cell.ffmc, cell.slope_pct], 0.0),
        Field::Rsz | Field::Rsf => NAN,
        // 19/20: 1; invalid: 0
        Field::Bui0 | Field::BeMax => {
            if valid {
                1.0
            } else {
                0.0
            }
        }
        // 19/20: masked; invalid: zero unless bui is missing
        Field::Be if valid => NAN,
        Field::Be => unless_missing(&[cell.bui], 0.0),
        // 19/20: zero; invalid: masked
        Field::Csfi | Field::Sros | Field::Cros | Field::Cfb => {
            if valid {
                0.0
            } else {
                NAN
            }
        }
        // 19/20: masked; invalid: zero
        Field::Cbh | Field::Cfl => {
            if valid {
                NAN
            } else {
                0.0
            }
        }
        // input-only and internal fields: Python has no such output
        _ => computed,
    }
}

/// The grid pass: computes the requested `fields` for every cell.
///
/// Each cell is the scalar [`crate::fbp::run`] of its inputs, with the
/// explicit `d0` / `dj` overrides of [`GridInput`] (a NaN override masks that
/// cell's foliar moisture). Fields that are not outputs (the input-only and
/// internal fields of [`Field`]) may be requested too; the cell is then
/// computed stage by stage ([`crate::state::CellState`]), which gives the same
/// outputs bit for bit. A field listed twice is returned once, at its first
/// position. `non_modeled` chooses what a cell whose fuel is not modeled gives.
///
/// `threads` works as in [`run_grid_with_threads`]: 1 computes on the calling
/// thread, 0 uses every available core, and grids under 4,096 cells always run
/// serially; the result is identical for every value.
///
/// # Errors
///
/// [`GridError::InvalidDate`] if `wx_date` is not a real `YYYYMMDD` calendar
/// date, then [`GridError::LengthMismatch`] for the first [`PerCell::Cells`]
/// input (in [`GridInput`] field order) whose length differs from `fuel_type`,
/// then [`GridError::InfiniteDj`] for an infinite `dj` override. Every check
/// runs before any cell is computed.
///
/// # Panics
///
/// Never panics on bad input; it panics only if a worker thread panics.
///
/// # Examples
///
/// ```
/// use cffbps_rust::field::Field;
/// use cffbps_rust::grid::{run_grid_fields, GridInput, NonModeled, PerCell};
///
/// let input = GridInput {
///     fuel_type: &[2, 19], // C-2 and a non-fuel cell
///     lat: 52.0.into(),
///     long: (-115.0).into(),
///     elevation: 800.0.into(),
///     slope_pct: (&[10.0, 0.0]).into(),
///     aspect_deg: 180.0.into(),
///     pct_conifer: 0.0.into(),
///     pct_dead_fir: 0.0.into(),
///     grass_fuel_load: 0.0.into(),
///     grass_curing: 0.0.into(),
///     ws: 20.0.into(),
///     wd: 270.0.into(),
///     ffmc: 90.0.into(),
///     bui: 80.0.into(),
///     wx_date: 20_230_615,
///     percentile_growth: 50.0,
///     d0: None,
///     dj: Some(PerCell::Const(166.7)), // floored to day 166
/// };
/// let out = run_grid_fields(&input, &[Field::Hros, Field::Dj, Field::FireType], 1, NonModeled::PythonCompat)
///     .unwrap();
/// assert!(out.get(Field::Hros).unwrap()[0] > 0.0);
/// assert!(out.get(Field::Hros).unwrap()[1].is_nan());
/// assert_eq!(out.get(Field::Dj).unwrap(), &[166.0, 166.0]);
/// assert_eq!(out.get(Field::FireType).unwrap()[1], 0.0); // Python's value for a non-fuel cell
/// assert_eq!(out.dj_override_clipped, 0);
/// ```
pub fn run_grid_fields(
    input: &GridInput<'_>,
    fields: &[Field],
    threads: usize,
    non_modeled: NonModeled,
) -> Result<GridFields, GridError> {
    let (dj, dj_override_clipped) = input.validate()?;
    let mut unique: Vec<Field> = Vec::with_capacity(fields.len());
    for &f in fields {
        if !unique.contains(&f) {
            unique.push(f);
        }
    }
    let n = input.fuel_type.len();
    let mut out: Vec<Vec<f64>> = vec![vec![f64::NAN; n]; unique.len()];
    let pass = Pass {
        input,
        dj,
        staged: unique.iter().any(|f| !f.is_output()),
        fields: unique,
        non_modeled,
    };
    if !pass.fields.is_empty() {
        fill_threaded(&pass, n, threads, &mut out);
    }
    Ok(GridFields {
        fields: pass.fields.into_iter().zip(out).collect(),
        dj_override_clipped,
    })
}

/// Computes every cell of `pass` into `out` (one vector of `n` values per
/// requested field), on the calling thread or on `threads` workers.
fn fill_threaded(pass: &Pass<'_, '_>, n: usize, threads: usize, out: &mut [Vec<f64>]) {
    fill_chunks(n, threads, out, |start, len, part| {
        pass.fill(start, len, part);
    });
}

/// Calls `fill(start, len, part)` for consecutive chunks of the `n` cells,
/// `part` holding cells `start..start + len` of every vector of `out`: once on
/// the calling thread when `threads` is 1 or the grid has fewer than
/// [`MIN_PARALLEL_CELLS`] cells, otherwise from `threads` workers (0 = every
/// available core) taking small chunks from a shared queue. Every cell is
/// filled exactly once, so the result does not depend on `threads`.
pub(crate) fn fill_chunks(
    n: usize,
    threads: usize,
    out: &mut [Vec<f64>],
    fill: impl Fn(usize, usize, &mut [&mut [f64]]) + Sync,
) {
    let threads = if threads == 0 {
        std::thread::available_parallelism().map_or(1, usize::from)
    } else {
        threads
    };
    let mut rest: Vec<&mut [f64]> = out.iter_mut().map(Vec::as_mut_slice).collect();
    if threads <= 1 || n < MIN_PARALLEL_CELLS {
        fill(0, n, &mut rest);
        return;
    }
    // Many small chunks handed out one at a time keep threads busy when fuel
    // types cost different amounts (non-modeled cells are nearly free).
    let chunk_len = n.div_ceil(threads * 8).max(1);
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < n {
        let len = chunk_len.min(n - start);
        let part: Vec<&mut [f64]> = rest
            .iter_mut()
            .map(|s| {
                let (head, tail) = std::mem::take(s).split_at_mut(len);
                *s = tail;
                head
            })
            .collect();
        chunks.push((start, len, part));
        start += len;
    }
    let queue = Mutex::new(chunks.into_iter());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let next = queue.lock().expect("queue poisoned").next();
                match next {
                    Some((chunk_start, len, mut part)) => fill(chunk_start, len, &mut part),
                    None => break,
                }
            });
        }
    });
}

/// The grid pass for a fire-growth engine: the ten [`BehaviourGrids`] for
/// every cell. One call per weather step; non-modeled cells (codes outside
/// 1..18) yield NaN behaviour, which engines exclude via their burnable masks.
/// A NaN input in a modeled cell is a missing/masked value and propagates to
/// that cell's outputs only.
///
/// [`run_grid_fields`] with the ten fields and [`NonModeled::Nan`] on the
/// calling thread; the Python counterpart is `cffdrs._rust.run_fbp_grid`.
///
/// # Errors
///
/// The same as [`run_grid_fields`].
///
/// # Panics
///
/// Never panics: bad input is reported through the `Err` variants above.
///
/// # Examples
///
/// ```
/// use cffbps_rust::grid::{run_grid, GridError, GridInput};
///
/// let input = GridInput {
///     fuel_type: &[2, 19], // C-2 and a non-fuel cell
///     lat: (&[52.0, 52.0]).into(),
///     long: (&[-115.0, -115.0]).into(),
///     elevation: (&[800.0, 800.0]).into(),
///     slope_pct: (&[10.0, 0.0]).into(),
///     aspect_deg: (&[180.0, 0.0]).into(),
///     pct_conifer: 0.0.into(),
///     pct_dead_fir: 0.0.into(),
///     grass_fuel_load: 0.0.into(),
///     grass_curing: 0.0.into(),
///     ws: 20.0.into(),
///     wd: 270.0.into(),
///     ffmc: 90.0.into(),
///     bui: 80.0.into(),
///     wx_date: 20_230_615,
///     percentile_growth: 50.0,
///     d0: None,
///     dj: None,
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
    run_grid_with_threads(input, 1)
}

/// [`run_grid`] with an explicit number of worker threads.
///
/// `threads == 1` computes on the calling thread, exactly like [`run_grid`].
/// `threads == 0` means "all available cores"; any other value is the number
/// of threads to use (capped by the number of work chunks). Grids with fewer
/// than 4,096 cells always run on the calling thread, because thread start-up
/// costs more than it saves.
///
/// Every cell is computed by the same [`crate::fbp::run`] into its own output
/// slot, so the result is identical for every `threads` value; threading only
/// changes how long the call takes and how many cores it occupies. The library
/// never starts threads unless asked: a caller that runs many grids at once
/// should keep `threads == 1` and parallelise across grids instead.
///
/// # Errors
///
/// The same as [`run_grid`].
///
/// # Panics
///
/// Never panics on bad input; it panics only if a worker thread panics.
pub fn run_grid_with_threads(
    input: &GridInput<'_>,
    threads: usize,
) -> Result<BehaviourGrids, GridError> {
    let out = run_grid_fields(input, &BEHAVIOUR_FIELDS, threads, NonModeled::Nan)?;
    let mut values = out.fields.into_iter().map(|(_, v)| v);
    let mut next = || values.next().unwrap_or_default();
    Ok(BehaviourGrids {
        hros: next(),
        bros: next(),
        raz: next(),
        wsv: next(),
        hfi: next(),
        rso: next(),
        sros: next(),
        sfc: next(),
        fmc: next(),
        accel: next(),
    })
}

/// Below this many cells the thread start-up cost outweighs the gain.
const MIN_PARALLEL_CELLS: usize = 4096;
