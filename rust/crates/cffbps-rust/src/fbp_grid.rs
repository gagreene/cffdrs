//! Pipeline stages over whole grids, holding only the arrays still needed.
//!
//! [`FbpGrid`] keeps one array per [`Field`] for a grid of cells and runs the
//! [`Stage`]s over it one at a time ([`FbpGrid::run_stage`],
//! [`FbpGrid::run_planned`]) or all at once through the fused per-cell pass
//! ([`FbpGrid::run_fused`]). Between stages a caller can read any field
//! ([`FbpGrid::get`]) or inject values ([`FbpGrid::set`]), as with the Python
//! `FBP` object's `calcX` methods and `setParams` / `getParams`.
//!
//! # Memory
//!
//! The grid owns a copy of every input (normalised as
//! [`fbp::run`](crate::fbp::run) normalises them; a constant stays one value)
//! and never frees it. A computed field is stored only while a planned stage
//! that has not run yet reads it, or when it is in `out_request` or the
//! `keep` list of [`FbpGrid::plan`]; otherwise it is dropped right after its
//! last planned reader ran, or not stored at all. Liveness depends on the
//! *set* of planned stages and which of them have run, not on their order, so
//! stages may run in any order. A field that was dropped is an error to read
//! ([`StageError::FreedRead`]) or to run a stage on
//! ([`StageError::Freed`]); a field that was never computed reads as its
//! template ([`Field::template_value`]; `wd` and `aspect` read as their
//! inputs until `invertWindAspect` runs).
//!
//! # Lifecycle
//!
//! A new grid is *fresh*; after a stage ran it is *stepping*; after
//! [`FbpGrid::run_fused`] it is *fused*. [`FbpGrid::run_fused`] always starts
//! from the stored inputs: it first discards every computed or injected value,
//! so repeated calls give identical results. Only `out_request` is stored by
//! the fused pass, so a stage after it returns [`StageError::AfterFused`] until
//! [`FbpGrid::reset`], which drops every computed field and restarts the
//! liveness bookkeeping. The inputs (including [`FbpGrid::set`] on an input,
//! [`FbpGrid::set_fuel`], the scalar parameters and the `d0` / `dj`
//! overrides) persist across both.
//!
//! Non-modeled cells (fuel codes 19, 20 and invalid codes) get the Python
//! package's values ([`NonModeled::PythonCompat`]) in both paths.
//!
//! # Examples
//!
//! ```
//! use cffbps_rust::fbp_grid::{FbpGrid, StageError};
//! use cffbps_rust::field::Field;
//! use cffbps_rust::grid::GridInput;
//! use cffbps_rust::stage::Stage;
//!
//! let input = GridInput {
//!     fuel_type: &[2, 6],
//!     lat: 52.0.into(),
//!     long: (-115.0).into(),
//!     elevation: 800.0.into(),
//!     slope_pct: 10.0.into(),
//!     aspect_deg: 180.0.into(),
//!     pct_conifer: 50.0.into(),
//!     pct_dead_fir: 35.0.into(),
//!     grass_fuel_load: 0.35.into(),
//!     grass_curing: 80.0.into(),
//!     ws: 20.0.into(),
//!     wd: 270.0.into(),
//!     ffmc: 90.0.into(),
//!     bui: 80.0.into(),
//!     wx_date: 20_230_615,
//!     percentile_growth: 50.0,
//!     d0: None,
//!     dj: None,
//! };
//! let mut grid = FbpGrid::new(&input, &[Field::Hros, Field::Hfi]).unwrap();
//! for stage in Stage::ALL {
//!     if stage == Stage::CalcFmc {
//!         grid.set(Field::Fmc, &[95.0, 95.0]).unwrap(); // in place of calcFMC
//!     } else {
//!         grid.run_stage(stage, 1).unwrap();
//!     }
//! }
//! assert!(grid.get(Field::Hfi).unwrap()[0] > 0.0);
//! // sfc was dropped after calcTFC, its last planned reader
//! assert_eq!(grid.get(Field::Sfc), Err(StageError::FreedRead(Field::Sfc)));
//! ```

use std::fmt;

use crate::fbp::{is_valid_wx_date, FbpInput};
use crate::field::Field;
use crate::fmc::fme_from_fmc;
use crate::fuel::FuelType;
use crate::grid::{
    fill_chunks, python_non_modeled, resolve_dj, run_grid_fields, GridError, GridInput, NonModeled,
    OwnedPerCell, PerCell,
};
use crate::liveness::Liveness;
use crate::normalize::normalize_input;
use crate::stage::Stage;
use crate::state::{CellState, StageParams};

/// Why a stage, `get` or `set` call on an [`FbpGrid`] was refused.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum StageError {
    /// `stage` needs `field`, which was dropped after every planned stage
    /// that reads it had run.
    Freed {
        /// The dropped field.
        field: Field,
        /// The stage that needs it.
        stage: Stage,
    },
    /// A stage was run after [`FbpGrid::run_fused`] without
    /// [`FbpGrid::reset`].
    AfterFused,
    /// [`FbpGrid::get`] of a field that was dropped.
    FreedRead(Field),
    /// [`FbpGrid::set`] with a number of values other than the number of cells.
    LengthMismatch {
        /// The field being set.
        field: Field,
        /// The number of cells.
        expected: usize,
        /// The number of values supplied.
        found: usize,
    },
    /// The field has its own route: `fuel_type` ([`FbpGrid::set_fuel`]),
    /// `d0_override` and `dj_override` ([`FbpGrid::set_override`]).
    NotSettable(Field),
}

impl fmt::Display for StageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StageError::Freed { field, stage } => write!(
                f,
                "{} needs {}, which was freed after the last planned stage reading it ran; \
                 add it to out_request or to the keep list of plan()",
                stage.name(),
                field.name()
            ),
            StageError::AfterFused => write!(
                f,
                "the fused pass stores only out_request: call reset() before running stages"
            ),
            StageError::FreedRead(field) => write!(
                f,
                "{} was freed after the last planned stage reading it ran; \
                 add it to out_request to read it",
                field.name()
            ),
            StageError::LengthMismatch {
                field,
                expected,
                found,
            } => write!(
                f,
                "{} has {found} values but the grid has {expected} cells",
                field.name()
            ),
            StageError::NotSettable(field) => write!(
                f,
                "{} cannot be set: use set_fuel for fuel_type and set_override for \
                 d0_override and dj_override",
                field.name()
            ),
        }
    }
}

impl std::error::Error for StageError {}

/// A scalar stage parameter changed with [`FbpGrid::set_param`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParamName {
    /// `wx_date` (`YYYYMMDD`), read by `calcFMC`.
    WxDate,
    /// `percentile_growth`, read by `calcRosPercentileGrowth`.
    PercentileGrowth,
}

/// The 13 stored inputs (normalised; never freed). `fuel_type` and the
/// `d0` / `dj` overrides are held separately.
const INPUTS: [Field; 13] = [
    Field::Lat,
    Field::Long,
    Field::Elevation,
    Field::Slope,
    Field::AspectIn,
    Field::Pc,
    Field::Pdf,
    Field::Gfl,
    Field::Gcf,
    Field::Ws,
    Field::WdIn,
    Field::Ffmc,
    Field::Bui,
];

/// The six outputs of `calcFMC`.
const FMC_OUTPUTS: [Field; 6] = [
    Field::Latn,
    Field::D0,
    Field::Dj,
    Field::Nd,
    Field::Fmc,
    Field::Fme,
];

/// The inputs of `calcFMC` with a version counter (D15.4 of the FMC contract).
#[derive(Clone, Copy)]
enum FmcInput {
    Lat,
    Long,
    Elevation,
    D0,
    Dj,
    WxDate,
}

/// What a field holds.
#[derive(Clone, Debug)]
enum Slot {
    /// An input, normalised; never freed.
    Input(OwnedPerCell),
    /// A computed field that was never written: reads as its template.
    Never,
    /// A computed or injected value.
    Stored(OwnedPerCell),
    /// A computed field that was dropped (or computed and never stored).
    Freed,
}

/// Fresh (no stage ran), stepping (a stage ran) or fused (`run_fused` ran).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Fresh,
    Stepping,
    Fused,
}

/// An explicit `dj` override: the values as supplied and resolved to the
/// days of the year of `wx_date`.
#[derive(Clone, Debug)]
struct DjOverride {
    supplied: OwnedPerCell,
    resolved: OwnedPerCell,
    clipped: usize,
}

/// A grid of cells run through the pipeline stages one at a time or fused,
/// holding only the arrays still needed (see the [module docs](self)).
///
/// The grid owns copies of its inputs; changes reach it only through
/// [`FbpGrid::set`], [`FbpGrid::set_const`], [`FbpGrid::set_fuel`],
/// [`FbpGrid::set_param`] and [`FbpGrid::set_override`]. Building a new
/// `FbpGrid` is the way to start over from different inputs.
#[derive(Debug, Clone)]
pub struct FbpGrid {
    n: usize,
    /// Fuel codes as given (invalid codes and masked cells as given, e.g. 0).
    fuel: Vec<i32>,
    /// One slot per field, indexed by [`Field`] position.
    slots: Vec<Slot>,
    wx_date: i64,
    percentile_growth: f64,
    d0: Option<OwnedPerCell>,
    dj: Option<DjOverride>,
    /// `out_request` (each field once).
    out_request: Vec<Field>,
    /// `out_request` and the `keep` list of [`FbpGrid::plan`].
    retained: [bool; Field::COUNT],
    /// The planned stages, in canonical order.
    planned: Vec<Stage>,
    liveness: Liveness,
    phase: Phase,
    /// Version counters of the `calcFMC` inputs, indexed by [`FmcInput`].
    fmc_versions: [u64; 6],
    /// The counters `calcFMC` last computed with, while its outputs are
    /// unmodified.
    fmc_record: Option<[u64; 6]>,
    fmc_computations: u64,
}

impl FbpGrid {
    /// A grid over the cells of `input`, storing `out_request` (any fields,
    /// each kept for reading after the run).
    ///
    /// Every input is copied and normalised as [`fbp::run`](crate::fbp::run)
    /// does (negative `slope`, `ws`, `ffmc`, `bui`, `pc`, `pdf`, `gfl` clamp to
    /// 0, a negative aspect becomes 270, a zero `gcf` becomes 0.1, `long` is
    /// stored as its magnitude); the normalisation is idempotent. The `d0` /
    /// `dj` overrides of `input` are stored as with
    /// [`FbpGrid::set_override`]. The plan is every stage.
    ///
    /// # Errors
    ///
    /// The checks of [`run_grid_fields`], in
    /// its order: [`GridError::InvalidDate`], [`GridError::LengthMismatch`],
    /// [`GridError::InfiniteDj`].
    pub fn new(input: &GridInput<'_>, out_request: &[Field]) -> Result<Self, GridError> {
        let (resolved, clipped) = input.validate()?;
        let mut slots = vec![Slot::Never; Field::COUNT];
        for (f, p) in [
            (Field::Lat, input.lat),
            (Field::Long, input.long),
            (Field::Elevation, input.elevation),
            (Field::Slope, input.slope_pct),
            (Field::AspectIn, input.aspect_deg),
            (Field::Pc, input.pct_conifer),
            (Field::Pdf, input.pct_dead_fir),
            (Field::Gfl, input.grass_fuel_load),
            (Field::Gcf, input.grass_curing),
            (Field::Ws, input.ws),
            (Field::WdIn, input.wd),
            (Field::Ffmc, input.ffmc),
            (Field::Bui, input.bui),
        ] {
            slots[f.index()] = Slot::Input(normalized(f, p));
        }
        let dj = match (input.dj, resolved) {
            (Some(supplied), Some(resolved)) => Some(DjOverride {
                supplied: supplied.into(),
                resolved,
                clipped,
            }),
            _ => None,
        };
        let mut unique: Vec<Field> = Vec::with_capacity(out_request.len());
        for &f in out_request {
            if !unique.contains(&f) {
                unique.push(f);
            }
        }
        let mut grid = Self {
            n: input.fuel_type.len(),
            fuel: input.fuel_type.to_vec(),
            slots,
            wx_date: input.wx_date,
            percentile_growth: input.percentile_growth,
            d0: input.d0.map(OwnedPerCell::from),
            dj,
            out_request: unique,
            retained: [false; Field::COUNT],
            planned: Stage::ALL.to_vec(),
            liveness: Liveness::new(&Stage::ALL),
            phase: Phase::Fresh,
            fmc_versions: [0; 6],
            fmc_record: None,
            fmc_computations: 0,
        };
        grid.plan(&Stage::ALL, &[]);
        Ok(grid)
    }

    /// Set the planned stages (the order of `stages` does not matter;
    /// default [`Stage::ALL`]) and the fields to keep besides `out_request`.
    ///
    /// A computed field is dropped once every planned stage that reads it has
    /// run, unless it is kept. A planned stage that is skipped keeps its
    /// inputs stored; a caller that skips a stage on purpose (for example
    /// `calcFMC` when injecting `fmc`) passes the exact list so those inputs
    /// can be dropped. Stages that already ran stay counted as run.
    pub fn plan(&mut self, stages: &[Stage], keep: &[Field]) {
        self.planned = Stage::ALL
            .into_iter()
            .filter(|s| stages.contains(s))
            .collect();
        self.liveness.replan(&self.planned);
        self.retained = [false; Field::COUNT];
        for f in self.out_request.iter().chain(keep) {
            self.retained[f.index()] = true;
        }
    }

    /// Run one stage over every cell: load the fields it reads (stored
    /// values, inputs, or the template of a field never computed), apply it,
    /// and store the fields it writes that are still needed. Then drop the
    /// fields whose planned readers have all run. A stage outside the plan
    /// may run while its inputs are stored.
    ///
    /// `calcFMC` reuses its previous result when none of its inputs (`lat`,
    /// `long`, `elevation`, the `d0` / `dj` overrides and `wx_date`) changed
    /// and its outputs are still stored and were not set since.
    ///
    /// `threads` works as in [`run_grid_with_threads`](crate::grid::run_grid_with_threads)
    /// (1 = calling thread, 0 = every core, grids under 4,096 cells serial);
    /// the result does not depend on it.
    ///
    /// # Errors
    ///
    /// [`StageError::AfterFused`] after [`FbpGrid::run_fused`] (call
    /// [`FbpGrid::reset`] first); [`StageError::Freed`] for the first field
    /// the stage reads (in [`Stage::reads`] order) that was dropped. Nothing
    /// changes on an error.
    pub fn run_stage(&mut self, stage: Stage, threads: usize) -> Result<(), StageError> {
        self.run_stage_with_fuel(stage, threads, None)
    }

    fn run_stage_with_fuel(
        &mut self,
        stage: Stage,
        threads: usize,
        fuel: Option<&[i32]>,
    ) -> Result<(), StageError> {
        if self.phase == Phase::Fused {
            return Err(StageError::AfterFused);
        }
        let mut after = self.liveness.clone();
        let last_readers = after.record_run(stage);
        let keep: Vec<Field> = stage
            .writes()
            .iter()
            .copied()
            .filter(|f| self.retained[f.index()] || after.needed(*f))
            .collect();
        let cached = stage == Stage::CalcFmc
            && self.fmc_record == Some(self.fmc_versions)
            && keep
                .iter()
                .all(|f| matches!(self.slots[f.index()], Slot::Stored(_)));
        if cached {
            for f in stage.writes() {
                if !keep.contains(f) {
                    self.slots[f.index()] = Slot::Freed;
                }
            }
        } else {
            let values = if let Some(codes) = fuel {
                self.compute_with_fuel(stage, &keep, threads, codes)?
            } else {
                self.compute(stage, &keep, threads)?
            };
            for f in stage.writes() {
                self.slots[f.index()] = Slot::Freed;
            }
            for (f, v) in keep.iter().zip(values) {
                self.slots[f.index()] = Slot::Stored(OwnedPerCell::Cells(v));
            }
            if stage == Stage::CalcFmc {
                self.fmc_record = Some(self.fmc_versions);
                self.fmc_computations += 1;
            }
        }
        for f in last_readers {
            if !self.retained[f.index()] && matches!(self.slots[f.index()], Slot::Stored(_)) {
                self.slots[f.index()] = Slot::Freed;
            }
        }
        self.liveness = after;
        self.phase = Phase::Stepping;
        Ok(())
    }

    /// Every planned stage, in canonical ([`Stage::ALL`]) order, with
    /// [`FbpGrid::run_stage`].
    ///
    /// # Errors
    ///
    /// The first error of [`FbpGrid::run_stage`]; the stages before it have
    /// run.
    pub fn run_planned(&mut self, threads: usize) -> Result<(), StageError> {
        for stage in self.planned.clone() {
            self.run_stage(stage, threads)?;
        }
        Ok(())
    }

    /// The fused pass: discard every computed and injected value (in every
    /// state), then compute every cell with the fused per-cell pass, as
    /// applying [`Stage::ALL`] would, and store only `out_request`. Fields
    /// outside `out_request` read as freed afterwards.
    ///
    /// # Errors
    ///
    /// The errors of [`run_grid_fields`]; the
    /// stored inputs were validated already, so none is expected.
    pub fn run_fused(&mut self, threads: usize) -> Result<(), GridError> {
        self.discard_computed();
        let fields: Vec<Field> = self
            .out_request
            .iter()
            .copied()
            .filter(|f| is_computed(*f))
            .collect();
        let out = run_grid_fields(
            &self.grid_input(),
            &fields,
            threads,
            NonModeled::PythonCompat,
        )?;
        for f in Field::ALL {
            if is_computed(f) {
                self.slots[f.index()] = Slot::Freed;
            }
        }
        for (f, v) in out.fields {
            self.slots[f.index()] = Slot::Stored(OwnedPerCell::Cells(v));
        }
        self.phase = Phase::Fused;
        Ok(())
    }

    /// Store `values` (one per cell) in `field`, as `setParams` does.
    ///
    /// On a computed field the value is stored (un-freeing it) until a stage
    /// writes it, [`FbpGrid::reset`] or [`FbpGrid::run_fused`] discards it;
    /// `set(Hros, ..)` also resets the `hros_growth_nan` provenance to 0,
    /// and setting a `calcFMC` output makes the next `calcFMC` recompute. On
    /// an input (`lat long elevation slope pc pdf gfl gcf ws ffmc bui`) it
    /// replaces the stored input, normalised like [`FbpGrid::new`], and
    /// persists across `reset` and `run_fused`. `wd` / `wd_in` and `aspect` /
    /// `aspect_in` replace the wind direction / aspect input and discard the
    /// inverted pair (so both read as the inputs until `invertWindAspect`
    /// runs again).
    ///
    /// # Errors
    ///
    /// [`StageError::LengthMismatch`] unless there is one value per cell;
    /// [`StageError::NotSettable`] for `fuel_type`, `d0_override` and
    /// `dj_override`. Nothing changes on an error.
    pub fn set(&mut self, field: Field, values: &[f64]) -> Result<(), StageError> {
        if values.len() != self.n {
            return Err(StageError::LengthMismatch {
                field,
                expected: self.n,
                found: values.len(),
            });
        }
        self.store(field, OwnedPerCell::Cells(values.to_vec()))
    }

    /// [`FbpGrid::set`] with one value for every cell (held as one value).
    ///
    /// # Errors
    ///
    /// [`StageError::NotSettable`] as for [`FbpGrid::set`].
    pub fn set_const(&mut self, field: Field, value: f64) -> Result<(), StageError> {
        self.store(field, OwnedPerCell::Const(value))
    }

    /// Replace the fuel codes (one per cell; masked and invalid cells as an
    /// invalid code such as 0). The codes persist across [`FbpGrid::reset`]
    /// and [`FbpGrid::run_fused`].
    ///
    /// # Errors
    ///
    /// [`GridError::LengthMismatch`] (name `fuel_type`) unless there is one
    /// code per cell.
    pub fn set_fuel(&mut self, codes: &[i32]) -> Result<(), GridError> {
        if codes.len() != self.n {
            return Err(GridError::LengthMismatch {
                name: "fuel_type",
                expected: self.n,
                found: codes.len(),
            });
        }
        self.fuel = codes.to_vec();
        // the Python values of non-modeled cells depend on the fuel
        self.fmc_record = None;
        Ok(())
    }

    /// Change a scalar stage parameter. `wx_date` is a `YYYYMMDD` date (an
    /// integral value); a stored `dj` override is resolved again for the new
    /// year (the [`FbpGrid::dj_override_clipped`] count follows).
    /// `percentile_growth` is used by the next growth stage as given.
    ///
    /// # Errors
    ///
    /// [`GridError::InvalidDate`] for a `wx_date` that is not a real calendar
    /// date (or not an integer); nothing changes then.
    pub fn set_param(&mut self, param: ParamName, value: f64) -> Result<(), GridError> {
        match param {
            ParamName::WxDate => {
                // saturating; a non-integral value is rejected below
                #[allow(clippy::cast_possible_truncation)]
                let date = value as i64;
                if !value.is_finite() || value.fract() != 0.0 || !is_valid_wx_date(date) {
                    return Err(GridError::InvalidDate(date));
                }
                if let Some(dj) = &mut self.dj {
                    let (resolved, clipped) = resolve_dj(dj.supplied.as_per_cell(), date)?;
                    dj.resolved = resolved;
                    dj.clipped = clipped;
                }
                self.wx_date = date;
                self.bump(FmcInput::WxDate);
            }
            ParamName::PercentileGrowth => self.percentile_growth = value,
        }
        Ok(())
    }

    /// Store an explicit `d0` ([`Field::D0Override`]) or `dj`
    /// ([`Field::DjOverride`]) for `calcFMC`, a constant or one value per
    /// cell (NaN = masked: that cell's `fmc` and `fme` are NaN), or clear it
    /// with `None` (back to the derived value). `d0` is used as given (no
    /// range limit). `dj` is kept as supplied and used floored, then clipped
    /// to 1..365 (1..366 in a leap year) of `wx_date`, re-resolved whenever
    /// `wx_date` changes; see [`FbpGrid::dj_override_clipped`]. Overrides
    /// persist across stages, `reset` and `run_fused` until replaced or
    /// cleared.
    ///
    /// # Errors
    ///
    /// [`GridError::NotAnOverride`] for any other field,
    /// [`GridError::LengthMismatch`] (name `d0` / `dj`) for a per-cell
    /// override of the wrong length, [`GridError::InfiniteDj`] for an
    /// infinite `dj`. Nothing changes on an error.
    pub fn set_override(
        &mut self,
        field: Field,
        values: Option<PerCell<'_>>,
    ) -> Result<(), GridError> {
        let name = match field {
            Field::D0Override => "d0",
            Field::DjOverride => "dj",
            _ => return Err(GridError::NotAnOverride(field)),
        };
        if let Some(PerCell::Cells(v)) = values {
            if v.len() != self.n {
                return Err(GridError::LengthMismatch {
                    name,
                    expected: self.n,
                    found: v.len(),
                });
            }
        }
        if field == Field::D0Override {
            self.d0 = values.map(OwnedPerCell::from);
            self.bump(FmcInput::D0);
        } else {
            self.dj = match values {
                None => None,
                Some(supplied) => {
                    let (resolved, clipped) = resolve_dj(supplied, self.wx_date)?;
                    Some(DjOverride {
                        supplied: supplied.into(),
                        resolved,
                        clipped,
                    })
                }
            };
            self.bump(FmcInput::Dj);
        }
        Ok(())
    }

    /// Drop every computed and injected value (the inverted `wd` / `aspect`
    /// too, so they read as the inputs again) and restart the liveness
    /// bookkeeping from the plan, so the grid frees fields exactly like a new
    /// one. Keeps the inputs (including [`FbpGrid::set`] on inputs), the fuel
    /// codes, the scalar parameters, the overrides, `out_request` and the
    /// plan.
    pub fn reset(&mut self) {
        self.discard_computed();
        self.phase = Phase::Fresh;
    }

    /// The values of `field`, one per cell: a stored or input value, or the
    /// template ([`Field::template_value`]) of a field never computed (`wd`
    /// and `aspect` read as their inputs until `invertWindAspect` runs).
    /// `fuel_type` is the stored code, 0 for an invalid code (as the Python
    /// package exports it); `d0_override` is the supplied and `dj_override`
    /// the resolved override (NaN without one).
    ///
    /// # Errors
    ///
    /// [`StageError::FreedRead`] for a field that was dropped; add it to
    /// `out_request` (or the `keep` list of [`FbpGrid::plan`]) to read it.
    pub fn get(&self, field: Field) -> Result<Vec<f64>, StageError> {
        if field == Field::FuelType {
            return Ok(self
                .fuel
                .iter()
                .map(|&code| {
                    let ft = FuelType::from_code(code);
                    let v = f64::from(code);
                    if ft.is_modeled() {
                        v
                    } else {
                        python_non_modeled(field, ft, &FbpInput::new(code, self.wx_date), v)
                    }
                })
                .collect());
        }
        match self.view(field) {
            Some(PerCell::Const(v)) => Ok(vec![v; self.n]),
            Some(PerCell::Cells(v)) => Ok(v.to_vec()),
            None => Err(StageError::FreedRead(field)),
        }
    }

    /// Bytes held by the stored arrays: the inputs, the fuel codes, the
    /// overrides and the computed fields still stored (a constant counts as
    /// one value).
    #[must_use]
    pub fn live_bytes(&self) -> usize {
        let slots: usize = self
            .slots
            .iter()
            .map(|s| match s {
                Slot::Input(v) | Slot::Stored(v) => v.bytes(),
                Slot::Never | Slot::Freed => 0,
            })
            .sum();
        let overrides = self.d0.as_ref().map_or(0, OwnedPerCell::bytes)
            + self
                .dj
                .as_ref()
                .map_or(0, |d| d.supplied.bytes() + d.resolved.bytes());
        slots + overrides + std::mem::size_of_val(self.fuel.as_slice())
    }

    /// The number of cells.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the grid has no cells.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// How many supplied `dj` override values the resolution for the current
    /// `wx_date` changed by clipping (below 1, or above 365/366; flooring a
    /// fraction and NaN do not count; a constant is one value); 0 without a
    /// `dj` override. Callers use it to announce clipping.
    #[must_use]
    pub fn dj_override_clipped(&self) -> usize {
        self.dj.as_ref().map_or(0, |d| d.clipped)
    }

    /// How many times `calcFMC` computed rather than reused its previous
    /// result (a test hook for the exact-input cache).
    #[doc(hidden)]
    #[must_use]
    pub fn fmc_computations(&self) -> u64 {
        self.fmc_computations
    }

    // ------------------------------------------------------------ internals

    /// The current values of `field`, or `None` when it was dropped.
    /// `fuel_type` is not held here (see [`FbpGrid::get`]).
    fn view(&self, field: Field) -> Option<PerCell<'_>> {
        match field {
            Field::D0Override => Some(
                self.d0
                    .as_ref()
                    .map_or(PerCell::Const(f64::NAN), OwnedPerCell::as_per_cell),
            ),
            Field::DjOverride => Some(
                self.dj
                    .as_ref()
                    .map_or(PerCell::Const(f64::NAN), |d| d.resolved.as_per_cell()),
            ),
            _ => match &self.slots[field.index()] {
                Slot::Input(v) | Slot::Stored(v) => Some(v.as_per_cell()),
                Slot::Never => Some(match field {
                    // the working pair reads as the inputs until inverted
                    Field::Wd => self.input(Field::WdIn),
                    Field::Aspect => self.input(Field::AspectIn),
                    _ => PerCell::Const(field.template_value()),
                }),
                Slot::Freed => None,
            },
        }
    }

    /// A stored input.
    fn input(&self, field: Field) -> PerCell<'_> {
        match &self.slots[field.index()] {
            Slot::Input(v) => v.as_per_cell(),
            _ => PerCell::Const(f64::NAN),
        }
    }

    /// The stored inputs as a [`GridInput`] for the fused pass.
    fn grid_input(&self) -> GridInput<'_> {
        GridInput {
            fuel_type: &self.fuel,
            lat: self.input(Field::Lat),
            long: self.input(Field::Long),
            elevation: self.input(Field::Elevation),
            slope_pct: self.input(Field::Slope),
            aspect_deg: self.input(Field::AspectIn),
            pct_conifer: self.input(Field::Pc),
            pct_dead_fir: self.input(Field::Pdf),
            grass_fuel_load: self.input(Field::Gfl),
            grass_curing: self.input(Field::Gcf),
            ws: self.input(Field::Ws),
            wd: self.input(Field::WdIn),
            ffmc: self.input(Field::Ffmc),
            bui: self.input(Field::Bui),
            wx_date: self.wx_date,
            percentile_growth: self.percentile_growth,
            d0: self.d0.as_ref().map(OwnedPerCell::as_per_cell),
            dj: self.dj.as_ref().map(|d| d.supplied.as_per_cell()),
        }
    }

    /// Runs `stage` over every cell and returns the values of `keep` (in its
    /// order). Checks every read first.
    fn compute(
        &self,
        stage: Stage,
        keep: &[Field],
        threads: usize,
    ) -> Result<Vec<Vec<f64>>, StageError> {
        self.compute_with_fuel(stage, keep, threads, &self.fuel)
    }

    /// Evaluate acceleration with wrapper-retained fuel codes and the current
    /// CFB/template/injection, without changing stored fuel. Normal stage liveness and retention apply.
    /// This preserves the original modeled code underneath a Python fuel mask.
    ///
    /// # Errors
    ///
    /// Returns [`StageError::LengthMismatch`] for a different fuel count, or
    /// the lifecycle/read errors of [`Self::run_stage`]. No state changes on error.
    pub fn run_acceleration_for_fuel(
        &mut self,
        codes: &[i32],
        threads: usize,
    ) -> Result<(), StageError> {
        if self.phase == Phase::Fused {
            return Err(StageError::AfterFused);
        }
        if codes.len() != self.n {
            return Err(StageError::LengthMismatch {
                field: Field::FuelType,
                expected: self.n,
                found: codes.len(),
            });
        }
        self.run_stage_with_fuel(Stage::CalcAccelParam, threads, Some(codes))
    }

    fn compute_with_fuel(
        &self,
        stage: Stage,
        keep: &[Field],
        threads: usize,
        fuel: &[i32],
    ) -> Result<Vec<Vec<f64>>, StageError> {
        let mut sources: Vec<(Field, PerCell<'_>)> = Vec::with_capacity(stage.reads().len());
        for &f in stage.reads() {
            // the fuel code is loaded per cell below
            if f != Field::FuelType {
                let p = self.view(f).ok_or(StageError::Freed { field: f, stage })?;
                sources.push((f, p));
            }
        }
        let mut out = vec![vec![0.0; self.n]; keep.len()];
        if keep.is_empty() {
            return Ok(out);
        }
        let params = StageParams {
            wx_date: self.wx_date,
            percentile_growth: self.percentile_growth,
            // the has-override flags; the values are loaded per cell
            d0: self.d0.as_ref().map(|_| f64::NAN),
            dj: self.dj.as_ref().map(|_| f64::NAN),
        };
        let (ffmc, slope, bui) = (
            self.input(Field::Ffmc),
            self.input(Field::Slope),
            self.input(Field::Bui),
        );
        let dj = self.dj.as_ref().map(|d| d.resolved.as_per_cell());
        let wx_date = self.wx_date;
        fill_chunks(self.n, threads, &mut out, |start, len, part| {
            // A stage reads only its declared fields, so one state serves the
            // whole chunk once those are loaded for each cell.
            let mut s = CellState::template(0, params);
            for j in 0..len {
                let i = start + j;
                let code = fuel[i];
                s.set(Field::FuelType, f64::from(code));
                for (f, p) in &sources {
                    s.set(*f, p.at(i));
                }
                let ft = FuelType::from_code(code);
                // Invalid/masked fuels are absent from the Python canopy
                // lookup loop. Preserve the consumed state at this boundary;
                // the legacy scalar stage composition remains unchanged.
                if stage != Stage::GetCbhCfl || !matches!(ft, FuelType::Unknown(_)) {
                    s.apply(stage);
                }
                // A masked activation condition masks CROS in the Python
                // stage, independently of finite ISI/FME or current weather.
                if stage == Stage::CalcC6Cros && code == 6 && s.get(Field::C6BlendCfc).is_nan() {
                    s.set(Field::Cros, f64::NAN);
                }
                // Arithmetic stages consume actual inputs, including injections.
                if ft.is_modeled()
                    || matches!(
                        stage,
                        Stage::CalcTfc
                            | Stage::CalcRso
                            | Stage::GetCbhCfl
                            | Stage::CalcRosPercentileGrowth
                            | Stage::CalcHfi
                            | Stage::CalcFireIntensityClass
                            | Stage::CalcAccelParam
                    )
                    || (matches!(ft, FuelType::NonFuel | FuelType::Water)
                        && matches!(
                            stage,
                            Stage::CalcRos | Stage::CalcC6Cros | Stage::CalcC6Hros | Stage::CalcCfc
                        ))
                {
                    for (o, f) in part.iter_mut().zip(keep) {
                        o[j] = s.get(*f);
                    }
                } else {
                    let mut cell = FbpInput::new(code, wx_date);
                    cell.ffmc = ffmc.at(i);
                    cell.slope_pct = slope.at(i);
                    cell.bui = bui.at(i);
                    cell.dj_override = dj.map(|d| d.at(i));
                    for (o, f) in part.iter_mut().zip(keep) {
                        o[j] = python_non_modeled(*f, ft, &cell, s.get(*f));
                    }
                }
            }
        });
        Ok(out)
    }

    /// The routes of [`FbpGrid::set`] and [`FbpGrid::set_const`].
    fn store(&mut self, field: Field, values: OwnedPerCell) -> Result<(), StageError> {
        match field {
            Field::FuelType | Field::D0Override | Field::DjOverride => {
                return Err(StageError::NotSettable(field));
            }
            Field::Wd | Field::WdIn | Field::Aspect | Field::AspectIn => {
                let input = if matches!(field, Field::Wd | Field::WdIn) {
                    Field::WdIn
                } else {
                    Field::AspectIn
                };
                self.set_input(input, &values);
                // the inverted pair is derived again from the inputs
                self.slots[Field::Wd.index()] = Slot::Never;
                self.slots[Field::Aspect.index()] = Slot::Never;
            }
            f if INPUTS.contains(&f) => {
                self.set_input(f, &values);
                match f {
                    Field::Lat => self.bump(FmcInput::Lat),
                    Field::Long => self.bump(FmcInput::Long),
                    Field::Elevation => self.bump(FmcInput::Elevation),
                    _ => {}
                }
            }
            f => {
                if f == Field::Hros {
                    // an injected head ROS has no growth provenance
                    self.slots[Field::HrosGrowthNan.index()] =
                        Slot::Stored(OwnedPerCell::Const(0.0));
                }
                if FMC_OUTPUTS.contains(&f) {
                    self.fmc_record = None;
                }
                if f == Field::Fmc {
                    // a provided fmc also gives the matching foliar moisture effect;
                    // a later set of fme itself replaces it
                    self.slots[Field::Fme.index()] = Slot::Stored(values.map(fme_from_fmc));
                }
                self.slots[f.index()] = Slot::Stored(values);
            }
        }
        Ok(())
    }

    /// Replace a stored input, normalised.
    fn set_input(&mut self, field: Field, values: &OwnedPerCell) {
        self.slots[field.index()] = Slot::Input(values.map(|v| normalize_input(field, v)));
    }

    /// A `calcFMC` input changed.
    fn bump(&mut self, input: FmcInput) {
        self.fmc_versions[input as usize] += 1;
    }

    /// Every computed field back to its template, the liveness bookkeeping
    /// restarted and the `calcFMC` record dropped.
    fn discard_computed(&mut self) {
        for f in Field::ALL {
            if is_computed(f) {
                self.slots[f.index()] = Slot::Never;
            }
        }
        self.liveness.restart();
        self.fmc_record = None;
    }
}

/// Whether `field` is computed by a stage (everything except the stored
/// inputs, the fuel code and the overrides).
fn is_computed(field: Field) -> bool {
    !INPUTS.contains(&field)
        && !matches!(
            field,
            Field::FuelType | Field::D0Override | Field::DjOverride
        )
}

/// `p` copied and normalised as input `field`.
fn normalized(field: Field, p: PerCell<'_>) -> OwnedPerCell {
    match p {
        PerCell::Const(v) => OwnedPerCell::Const(normalize_input(field, v)),
        PerCell::Cells(values) => {
            OwnedPerCell::Cells(values.iter().map(|&v| normalize_input(field, v)).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stored_inputs_are_the_non_computed_fields() {
        let not_computed: Vec<Field> = Field::ALL
            .into_iter()
            .filter(|f| !is_computed(*f))
            .collect();
        assert_eq!(not_computed.len(), INPUTS.len() + 3);
        for f in INPUTS {
            assert!(
                !f.is_output() || matches!(f, Field::Ws | Field::Ffmc | Field::Bui),
                "{f:?}"
            );
        }
        // every stage writes only computed fields
        for s in Stage::ALL {
            for f in s.writes() {
                assert!(is_computed(*f), "{s:?} writes {f:?}");
            }
        }
    }
}
