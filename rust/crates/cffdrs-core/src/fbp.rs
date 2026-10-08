//! The CFFBPS scalar core: one cell in, all 54 outputs out.
//!
//! Mirrors `src/cffdrs/cffbps/` (the reference implementation) function by
//! function: `inputs._verify_inputs` normalization, then the facade's
//! `runFBP()` chain in the same order, with the same fuel-code conventions
//! (1..18 modeled, 19 non-fuel, 20 water) and the same NaN semantics (masked cells
//! surface as NaN).
//!
//! Every quantity in [`FbpResult`] is asserted against the shared golden
//! snapshot (`tests/cffbps/data/golden/wotton2009_scalar_snapshot.json`) —
//! the same file the Python suite validates against.

use crate::consumption::{calc_fire_intensity_class, calc_hfi, calc_tfc};
use crate::crown::{
    calc_cfc, calc_csfi, calc_fire_type, calc_rso, cfb_from_ros, directional_cfb, final_cfb,
};
pub use crate::fmc::is_valid_wx_date;
use crate::fmc::{calc_fmc, injected_fmc};
use crate::fuel::{CrownFuel, FuelType, RosParams};
use crate::growth::calc_accel_param;
use crate::normalize::{invert_wind_aspect, normalize, Normalized};
use crate::percentile::{percentile_ros, percentile_tinv, wind_decay};
use crate::ros::{calc_c6_blend, calc_ros, Ros};
use crate::slope_wind::{calc_isi_rsi_be, calc_isz, calc_sf, Isz, SpreadIndices};
use crate::surface::{calc_sfc, SurfaceFuel};

/// Scalar inputs for one cell, matching `FBP.initialize` in the Python
/// package. NaN in any numeric field means a missing/masked value and
/// propagates to the dependent outputs.
#[derive(Debug, Clone, PartialEq)]
pub struct FbpInput {
    /// CFFBPS numeric fuel code (1..18 modeled; 19 non-fuel; 20 water).
    pub fuel_type: i32,
    /// Date as YYYYMMDD (drives day-of-year and foliar moisture).
    pub wx_date: i64,
    /// Latitude, decimal degrees (north positive); drives foliar moisture.
    pub lat: f64,
    /// Longitude, decimal degrees (west negative); only its magnitude is used.
    pub long: f64,
    /// Elevation above sea level, metres; drives foliar moisture.
    pub elevation: f64,
    /// Ground slope in percent.
    pub slope_pct: f64,
    /// Aspect (downhill direction), compass degrees.
    pub aspect_deg: f64,
    /// 10-m open wind speed, km/h.
    pub ws: f64,
    /// Wind direction, compass degrees.
    pub wd: f64,
    /// Fine Fuel Moisture Code (dimensionless, 0-101).
    pub ffmc: f64,
    /// Buildup Index (dimensionless, >= 0).
    pub bui: f64,
    /// Percent conifer (M-1/M-2).
    pub pc: f64,
    /// Percent dead fir (M-3/M-4).
    pub pdf: f64,
    /// Grass fuel load, kg/m^2 (O-1a/b).
    pub gfl: f64,
    /// Grass curing factor, percent (O-1a/b).
    pub gcf: f64,
    /// Percentile growth: a percentile (0-100) of the ROS distribution, not a
    /// percent change; 50 is the exact no-op. Values outside (0.001, 99.999)
    /// are capped; NaN propagates. See `percentile` for the model.
    pub percentile_growth: f64,
    /// Optional caller-supplied Julian date of minimum foliar moisture
    /// (`initialize(d0=...)`); derived from latitude/elevation when None.
    pub d0_override: Option<f64>,
    /// Optional caller-supplied Julian date (`initialize(dj=...)`); derived
    /// from `wx_date` when None.
    pub dj_override: Option<f64>,
    /// Injected foliar moisture (`setParams({'fmc': ...})` in place of
    /// calcFMC). When Some, calcFMC is skipped entirely: latn/d0/dj/nd keep
    /// their zero template values and — critically — so does fme, which
    /// zeroes the C-6 crown ROS. This mirrors the recompute sequence
    /// fire-growth engines run to regenerate output quantities from stamped
    /// per-cell weather.
    pub fmc_override: Option<f64>,
    /// Injected head ROS (`setParams({'hros': ...})` after calcROS): replaces
    /// hros before calcCSFI onward; bros and sros keep their computed values,
    /// so the C-6 blend (which reads sros) can overwrite the injected value.
    pub hros_override: Option<f64>,
}

impl FbpInput {
    /// A cell with the Python package's `initialize` defaults: `pc` 50, `pdf` 35,
    /// `gfl` 0.35, `gcf` 80, `percentile_growth` 50, no overrides. The environment
    /// inputs (`lat`, `long`, `elevation`, `slope_pct`, `aspect_deg`, `ws`, `wd`,
    /// `ffmc`, `bui`) start as NaN, i.e. missing, and must be set by the caller.
    ///
    /// Note that `PartialEq` follows `f64`: an input holding NaN is not equal to
    /// itself.
    ///
    /// ```
    /// use cffdrs_core::fbp::{run, FbpInput};
    ///
    /// let mut input = FbpInput::new(2, 20230615);
    /// input.lat = 55.0;
    /// input.long = -115.0;
    /// input.elevation = 500.0;
    /// input.ws = 20.0;
    /// input.wd = 270.0;
    /// input.ffmc = 90.0;
    /// input.bui = 60.0;
    /// input.slope_pct = 0.0;
    /// input.aspect_deg = 0.0;
    /// assert!(run(&input).hros > 0.0);
    /// ```
    #[must_use]
    pub fn new(fuel_type: i32, wx_date: i64) -> Self {
        Self {
            fuel_type,
            wx_date,
            lat: f64::NAN,
            long: f64::NAN,
            elevation: f64::NAN,
            slope_pct: f64::NAN,
            aspect_deg: f64::NAN,
            ws: f64::NAN,
            wd: f64::NAN,
            ffmc: f64::NAN,
            bui: f64::NAN,
            pc: 50.0,
            pdf: 35.0,
            gfl: 0.35,
            gcf: 80.0,
            percentile_growth: 50.0,
            d0_override: None,
            dj_override: None,
            fmc_override: None,
            hros_override: None,
        }
    }
}

/// Everything the scalar pass computes: the snapshot's 54 quantities.
/// All f64, including code-like values (`fire_type`, `fi_class`), so golden
/// comparison is uniform; consumers cast as needed.
///
/// Field names follow the Python package's `FBP.getParams` names, except that
/// `fF` and `fW` are spelled `f_f` and `f_w` (see [`FbpResult::named_values`]
/// for the exact name table). A NaN field means the quantity is missing or
/// masked for that cell (for example a non-fuel cell, or a NaN input that
/// propagated); it is not an error.
///
/// The struct is `#[non_exhaustive]`: read its fields, or start from
/// `FbpResult::default()`, rather than building or destructuring it
/// exhaustively.
///
/// Note that `PartialEq` follows `f64`: a result holding NaN is not equal to
/// itself.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct FbpResult {
    // wind/slope vectoring
    /// Observed wind speed as used by the chain (after input normalisation), km/h.
    pub ws: f64,
    /// Wind direction as used by the slope/wind step: the supplied direction inverted by 180 degrees, compass degrees.
    pub wd: f64,
    /// Slope-equivalent wind speed (the wind speed whose effect equals the slope effect), km/h.
    pub wse: f64,
    /// Original slope-equivalent wind speed, used where WSE1 <= 40, km/h.
    pub wse1: f64,
    /// Revised slope-equivalent wind speed, used where WSE1 > 40, km/h.
    pub wse2: f64,
    /// Net vectorized wind speed in the x (east) direction, km/h.
    pub wsx: f64,
    /// Net vectorized wind speed in the y (north) direction, km/h.
    pub wsy: f64,
    /// Net vectorized (wind plus slope) wind speed, km/h.
    pub wsv: f64,
    /// Net spread direction (rate-of-spread azimuth), compass degrees.
    pub raz: f64,
    // moisture / ISI chain
    /// Fine fuel moisture content, percent.
    pub m: f64,
    /// Fine fuel moisture function `fF` (key `"fF"`).
    pub f_f: f64,
    /// Wind function `fW` (key `"fW"`).
    pub f_w: f64,
    /// Fine Fuel Moisture Code as used.
    pub ffmc: f64,
    /// Final ISI, accounting for wind and slope.
    pub isi: f64,
    /// Buildup Index as used.
    pub bui: f64,
    // fuel-type ROS parameterization
    /// Surface ROS parameter `a`.
    pub a: f64,
    /// Surface ROS parameter `b`.
    pub b: f64,
    /// Surface ROS parameter `c`.
    pub c: f64,
    /// Proportion of maximum rate of spread at BUI equal to 50 (`q`).
    pub q: f64,
    /// Average BUI for the fuel type, `BUI0`.
    pub bui0: f64,
    /// Buildup effect on spread rate.
    pub be: f64,
    /// Maximum buildup effect, `BE_max`.
    pub be_max: f64,
    // slope-adjusted spread chain
    /// Slope factor.
    pub sf: f64,
    /// Surface ROS at zero wind on flat ground, m/min.
    pub rsz: f64,
    /// Surface ROS with the slope effect at zero wind, m/min.
    pub rsf: f64,
    /// ISI with the slope effect at zero wind.
    pub isf: f64,
    /// Initial spread rate without buildup effect, m/min.
    pub rsi: f64,
    // foliar moisture
    /// Normalised latitude used for the foliar moisture date calculation.
    pub latn: f64,
    /// Julian date of the fire (day of year).
    pub dj: f64,
    /// Julian date of minimum foliar moisture content.
    pub d0: f64,
    /// Days from the date of minimum foliar moisture.
    pub nd: f64,
    /// Foliar moisture content, percent.
    pub fmc: f64,
    /// Foliar moisture effect (crown ROS term).
    pub fme: f64,
    // consumption
    /// Forest-floor fuel consumption, kg/m^2.
    pub ffc: f64,
    /// Woody fuel consumption, kg/m^2.
    pub wfc: f64,
    /// Surface fuel consumption, kg/m^2.
    pub sfc: f64,
    /// Crown fuel load, kg/m^2.
    pub cfl: f64,
    /// Crown fuel consumption, kg/m^2.
    pub cfc: f64,
    /// Total fuel consumption, kg/m^2.
    pub tfc: f64,
    /// Crown base height, m.
    pub cbh: f64,
    // crowning
    /// Critical surface fire intensity, kW/m.
    pub csfi: f64,
    /// Critical surface ROS for crowning, m/min.
    pub rso: f64,
    /// Crown fraction burned (0-1).
    pub cfb: f64,
    /// Fire type code as a float: 1 surface, 2 intermittent crown, 3 active crown, 0 non-fuel or masked.
    pub fire_type: f64,
    // spread rates
    /// Head fire rate of spread, m/min.
    pub hros: f64,
    /// Surface head fire rate of spread, m/min.
    pub sros: f64,
    /// Active crown fire rate of spread, m/min.
    pub cros: f64,
    /// Backing fire wind function.
    pub bfw: f64,
    /// Backing fire ISI.
    pub bisi: f64,
    /// Backing fire rate of spread, m/min.
    pub bros: f64,
    // intensity / class / growth
    /// Head fire intensity, kW/m.
    pub hfi: f64,
    /// Fire intensity class, 1-6 by HFI thresholds; -99 when HFI <= 0 or masked.
    pub fi_class: f64,
    /// Acceleration parameter for fire growth.
    pub accel: f64,
    /// Fuel type code, as a float.
    pub fuel_type: f64,
}

impl FbpResult {
    /// Every output paired with its Python-package name (the names
    /// `FBP.getParams` accepts, e.g. `hros`, `fF`, `fire_type`), in a fixed
    /// order. This table is the single place the names are defined.
    #[must_use]
    pub fn named_values(&self) -> [(&'static str, f64); 54] {
        [
            ("ws", self.ws),
            ("wd", self.wd),
            ("wse", self.wse),
            ("wse1", self.wse1),
            ("wse2", self.wse2),
            ("wsx", self.wsx),
            ("wsy", self.wsy),
            ("wsv", self.wsv),
            ("raz", self.raz),
            ("m", self.m),
            ("fF", self.f_f),
            ("fW", self.f_w),
            ("ffmc", self.ffmc),
            ("isi", self.isi),
            ("bui", self.bui),
            ("a", self.a),
            ("b", self.b),
            ("c", self.c),
            ("q", self.q),
            ("bui0", self.bui0),
            ("be", self.be),
            ("be_max", self.be_max),
            ("sf", self.sf),
            ("rsz", self.rsz),
            ("rsf", self.rsf),
            ("isf", self.isf),
            ("rsi", self.rsi),
            ("latn", self.latn),
            ("dj", self.dj),
            ("d0", self.d0),
            ("nd", self.nd),
            ("fmc", self.fmc),
            ("fme", self.fme),
            ("ffc", self.ffc),
            ("wfc", self.wfc),
            ("sfc", self.sfc),
            ("cfl", self.cfl),
            ("cfc", self.cfc),
            ("tfc", self.tfc),
            ("cbh", self.cbh),
            ("csfi", self.csfi),
            ("rso", self.rso),
            ("cfb", self.cfb),
            ("fire_type", self.fire_type),
            ("hros", self.hros),
            ("sros", self.sros),
            ("cros", self.cros),
            ("bfw", self.bfw),
            ("bisi", self.bisi),
            ("bros", self.bros),
            ("hfi", self.hfi),
            ("fi_class", self.fi_class),
            ("accel", self.accel),
            ("fuel_type", self.fuel_type),
        ]
    }

    /// Value by its Python-package output name (see
    /// [`FbpResult::named_values`]); `None` for an unknown name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<f64> {
        self.named_values()
            .into_iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| v)
    }
}

// ---------------------------------------------------------------------------
// the scalar chain — facade.runFBP order

/// Run the scalar FBP chain for one cell. Mirror of the Python package's
/// `FBP.initialize(...)` + `runFBP()` for a single-point input.
///
/// A NaN input is a missing/masked cell and propagates to the outputs that
/// depend on it. Fuel codes outside 1..=18 (19 non-fuel, 20 water, unknown) are
/// not an error, and `run` does not force them to NaN: it applies the table
/// fallbacks. For such a code (checked for 0, 19, 20 and 99) `sfc`, `rso`,
/// `tfc` and `hfi` are NaN, while `hros`, `bros` and `cfb` are 0, `fi_class` is
/// -99 and `fmc` is finite; `fire_type` is 0 (1 for code 0), not NaN. Only
/// [`grid::run_grid`](crate::grid::run_grid) forces every output to NaN for a
/// non-modeled cell. An invalid `wx_date` is treated as missing (NaN foliar
/// moisture, so NaN `rso` for codes 1..=13 and, at any percentile other than
/// 50, NaN percentile-adjusted `hros`/`bros`); use [`is_valid_wx_date`] to
/// check it first.
///
/// # Panics
///
/// Never panics, for any input including NaN, infinities and out-of-range
/// values. It returns no `Result` because there is no error path: bad data
/// surfaces as NaN.
///
/// # Examples
///
/// ```
/// use cffdrs_core::fbp::{run, FbpInput};
///
/// let input = FbpInput {
///     fuel_type: 2, // C-2 boreal spruce
///     wx_date: 20_230_615,
///     lat: 52.0,
///     long: -115.0,
///     elevation: 800.0,
///     slope_pct: 10.0,
///     aspect_deg: 180.0,
///     ws: 20.0,
///     wd: 270.0,
///     ffmc: 90.0,
///     bui: 80.0,
///     pc: 0.0,
///     pdf: 0.0,
///     gfl: 0.0,
///     gcf: 0.0,
///     percentile_growth: 50.0,
///     d0_override: None,
///     dj_override: None,
///     fmc_override: None,
///     hros_override: None,
/// };
/// let result = run(&input);
/// assert!(result.hros > 0.0);
/// assert!([1.0, 2.0, 3.0].contains(&result.fire_type));
/// assert_eq!(result.get("hros"), Some(result.hros));
/// ```
// One straight-line pass that mirrors the Python `FBP` pipeline stage by stage;
// splitting it would obscure that correspondence.
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn run(input: &FbpInput) -> FbpResult {
    let ft = FuelType::from_code(input.fuel_type);

    // --- inputs._verify_inputs normalization
    let lat = input.lat;
    let abs_long = input.long.abs();
    let elevation = input.elevation;
    let n = normalize(input);

    // --- invert_wind_aspect
    let (wd, aspect) = invert_wind_aspect(n.wd, n.aspect);
    let n = Normalized { aspect, wd, ..n };
    let (ws, ffmc, bui, pc, pdf) = (n.ws, n.ffmc, n.bui, n.pc, n.pdf);

    // --- calc_sf
    let sf = calc_sf(n.slope);

    // --- calc_isz
    let Isz { m, f_f, isz } = calc_isz(n.ffmc);

    // --- calc_fmc (or the setParams({'fmc': ...}) injection in its place:
    // calcFMC never runs, so latn/d0/dj/nd/fme keep their zero templates —
    // the zero fme is load-bearing for C-6, see FbpInput::fmc_override)
    let fmc = match input.fmc_override {
        Some(v) => injected_fmc(v),
        None => calc_fmc(
            lat,
            abs_long,
            elevation,
            input.wx_date,
            input.d0_override,
            input.dj_override,
        ),
    };

    // --- calc_isi_rsi_be
    let params = ft.ros_params();
    let RosParams {
        a,
        b,
        c,
        q,
        bui0,
        be_max,
    } = params;
    let SpreadIndices {
        rsz,
        rsf,
        isf,
        sw,
        rsi,
        brsi,
        be,
    } = calc_isi_rsi_be(ft, &params, &n, f_f, isz, sf);
    let isi = sw.isi;
    let bisi = sw.bisi;

    // --- calc_ros
    let Ros {
        mut hros,
        mut bros,
        sros,
    } = calc_ros(ft, rsi, brsi, be, bui, input.hros_override);

    // --- calc_sfc
    let SurfaceFuel { sfc, ffc, wfc } = calc_sfc(ft, &n);

    // --- getCBH_CFL
    let CrownFuel { cbh, cfl } = ft.crown_fuel();

    // --- calc_csfi / calc_rso
    let csfi = calc_csfi(ft, cbh, fmc.fmc);
    let rso = calc_rso(sfc, csfi);

    // --- deterministic C-6 blend: SROS-derived CFB -> CFC -> CROS -> blended
    // HROS. This CFB is temporary; it is not the CFB used downstream.
    let mut cros = 0.0;
    if ft == FuelType::C6 {
        let blend_cfb = cfb_from_ros(sros, rso);
        let blend = calc_c6_blend(sros, blend_cfb, cfl, isi, fmc.fme);
        cros = blend.cros;
        hros = blend.hros;
    }

    // --- directional CFB used only to pick the percentile-growth regime
    let percentile_cfb = directional_cfb(ft, hros, rso);
    let percentile_bros_cfb = directional_cfb(ft, bros, rso);

    // --- calc_ros_percentile_growth: 50 is an exact no-op; NaN propagates
    let hros_before_percentile = hros;
    if input.percentile_growth != 50.0 {
        let tinv = percentile_tinv(input.percentile_growth);
        hros = percentile_ros(ft, hros, percentile_cfb, tinv, 1.0);
        bros = percentile_ros(ft, bros, percentile_bros_cfb, tinv, wind_decay(sw.wsv));
    }

    // --- final CFB from the percentile-adjusted head ROS (NaN rule in final_cfb)
    let cfb = final_cfb(ft, hros, hros_before_percentile, rso);

    // --- calc_accel_param
    let accel = calc_accel_param(ft, cfb);

    // --- calc_fire_type
    let fire_type = calc_fire_type(ft, cfb);

    // --- calc_cfc
    let cfc = calc_cfc(ft, cfb, cfl, pc, pdf);

    // ffc/wfc stay NaN where the Python package masks them (fuels without
    // a fine/woody split): the GRID path surfaces masked cells as NaN.
    // The scalar getParams export maps masked -> 0.0 via `.item()`; that
    // convention belongs to the export layer and is applied by the Wotton
    // golden harness, not here.

    // --- calc_tfc / calc_hfi
    let tfc = calc_tfc(sfc, cfc);
    let hfi = calc_hfi(hros, tfc);

    // --- calc_fire_intensity_class
    let fi_class = calc_fire_intensity_class(hfi);

    FbpResult {
        ws,
        wd,
        wse: sw.wse,
        wse1: sw.wse1,
        wse2: sw.wse2,
        wsx: sw.wsx,
        wsy: sw.wsy,
        wsv: sw.wsv,
        raz: sw.raz,
        m,
        f_f,
        f_w: sw.f_w,
        ffmc,
        isi,
        bui,
        a,
        b,
        c,
        q,
        bui0,
        be,
        be_max,
        sf,
        rsz,
        rsf,
        isf,
        rsi,
        latn: fmc.latn,
        dj: fmc.dj,
        d0: fmc.d0,
        nd: fmc.nd,
        fmc: fmc.fmc,
        fme: fmc.fme,
        ffc,
        wfc,
        sfc,
        cfl,
        cfc,
        tfc,
        cbh,
        csfi,
        rso,
        cfb,
        fire_type,
        hros,
        sros,
        cros,
        bfw: sw.bfw,
        bisi,
        bros,
        hfi,
        fi_class,
        accel,
        fuel_type: f64::from(ft.code()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn field_name_table_is_complete_and_unique() {
        let names: Vec<&str> = FbpResult::default()
            .named_values()
            .iter()
            .map(|(n, _)| *n)
            .collect();
        assert_eq!(names.len(), 54);
        let unique: HashSet<&str> = names.iter().copied().collect();
        assert_eq!(unique.len(), 54, "duplicate output names");
    }

    /// Python `getParams` names, each with the value of the field it must read
    /// in `named_values_pair_each_name_with_its_own_field`.
    const EXPECTED_NAMED_VALUES: [(&str, f64); 54] = [
        ("ws", 1.0),
        ("wd", 2.0),
        ("wse", 3.0),
        ("wse1", 4.0),
        ("wse2", 5.0),
        ("wsx", 6.0),
        ("wsy", 7.0),
        ("wsv", 8.0),
        ("raz", 9.0),
        ("m", 10.0),
        ("fF", 11.0),
        ("fW", 12.0),
        ("ffmc", 13.0),
        ("isi", 14.0),
        ("bui", 15.0),
        ("a", 16.0),
        ("b", 17.0),
        ("c", 18.0),
        ("q", 19.0),
        ("bui0", 20.0),
        ("be", 21.0),
        ("be_max", 22.0),
        ("sf", 23.0),
        ("rsz", 24.0),
        ("rsf", 25.0),
        ("isf", 26.0),
        ("rsi", 27.0),
        ("latn", 28.0),
        ("dj", 29.0),
        ("d0", 30.0),
        ("nd", 31.0),
        ("fmc", 32.0),
        ("fme", 33.0),
        ("ffc", 34.0),
        ("wfc", 35.0),
        ("sfc", 36.0),
        ("cfl", 37.0),
        ("cfc", 38.0),
        ("tfc", 39.0),
        ("cbh", 40.0),
        ("csfi", 41.0),
        ("rso", 42.0),
        ("cfb", 43.0),
        ("fire_type", 44.0),
        ("hros", 45.0),
        ("sros", 46.0),
        ("cros", 47.0),
        ("bfw", 48.0),
        ("bisi", 49.0),
        ("bros", 50.0),
        ("hfi", 51.0),
        ("fi_class", 52.0),
        ("accel", 53.0),
        ("fuel_type", 54.0),
    ];

    /// Every field holds a distinct value (1.0..=54.0 in declaration order), so
    /// a name wired to the wrong field in `named_values` no longer matches
    /// `EXPECTED_NAMED_VALUES`.
    #[test]
    fn named_values_pair_each_name_with_its_own_field() {
        let r = FbpResult {
            ws: 1.0,
            wd: 2.0,
            wse: 3.0,
            wse1: 4.0,
            wse2: 5.0,
            wsx: 6.0,
            wsy: 7.0,
            wsv: 8.0,
            raz: 9.0,
            m: 10.0,
            f_f: 11.0,
            f_w: 12.0,
            ffmc: 13.0,
            isi: 14.0,
            bui: 15.0,
            a: 16.0,
            b: 17.0,
            c: 18.0,
            q: 19.0,
            bui0: 20.0,
            be: 21.0,
            be_max: 22.0,
            sf: 23.0,
            rsz: 24.0,
            rsf: 25.0,
            isf: 26.0,
            rsi: 27.0,
            latn: 28.0,
            dj: 29.0,
            d0: 30.0,
            nd: 31.0,
            fmc: 32.0,
            fme: 33.0,
            ffc: 34.0,
            wfc: 35.0,
            sfc: 36.0,
            cfl: 37.0,
            cfc: 38.0,
            tfc: 39.0,
            cbh: 40.0,
            csfi: 41.0,
            rso: 42.0,
            cfb: 43.0,
            fire_type: 44.0,
            hros: 45.0,
            sros: 46.0,
            cros: 47.0,
            bfw: 48.0,
            bisi: 49.0,
            bros: 50.0,
            hfi: 51.0,
            fi_class: 52.0,
            accel: 53.0,
            fuel_type: 54.0,
        };
        assert_eq!(r.named_values(), EXPECTED_NAMED_VALUES);
    }

    #[test]
    fn get_resolves_names_to_fields() {
        let r = FbpResult {
            f_f: 1.5,
            f_w: 2.5,
            be_max: 3.5,
            fuel_type: 7.0,
            ..FbpResult::default()
        };
        assert_eq!(r.get("fF"), Some(r.f_f));
        assert_eq!(r.get("fW"), Some(r.f_w));
        assert_eq!(r.get("be_max"), Some(r.be_max));
        assert_eq!(r.get("fuel_type"), Some(r.fuel_type));
        assert_eq!(r.get("nope"), None);
    }
}
