//! The CFFBPS scalar core and grid pass.
//!
//! Mirrors `src/cffdrs/cffbps/` (the reference implementation) function by
//! function: `inputs._verify_inputs` normalization, then the facade's
//! `runFBP()` chain in the same order, with the same fuel-code conventions
//! (1..18 modeled, 19/20 non-fuel) and the same NaN semantics (masked cells
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
use crate::ros::{calc_c6_blend, calc_ros, Ros};
use crate::slope_wind::{calc_isi_rsi_be, calc_isz, calc_sf, Isz, SpreadIndices};
use crate::surface::{calc_sfc, SurfaceFuel};

/// Scalar inputs, matching `FBP.initialize` in the Python package.
#[derive(Debug, Clone)]
pub struct FbpInput {
    /// CFFBPS numeric fuel code (1..18 modeled; 19/20 non-fuel).
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

/// Everything the scalar pass computes: the snapshot's 54 quantities.
/// All f64, including code-like values (`fire_type`, `fi_class`), so golden
/// comparison is uniform; consumers cast as needed.
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
}

impl FbpResult {
    /// Every output paired with its Python-package name (the names
    /// `FBP.getParams` accepts, e.g. "hros", "fF", "fire_type"), in a fixed
    /// order. This table is the single place the names are defined.
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

    /// Value by its Python-package output name; `None` for an unknown name.
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
pub fn run(input: &FbpInput) -> FbpResult {
    let ft = FuelType::from_code(input.fuel_type);

    // --- inputs._verify_inputs normalization
    let lat = input.lat;
    let abs_long = input.long.abs();
    let elevation = input.elevation;
    let n = normalize(input);

    // --- invert_wind_aspect
    let (wd, aspect) = invert_wind_aspect(n.wd, n.aspect);
    let n = Normalized { wd, aspect, ..n };
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
        let tinv = crate::percentile::percentile_tinv(input.percentile_growth);
        hros = crate::percentile::percentile_ros(ft, hros, percentile_cfb, tinv, 1.0);
        bros = crate::percentile::percentile_ros(
            ft,
            bros,
            percentile_bros_cfb,
            tinv,
            crate::percentile::wind_decay(sw.wsv),
        );
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
        fuel_type: ft.code() as f64,
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
