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

pub use crate::fmc::is_valid_wx_date;
use crate::fmc::{calc_fmc, injected_fmc};
use crate::fuel::{CrownFuel, FuelType, RosParams};
use crate::normalize::{invert_wind_aspect, normalize};

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
    /// Value by its Python-package output name (the names `getParams`
    /// accepts, e.g. "hros", "fF", "fire_type"). None for unknown names.
    pub fn get(&self, name: &str) -> Option<f64> {
        Some(match name {
            "ws" => self.ws,
            "wd" => self.wd,
            "wse" => self.wse,
            "wse1" => self.wse1,
            "wse2" => self.wse2,
            "wsx" => self.wsx,
            "wsy" => self.wsy,
            "wsv" => self.wsv,
            "raz" => self.raz,
            "m" => self.m,
            "fF" => self.f_f,
            "fW" => self.f_w,
            "ffmc" => self.ffmc,
            "isi" => self.isi,
            "bui" => self.bui,
            "a" => self.a,
            "b" => self.b,
            "c" => self.c,
            "q" => self.q,
            "bui0" => self.bui0,
            "be" => self.be,
            "be_max" => self.be_max,
            "sf" => self.sf,
            "rsz" => self.rsz,
            "rsf" => self.rsf,
            "isf" => self.isf,
            "rsi" => self.rsi,
            "latn" => self.latn,
            "dj" => self.dj,
            "d0" => self.d0,
            "nd" => self.nd,
            "fmc" => self.fmc,
            "fme" => self.fme,
            "ffc" => self.ffc,
            "wfc" => self.wfc,
            "sfc" => self.sfc,
            "cfl" => self.cfl,
            "cfc" => self.cfc,
            "tfc" => self.tfc,
            "cbh" => self.cbh,
            "csfi" => self.csfi,
            "rso" => self.rso,
            "cfb" => self.cfb,
            "fire_type" => self.fire_type,
            "hros" => self.hros,
            "sros" => self.sros,
            "cros" => self.cros,
            "bfw" => self.bfw,
            "bisi" => self.bisi,
            "bros" => self.bros,
            "hfi" => self.hfi,
            "fi_class" => self.fi_class,
            "accel" => self.accel,
            "fuel_type" => self.fuel_type,
            _ => return None,
        })
    }
}

// ---------------------------------------------------------------------------
// constants.py

// ---------------------------------------------------------------------------
// slope_wind.py

struct SlopeWindIsi {
    wse1: f64,
    wse2: f64,
    wse: f64,
    wsx: f64,
    wsy: f64,
    wsv: f64,
    raz: f64,
    f_w: f64,
    bfw: f64,
    isi: f64,
    bisi: f64,
}

/// `slope_wind.calc_slope_wind_isi`
fn calc_slope_wind_isi(isf: f64, f_f: f64, wd: f64, aspect: f64, ws: f64) -> SlopeWindIsi {
    let wse1 = (1.0 / 0.05039) * (isf / (0.208 * f_f)).ln();
    let wse2 = if isf.is_nan() {
        // masked isf keeps its mask through the where() — never the cap
        f64::NAN
    } else if isf < 0.999 * 2.496 * f_f {
        28.0 - (1.0 / 0.0818) * (1.0 - isf / (2.496 * f_f)).ln()
    } else {
        112.45 // cap maximum WSE
    };
    let wse = if wse1 <= 40.0 { wse1 } else { wse2 };

    let (sin_wd, cos_wd) = (wd.to_radians().sin(), wd.to_radians().cos());
    let (sin_asp, cos_asp) = (aspect.to_radians().sin(), aspect.to_radians().cos());
    let wsx = ws * sin_wd + wse * sin_asp;
    let wsy = ws * cos_wd + wse * cos_asp;
    let wsv = (wsx * wsx + wsy * wsy).sqrt();

    let acos_val = (wsy / wsv).clamp(-1.0, 1.0);
    let angle_deg = acos_val.acos().to_degrees();
    let mut raz = if wsx < 0.0 {
        360.0 - angle_deg
    } else {
        angle_deg
    };
    // wsv == 0: azimuth undefined, spread circular — substitute 0 to keep raz
    // finite for downstream consumers (matches the Python fix). NaN wsv is a
    // masked cell in Python (`where(wsv > 0, raz, 0)` keeps the mask), so
    // NaN must pass through, not become 0.
    // (`wsv <= 0.0` is false for NaN, so a NaN wsv is left untouched.)
    if wsv <= 0.0 {
        raz = 0.0;
    }

    let f_w = if wsv > 40.0 {
        12.0 * (1.0 - (-0.0818 * (wsv - 28.0)).exp())
    } else {
        (0.05039 * wsv).exp()
    };
    let bfw = (-0.05039 * wsv).exp();
    SlopeWindIsi {
        wse1,
        wse2,
        wse,
        wsx,
        wsy,
        wsv,
        raz,
        f_w,
        bfw,
        isi: 0.208 * f_f * f_w,
        bisi: 0.208 * f_f * bfw,
    }
}

/// One fuel's `a * (1 - exp(-b * x))^c` spread curve.
fn ros_curve(a: f64, b: f64, c: f64, x: f64) -> f64 {
    a * (1.0 - (-b * x).exp()).powf(c)
}

/// The `isf >= 0.01` numerator guard shared by every ISF branch.
/// NaN must PROPAGATE: in the Python package the numerator arrives masked
/// (NaN inputs are masked by `_coerce`), and `mask.where` keeps the mask no
/// matter which branch is selected — the observable ISF is NaN. A plain
/// `if` would silently take the finite fallback branch here.
fn isf_core(numer: f64, b: f64) -> f64 {
    if numer.is_nan() {
        f64::NAN
    } else if numer >= 0.01 {
        numer.ln() / -b
    } else {
        0.01_f64.ln() / -b
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
    let (ws, ffmc, bui, pc, pdf, gfl, gcf, slope) =
        (n.ws, n.ffmc, n.bui, n.pc, n.pdf, n.gfl, n.gcf, n.slope);

    // --- invert_wind_aspect
    let (wd, aspect) = invert_wind_aspect(n.wd, n.aspect);

    // --- calc_sf
    // where(slope < 70, exp(...), 10): NaN slope stays masked in Python —
    // propagate it rather than taking the finite cap branch.
    let sf = if slope.is_nan() {
        f64::NAN
    } else if slope < 70.0 {
        (3.533 * (slope / 100.0).powf(1.2)).exp()
    } else {
        10.0
    };

    // --- calc_isz
    let m = (250.0 * (59.5 / 101.0) * (101.0 - ffmc)) / (59.5 + ffmc);
    let f_f = (91.9 * (-0.1386 * m).exp()) * (1.0 + m.powf(5.31) / (4.93 * 1.0e7));
    let isz = 0.208 * f_f;

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
    let RosParams {
        a,
        b,
        c,
        q,
        bui0,
        be_max,
    } = ft.ros_params();
    let c2 = FuelType::C2.ros_params();
    let d1 = FuelType::D1.ros_params();
    let m12 = matches!(ft, FuelType::M1 | FuelType::M2);
    let m34 = matches!(ft, FuelType::M3 | FuelType::M4);
    let o1 = matches!(ft, FuelType::O1a | FuelType::O1b);

    let cf = if gcf.is_nan() {
        f64::NAN
    } else if gcf < 58.8 {
        0.005 * ((0.061 * gcf).exp() - 1.0)
    } else {
        0.176 + 0.02 * (gcf - 58.8)
    };

    let rsz_core = ros_curve(a, b, c, isz);
    let rsz_c2 = ros_curve(c2.a, c2.b, c2.c, isz);
    let rsz_d1 = ros_curve(d1.a, d1.b, d1.c, isz);
    let rsz = match ft {
        FuelType::M1 => (pc / 100.0) * rsz_c2 + (1.0 - pc / 100.0) * rsz_d1,
        FuelType::M2 => (pc / 100.0) * rsz_c2 + 0.2 * (1.0 - pc / 100.0) * rsz_d1,
        FuelType::O1a | FuelType::O1b => rsz_core * cf,
        _ => rsz_core,
    };

    let rsf_c2 = rsz_c2 * sf;
    let rsf_d1 = rsz_d1 * sf;
    let rsf = rsz * sf;

    let isf_c2_core = isf_core(1.0 - (rsf_c2 / c2.a).powf(1.0 / c2.c), c2.b);
    let isf_d1_core = isf_core(1.0 - (rsf_d1 / d1.a).powf(1.0 / d1.c), d1.b);
    let isf_m34_core = isf_core(1.0 - (rsf / a).powf(1.0 / c), b);
    let isf = if m12 {
        (pc / 100.0) * isf_c2_core + (1.0 - pc / 100.0) * isf_d1_core
    } else if m34 {
        (pdf / 100.0) * isf_m34_core + (1.0 - pdf / 100.0) * isf_d1_core
    } else {
        let numer = if o1 {
            1.0 - (rsf / (a * cf)).powf(1.0 / c)
        } else {
            1.0 - (rsf / a).powf(1.0 / c)
        };
        isf_core(numer, b)
    };

    let sw = calc_slope_wind_isi(isf, f_f, wd, aspect, ws);
    let isi = sw.isi;
    let bisi = sw.bisi;

    let rsi_c2 = ros_curve(c2.a, c2.b, c2.c, isi);
    let rsi_d1 = ros_curve(d1.a, d1.b, d1.c, isi);
    let rsi = match ft {
        FuelType::M3 => (pdf / 100.0) * ros_curve(a, b, c, isi) + (1.0 - pdf / 100.0) * rsi_d1,
        FuelType::M4 => {
            (pdf / 100.0) * ros_curve(a, b, c, isi) + 0.2 * (1.0 - pdf / 100.0) * rsi_d1
        }
        FuelType::M1 => (pc / 100.0) * rsi_c2 + (1.0 - pc / 100.0) * rsi_d1,
        FuelType::M2 => (pc / 100.0) * rsi_c2 + 0.2 * (1.0 - pc / 100.0) * rsi_d1,
        FuelType::O1a | FuelType::O1b => ros_curve(a, b, c, isi) * cf,
        _ => ros_curve(a, b, c, isi),
    };
    let brsi_c2 = ros_curve(c2.a, c2.b, c2.c, bisi);
    let brsi_d1 = ros_curve(d1.a, d1.b, d1.c, bisi);
    let brsi = match ft {
        FuelType::M3 => (pdf / 100.0) * ros_curve(a, b, c, bisi) + (1.0 - pdf / 100.0) * brsi_d1,
        FuelType::M4 => {
            (pdf / 100.0) * ros_curve(a, b, c, bisi) + 0.2 * (1.0 - pdf / 100.0) * brsi_d1
        }
        FuelType::M2 => (pc / 100.0) * brsi_c2 + 0.2 * (1.0 - pc / 100.0) * brsi_d1,
        FuelType::M1 => (pc / 100.0) * brsi_c2 + (1.0 - pc / 100.0) * brsi_d1,
        FuelType::O1a | FuelType::O1b => ros_curve(a, b, c, bisi) * cf,
        _ => ros_curve(a, b, c, bisi),
    };

    let be = {
        // Python: where((bui==0)|~isfinite(bui), 0, ...) — but a NaN bui
        // arrives MASKED there (inputs._coerce masks NaN), so the isfinite
        // branch only ever catches literal infinities; the masked NaN rides
        // through and the observable spread outputs (hros/hfi) are NaN.
        // Mirror the observables: NaN bui => NaN be; infinite bui => 0.
        let raw = if bui.is_nan() {
            f64::NAN
        } else if bui == 0.0 || bui.is_infinite() {
            0.0
        } else if bui0 == 0.0 || !bui0.is_finite() {
            1.0
        } else {
            (50.0 * q.ln() * (1.0 / bui - 1.0 / bui0)).exp()
        };
        raw.clamp(0.0, be_max)
    };

    // --- calc_ros
    let mut hros = rsi * be;
    let mut bros = brsi * be;
    let mut sros = 0.0;
    if ft == FuelType::C6 {
        sros = rsi * be;
    }
    if ft == FuelType::D2 {
        // D2 correction: zero out if BUI < 70, then scale by 0.2
        if bui < 70.0 {
            hros = 0.0;
            bros = 0.0;
        } else {
            hros *= 0.2;
            bros *= 0.2;
        }
    }
    // setParams({'hros': ...}) injection point: replaces hros after calcROS,
    // before calcCSFI onward. bros/sros keep their computed values (the C-6
    // blend reads sros and may overwrite the injected hros).
    if let Some(v) = input.hros_override {
        hros = v;
    }

    // --- calc_sfc
    let mut ffc = f64::NAN;
    let mut wfc = f64::NAN;
    let sfc = match ft {
        FuelType::C1 => {
            if ffmc > 84.0 {
                0.75 + 0.75 * (1.0 - (-0.23 * (ffmc - 84.0)).exp()).sqrt()
            } else {
                0.75 - 0.75 * (1.0 - (0.23 * (ffmc - 84.0)).exp()).sqrt()
            }
        }
        FuelType::C2 => 5.0 * (1.0 - (-0.0115 * bui).exp()),
        FuelType::C3 | FuelType::C4 => 5.0 * (1.0 - (-0.0164 * bui).exp()).powf(2.24),
        FuelType::C5 | FuelType::C6 => 5.0 * (1.0 - (-0.0149 * bui).exp()).powf(2.48),
        FuelType::C7 => {
            ffc = (2.0 * (1.0 - (-0.104 * (ffmc - 70.0)).exp())).max(0.0);
            wfc = 1.5 * (1.0 - (-0.0201 * bui).exp());
            ffc + wfc
        }
        FuelType::D1 | FuelType::D2 => 1.5 * (1.0 - (-0.0183 * bui).exp()),
        FuelType::M1 | FuelType::M2 => {
            let c2_sfc = 5.0 * (1.0 - (-0.0115 * bui).exp());
            let d1_sfc = 1.5 * (1.0 - (-0.0183 * bui).exp());
            (pc / 100.0) * c2_sfc + ((100.0 - pc) / 100.0) * d1_sfc
        }
        FuelType::M3 | FuelType::M4 => 5.0 * (1.0 - (-0.0115 * bui).exp()),
        FuelType::O1a | FuelType::O1b => gfl,
        FuelType::S1 => {
            ffc = 4.0 * (1.0 - (-0.025 * bui).exp());
            wfc = 4.0 * (1.0 - (-0.034 * bui).exp());
            ffc + wfc
        }
        FuelType::S2 => {
            ffc = 10.0 * (1.0 - (-0.013 * bui).exp());
            wfc = 6.0 * (1.0 - (-0.06 * bui).exp());
            ffc + wfc
        }
        FuelType::S3 => {
            ffc = 12.0 * (1.0 - (-0.0166 * bui).exp());
            wfc = 20.0 * (1.0 - (-0.021 * bui).exp());
            ffc + wfc
        }
        _ => f64::NAN,
    };

    // --- getCBH_CFL
    let CrownFuel { cbh, cfl } = ft.crown_fuel();

    // --- calc_csfi / calc_rso
    let csfi = if ft.code() < 14 {
        (0.01 * cbh * (460.0 + 25.9 * fmc.fmc)).powf(1.5)
    } else {
        0.0
    };
    let rso = if sfc > 0.0 { csfi / (300.0 * sfc) } else { 0.0 };

    // --- crown fraction burned. The equation is the same for every crowning
    // fuel (including C-6); only the ROS it is applied to differs per step.
    let cfb_from_ros = |ros: f64| -> f64 {
        let delta = ros - rso;
        let mut cfb = if delta < -3086.0 {
            0.0
        } else {
            1.0 - (-0.23 * delta).exp()
        };
        if !cfb.is_finite() && !cfb.is_nan() {
            // infinities zero out; NaN is a masked cell in Python and must
            // stay NaN (grid-truth: cfb/accel are NaN at NaN-input cells)
            cfb = 0.0;
        }
        cfb.clamp(0.0, 1.0)
    };
    let crowning = ft.is_modeled() && !ft.is_non_crowning();
    let directional_cfb = |ros: f64| -> f64 {
        if crowning {
            cfb_from_ros(ros)
        } else {
            0.0
        }
    };

    // --- deterministic C-6 blend: SROS-derived CFB -> CFC -> CROS -> blended
    // HROS. This CFB is temporary; it is not the CFB used downstream.
    let mut cros = 0.0;
    if ft == FuelType::C6 {
        let blend_cfb = cfb_from_ros(sros);
        let blend_cfc = blend_cfb * cfl;
        cros = if blend_cfc == 0.0 {
            0.0
        } else {
            60.0 * (1.0 - (-0.0497 * isi).exp()) * (fmc.fme / 0.778237)
        };
        hros = sros + blend_cfb * (cros - sros);
    }

    // --- directional CFB used only to pick the percentile-growth regime
    let percentile_cfb = directional_cfb(hros);
    let percentile_bros_cfb = directional_cfb(bros);

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

    // --- final CFB from the percentile-adjusted head ROS. A NaN that appears
    // only at the percentile step (NaN percentile) is an unmasked non-finite
    // value in Python, which the CFB sanitiser zeroes; a NaN that was already
    // there is a masked cell and stays NaN.
    let cfb = if hros.is_nan() && !hros_before_percentile.is_nan() {
        0.0
    } else {
        directional_cfb(hros)
    };

    // --- calc_accel_param
    let accel = if ft.is_open() {
        0.115
    } else if ft.is_modeled() {
        0.115 - 18.8 * cfb.powf(2.5) * (-8.0 * cfb).exp()
    } else {
        0.0
    };

    // --- calc_fire_type
    let fire_type = if ft.code() < 19 {
        if cfb.is_nan() {
            0.0 // masked cell: grid-truth observable is 0, not a class
        } else if cfb <= 0.1 {
            1.0
        } else if cfb < 0.9 {
            2.0
        } else {
            3.0
        }
    } else {
        0.0
    };

    // --- calc_cfc
    let cfc = match ft {
        FuelType::M1 | FuelType::M2 => cfb * cfl * pc / 100.0,
        FuelType::M3 | FuelType::M4 => cfb * cfl * pdf / 100.0,
        _ => cfb * cfl,
    };

    // ffc/wfc stay NaN where the Python package masks them (fuels without
    // a fine/woody split): the GRID path surfaces masked cells as NaN.
    // The scalar getParams export maps masked -> 0.0 via `.item()`; that
    // convention belongs to the export layer and is applied by the Wotton
    // golden harness, not here.

    // --- calc_tfc / calc_hfi
    let tfc = sfc + cfc;
    let hfi = 300.0 * hros * tfc;

    // --- calc_fire_intensity_class
    let fi_class = if hfi > 10000.0 {
        6.0
    } else if hfi > 4000.0 {
        5.0
    } else if hfi > 2000.0 {
        4.0
    } else if hfi > 500.0 {
        3.0
    } else if hfi > 10.0 {
        2.0
    } else if hfi > 0.0 {
        1.0
    } else {
        -99.0
    };

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

/// The grid pass: per-cell fuel/terrain plus per-cell wind, scalar ffmc/bui/
/// date. One call per weather step; non-modeled cells (codes outside 1..18)
/// yield NaN behaviour, which engines exclude via their burnable masks.
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
    let n = fuel_type.len();
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
        if !FuelType::from_code(fuel_type[i]).is_modeled() {
            continue;
        }
        let r = run(&FbpInput {
            fuel_type: fuel_type[i],
            wx_date,
            lat: lat[i],
            long: long[i],
            elevation: elevation[i],
            slope_pct: slope_pct[i],
            aspect_deg: aspect_deg[i],
            ws: ws[i],
            wd: wd[i],
            ffmc,
            bui,
            pc: pct_conifer[i],
            pdf: pct_dead_fir,
            gfl: grass_fuel_load,
            gcf: grass_curing[i],
            percentile_growth,
            d0_override: None,
            dj_override: None,
            fmc_override: None,
            hros_override: None,
        });
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
    out
}
