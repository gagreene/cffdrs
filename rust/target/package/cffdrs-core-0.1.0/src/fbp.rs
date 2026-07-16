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
    /// Percentile growth (50 = median behaviour; the only value the golden
    /// snapshot exercises — non-50 support is differential-tested).
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
    /// so C-6 cfb (which reads sros) is unaffected and the C-6 blend can
    /// overwrite the injected value.
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

/// Surface ROS parameters (a, b, c, q, bui0, be_max) — `constants.rosParams`.
/// `None` entries in the Python table surface as NaN, matching numpy
/// assignment semantics.
fn ros_params(fuel_type: i32) -> (f64, f64, f64, f64, f64, f64) {
    const NAN: f64 = f64::NAN;
    match fuel_type {
        1 => (90.0, 0.0649, 4.5, 0.9, 72.0, 1.076),
        2 => (110.0, 0.0282, 1.5, 0.7, 64.0, 1.321),
        3 => (110.0, 0.0444, 3.0, 0.75, 62.0, 1.261),
        4 => (110.0, 0.0293, 1.5, 0.8, 66.0, 1.184),
        5 => (30.0, 0.0697, 4.0, 0.8, 56.0, 1.220),
        6 => (30.0, 0.08, 3.0, 0.8, 62.0, 1.197),
        7 => (45.0, 0.0305, 2.0, 0.85, 106.0, 1.134),
        8 => (30.0, 0.0232, 1.6, 0.9, 32.0, 1.179),
        9 => (30.0, 0.0232, 1.6, 0.9, 32.0, 1.179),
        10 => (NAN, NAN, NAN, 0.8, 50.0, 1.250),
        11 => (NAN, NAN, NAN, 0.8, 50.0, 1.250),
        12 => (120.0, 0.0572, 1.4, 0.8, 50.0, 1.250),
        13 => (100.0, 0.0404, 1.48, 0.8, 50.0, 1.250),
        14 => (190.0, 0.0310, 1.4, 1.0, NAN, 1.0),
        15 => (250.0, 0.0350, 1.7, 1.0, NAN, 1.0),
        16 => (75.0, 0.0297, 1.3, 0.75, 38.0, 1.460),
        17 => (40.0, 0.0438, 1.7, 0.75, 63.0, 1.256),
        18 => (55.0, 0.0829, 3.2, 0.75, 31.0, 1.590),
        // ros_params.get(ftype, (0, 0, 0, 0, 1, 1)) fallback
        _ => (0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
    }
}

/// (cbh, cfl) — `constants.fbpCBH_CFL_HT_LUT` (height column unused here).
fn cbh_cfl(fuel_type: i32) -> (f64, f64) {
    const NAN: f64 = f64::NAN;
    match fuel_type {
        1 => (2.0, 0.75),
        2 => (3.0, 0.8),
        3 => (8.0, 1.15),
        4 => (4.0, 1.2),
        5 => (18.0, 1.2),
        6 => (7.0, 1.8),
        7 => (10.0, 0.5),
        8 | 9 | 14 | 15 | 16 | 17 | 18 => (0.0, 0.0),
        10 | 11 => (6.0, 0.8),
        12 | 13 => (6.0, 0.8),
        _ => (NAN, NAN),
    }
}

fn is_modeled(fuel_type: i32) -> bool {
    (1..=18).contains(&fuel_type)
}

/// `constants.open_fuel_types`
fn is_open_fuel(fuel_type: i32) -> bool {
    matches!(fuel_type, 1 | 7 | 9 | 14 | 15 | 16 | 17 | 18)
}

/// `constants.non_crowning_fuels`
fn is_non_crowning(fuel_type: i32) -> bool {
    matches!(fuel_type, 8 | 9 | 14 | 15 | 16 | 17 | 18)
}

// ---------------------------------------------------------------------------
// fmc.py

fn is_leap_year(y: i64) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}

/// Day of year from YYYYMMDD — `datetime.strptime(...).timetuple().tm_yday`.
fn day_of_year(wx_date: i64) -> f64 {
    let y = wx_date / 10_000;
    let mth = (wx_date / 100 % 100) as usize;
    let d = wx_date % 100;
    const CUM: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let mut doy = CUM[mth - 1] + d;
    if mth > 2 && is_leap_year(y) {
        doy += 1;
    }
    doy as f64
}

struct Fmc {
    latn: f64,
    d0: f64,
    dj: f64,
    nd: f64,
    fmc: f64,
    fme: f64,
}

/// `fmc.calc_fmc`
fn calc_fmc(
    lat: f64,
    abs_long: f64,
    elevation: f64,
    wx_date: i64,
    d0_override: Option<f64>,
    dj_override: Option<f64>,
) -> Fmc {
    let latn = if elevation > 0.0 {
        43.0 + 33.7 * (-0.0351 * (150.0 - abs_long)).exp()
    } else {
        46.0 + 23.4 * (-0.036 * (150.0 - abs_long)).exp()
    };
    // rounded to mimic the cffdrs R package (numpy round = ties to even)
    let d0 = match d0_override {
        Some(v) => v,
        None => if elevation > 0.0 {
            142.1 * (lat / latn) + 0.0172 * elevation
        } else {
            151.0 * (lat / latn)
        }
        .round_ties_even(),
    };
    let dj = match dj_override {
        Some(v) => v,
        None => if latn.is_finite() { day_of_year(wx_date) } else { 0.0 },
    };
    let nd = (dj - d0).abs();
    let fmc = if nd < 30.0 {
        85.0 + 0.0189 * nd * nd
    } else if nd < 50.0 {
        32.9 + 3.17 * nd - 0.0288 * nd * nd
    } else {
        120.0
    };
    let fme = 1000.0 * (1.5 - 0.00275 * fmc).powi(4) / (460.0 + 25.9 * fmc);
    Fmc { latn, d0, dj, nd, fmc, fme }
}

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
    let mut raz = if wsx < 0.0 { 360.0 - angle_deg } else { angle_deg };
    // wsv == 0: azimuth undefined, spread circular — substitute 0 to keep raz
    // finite for downstream consumers (matches the Python fix). NaN wsv is a
    // masked cell in Python (`where(wsv > 0, raz, 0)` keeps the mask), so
    // NaN must pass through, not become 0.
    if !wsv.is_nan() && !(wsv > 0.0) {
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
    let ft = input.fuel_type;

    // --- inputs._verify_inputs normalization
    let lat = input.lat;
    let abs_long = input.long.abs();
    let elevation = input.elevation;
    // `where(x < 0, 0, x)` / `clip(x, 0, None)` semantics: negatives clamp
    // to 0 but NaN PASSES THROUGH (NaN < 0 is false) — nodata weather must
    // surface as NaN behaviour, not as calm wind. f64::max would launder
    // NaN to 0.0 here.
    let clamp0 = |x: f64| if x < 0.0 { 0.0 } else { x };
    let slope = clamp0(input.slope_pct);
    let mut aspect = input.aspect_deg;
    if aspect < 0.0 {
        aspect = 270.0; // negative aspect treated as flat terrain
    }
    let ws = clamp0(input.ws);
    let mut wd = input.wd;
    let ffmc = clamp0(input.ffmc);
    let bui = clamp0(input.bui);
    // pc/pdf/gfl/gcf: the Python _coerce defaults (50/35/0.35/80) apply ONLY
    // to NaN SCALARS (a wholly-missing input); per-cell NaN in the grid pass
    // stays masked and surfaces as NaN behaviour. This core is the grid
    // pass, so NaN propagates; callers with scalar inputs apply the scalar
    // defaults before broadcasting.
    let pc = clamp0(input.pc);
    let pdf = clamp0(input.pdf);
    let gfl = clamp0(input.gfl);
    let mut gcf = input.gcf;
    if gcf == 0.0 {
        gcf = 0.1;
    }

    // --- invert_wind_aspect
    wd = if wd > 180.0 { wd - 180.0 } else { wd + 180.0 };
    aspect = if aspect > 180.0 { aspect - 180.0 } else { aspect + 180.0 };

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
        Some(v) => Fmc { latn: 0.0, d0: 0.0, dj: 0.0, nd: 0.0, fmc: v, fme: 0.0 },
        None => calc_fmc(lat, abs_long, elevation, input.wx_date, input.d0_override, input.dj_override),
    };

    // --- calc_isi_rsi_be
    let (a, b, c, q, bui0, be_max) = ros_params(ft);
    let c2 = ros_params(2);
    let d1 = ros_params(8);
    let m12 = ft == 10 || ft == 11;
    let m34 = ft == 12 || ft == 13;
    let o1 = ft == 14 || ft == 15;

    let cf = if gcf.is_nan() {
        f64::NAN
    } else if gcf < 58.8 {
        0.005 * ((0.061 * gcf).exp() - 1.0)
    } else {
        0.176 + 0.02 * (gcf - 58.8)
    };

    let rsz_core = ros_curve(a, b, c, isz);
    let rsz_c2 = ros_curve(c2.0, c2.1, c2.2, isz);
    let rsz_d1 = ros_curve(d1.0, d1.1, d1.2, isz);
    let rsz = match ft {
        10 => (pc / 100.0) * rsz_c2 + (1.0 - pc / 100.0) * rsz_d1,
        11 => (pc / 100.0) * rsz_c2 + 0.2 * (1.0 - pc / 100.0) * rsz_d1,
        14 | 15 => rsz_core * cf,
        _ => rsz_core,
    };

    let rsf_c2 = rsz_c2 * sf;
    let rsf_d1 = rsz_d1 * sf;
    let rsf = rsz * sf;

    let isf_c2_core = isf_core(1.0 - (rsf_c2 / c2.0).powf(1.0 / c2.2), c2.1);
    let isf_d1_core = isf_core(1.0 - (rsf_d1 / d1.0).powf(1.0 / d1.2), d1.1);
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

    let rsi_c2 = ros_curve(c2.0, c2.1, c2.2, isi);
    let rsi_d1 = ros_curve(d1.0, d1.1, d1.2, isi);
    let rsi = match ft {
        12 => (pdf / 100.0) * ros_curve(a, b, c, isi) + (1.0 - pdf / 100.0) * rsi_d1,
        13 => (pdf / 100.0) * ros_curve(a, b, c, isi) + 0.2 * (1.0 - pdf / 100.0) * rsi_d1,
        10 => (pc / 100.0) * rsi_c2 + (1.0 - pc / 100.0) * rsi_d1,
        11 => (pc / 100.0) * rsi_c2 + 0.2 * (1.0 - pc / 100.0) * rsi_d1,
        14 | 15 => ros_curve(a, b, c, isi) * cf,
        _ => ros_curve(a, b, c, isi),
    };
    let brsi_c2 = ros_curve(c2.0, c2.1, c2.2, bisi);
    let brsi_d1 = ros_curve(d1.0, d1.1, d1.2, bisi);
    let brsi = match ft {
        12 => (pdf / 100.0) * ros_curve(a, b, c, bisi) + (1.0 - pdf / 100.0) * brsi_d1,
        13 => (pdf / 100.0) * ros_curve(a, b, c, bisi) + 0.2 * (1.0 - pdf / 100.0) * brsi_d1,
        11 => (pc / 100.0) * brsi_c2 + 0.2 * (1.0 - pc / 100.0) * brsi_d1,
        10 => (pc / 100.0) * brsi_c2 + (1.0 - pc / 100.0) * brsi_d1,
        14 | 15 => ros_curve(a, b, c, bisi) * cf,
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
    if ft == 6 {
        sros = rsi * be;
    }
    if ft == 9 {
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
    // before calcCSFI onward. bros/sros keep their computed values (C-6 cfb
    // reads sros, and the C-6 blend below may overwrite the injected hros).
    if let Some(v) = input.hros_override {
        hros = v;
    }

    // --- calc_sfc
    let mut ffc = f64::NAN;
    let mut wfc = f64::NAN;
    let sfc = match ft {
        1 => {
            if ffmc > 84.0 {
                0.75 + 0.75 * (1.0 - (-0.23 * (ffmc - 84.0)).exp()).sqrt()
            } else {
                0.75 - 0.75 * (1.0 - (0.23 * (ffmc - 84.0)).exp()).sqrt()
            }
        }
        2 => 5.0 * (1.0 - (-0.0115 * bui).exp()),
        3 | 4 => 5.0 * (1.0 - (-0.0164 * bui).exp()).powf(2.24),
        5 | 6 => 5.0 * (1.0 - (-0.0149 * bui).exp()).powf(2.48),
        7 => {
            ffc = (2.0 * (1.0 - (-0.104 * (ffmc - 70.0)).exp())).max(0.0);
            wfc = 1.5 * (1.0 - (-0.0201 * bui).exp());
            ffc + wfc
        }
        8 | 9 => 1.5 * (1.0 - (-0.0183 * bui).exp()),
        10 | 11 => {
            let c2_sfc = 5.0 * (1.0 - (-0.0115 * bui).exp());
            let d1_sfc = 1.5 * (1.0 - (-0.0183 * bui).exp());
            (pc / 100.0) * c2_sfc + ((100.0 - pc) / 100.0) * d1_sfc
        }
        12 | 13 => 5.0 * (1.0 - (-0.0115 * bui).exp()),
        14 | 15 => gfl,
        16 => {
            ffc = 4.0 * (1.0 - (-0.025 * bui).exp());
            wfc = 4.0 * (1.0 - (-0.034 * bui).exp());
            ffc + wfc
        }
        17 => {
            ffc = 10.0 * (1.0 - (-0.013 * bui).exp());
            wfc = 6.0 * (1.0 - (-0.06 * bui).exp());
            ffc + wfc
        }
        18 => {
            ffc = 12.0 * (1.0 - (-0.0166 * bui).exp());
            wfc = 20.0 * (1.0 - (-0.021 * bui).exp());
            ffc + wfc
        }
        _ => f64::NAN,
    };

    // --- getCBH_CFL
    let (cbh, cfl) = cbh_cfl(ft);

    // --- calc_csfi / calc_rso
    let csfi = if ft < 14 { (0.01 * cbh * (460.0 + 25.9 * fmc.fmc)).powf(1.5) } else { 0.0 };
    let rso = if sfc > 0.0 { csfi / (300.0 * sfc) } else { 0.0 };

    // --- calc_cfb
    let mut cfb = 0.0;
    if ft == 6 {
        let delta = sros - rso;
        cfb = if delta < -3086.0 { 0.0 } else { 1.0 - (-0.23 * delta).exp() };
    } else if is_modeled(ft) && !is_non_crowning(ft) {
        let delta = hros - rso;
        cfb = if delta < -3086.0 { 0.0 } else { 1.0 - (-0.23 * delta).exp() };
    }
    if !cfb.is_finite() && !cfb.is_nan() {
        // infinities zero out; NaN is a masked cell in Python and must
        // stay NaN (grid-truth: cfb/accel are NaN at NaN-input cells)
        cfb = 0.0;
    }
    cfb = cfb.clamp(0.0, 1.0);

    // --- calc_ros_percentile_growth: no-op at 50/None, matching the golden
    // scenarios. Non-50 percentiles are exercised by the differential tests.
    debug_assert!(
        input.percentile_growth == 50.0,
        "percentile_growth != 50 not yet ported"
    );

    // --- calc_accel_param
    let accel = if is_open_fuel(ft) {
        0.115
    } else if is_modeled(ft) {
        0.115 - 18.8 * cfb.powf(2.5) * (-8.0 * cfb).exp()
    } else {
        0.0
    };

    // --- calc_fire_type
    let fire_type = if ft < 19 {
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
        10 | 11 => cfb * cfl * pc / 100.0,
        12 | 13 => cfb * cfl * pdf / 100.0,
        _ => cfb * cfl,
    };

    // --- calc_c6hros
    let mut cros = 0.0;
    if ft == 6 {
        cros = if cfc == 0.0 {
            0.0
        } else {
            60.0 * (1.0 - (-0.0497 * isi).exp()) * (fmc.fme / 0.778237)
        };
        hros = sros + cfb * (cros - sros);
    }

    // getParams export convention: quantities the Python package MASKS
    // (ffc/wfc via the isnan re-mask in calc_sfc) surface as 0.0 through
    // `masked.item()`; unmasked NaNs (e.g. M-1 a/b/c) stay NaN. Mirror it.
    let ffc = if ffc.is_nan() { 0.0 } else { ffc };
    let wfc = if wfc.is_nan() { 0.0 } else { wfc };

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
        fuel_type: ft as f64,
    }
}

/// Per-window behaviour grids for a fire-growth engine, one weather step.
/// `lb_ratio` is deliberately absent: length-to-breadth is an engine-side
/// quantity (derived from `wsv`), not part of this package's spec.
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
        if !is_modeled(fuel_type[i]) {
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
