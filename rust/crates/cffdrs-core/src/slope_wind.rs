//! Slope factor, initial spread index, slope/wind vectoring and the
//! surface-to-ISI/RSI/BE chain (`slope_wind.py` and the facade's
//! `calc_isi_rsi_be`).

use crate::fuel::{FuelType, RosParams};
use crate::normalize::Normalized;

/// `calc_sf`: slope factor.
pub(crate) fn calc_sf(slope: f64) -> f64 {
    // where(slope < 70, exp(...), 10): NaN slope stays masked in Python —
    // propagate it rather than taking the finite cap branch.
    if slope.is_nan() {
        f64::NAN
    } else if slope < 70.0 {
        (3.533 * (slope / 100.0).powf(1.2)).exp()
    } else {
        10.0
    }
}

// Field names mirror the Python `calc_isz` outputs (`isz` included).
#[allow(clippy::struct_field_names)]
pub(crate) struct Isz {
    pub m: f64,
    pub f_f: f64,
    pub isz: f64,
}

/// `calc_isz`: fine-fuel moisture function and zero-wind, zero-slope ISI.
pub(crate) fn calc_isz(ffmc: f64) -> Isz {
    let m = (250.0 * (59.5 / 101.0) * (101.0 - ffmc)) / (59.5 + ffmc);
    let f_f = (91.9 * (-0.1386 * m).exp()) * (1.0 + m.powf(5.31) / (4.93 * 1.0e7));
    let isz = 0.208 * f_f;
    Isz { m, f_f, isz }
}

pub(crate) struct SlopeWindIsi {
    pub(crate) wse1: f64,
    pub(crate) wse2: f64,
    pub(crate) wse: f64,
    pub(crate) wsx: f64,
    pub(crate) wsy: f64,
    pub(crate) wsv: f64,
    pub(crate) raz: f64,
    pub(crate) f_w: f64,
    pub(crate) bfw: f64,
    pub(crate) isi: f64,
    pub(crate) bisi: f64,
}

/// `slope_wind.calc_slope_wind_isi`
pub(crate) fn calc_slope_wind_isi(
    isf: f64,
    f_f: f64,
    wd: f64,
    aspect: f64,
    ws: f64,
) -> SlopeWindIsi {
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
pub(crate) fn ros_curve(a: f64, b: f64, c: f64, x: f64) -> f64 {
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

pub(crate) struct SpreadIndices {
    pub rsz: f64,
    pub rsf: f64,
    pub isf: f64,
    pub sw: SlopeWindIsi,
    pub rsi: f64,
    pub brsi: f64,
    pub be: f64,
}

/// `calc_isi_rsi_be`: ROS at zero wind/slope, slope-equivalent ISF, the
/// slope/wind ISI, RSI/BRSI and the buildup effect.
// Mirrors Python `calc_isi_rsi_be` branch for branch.
#[allow(clippy::too_many_lines)]
pub(crate) fn calc_isi_rsi_be(
    ft: FuelType,
    params: &RosParams,
    n: &Normalized,
    f_f: f64,
    isz: f64,
    sf: f64,
) -> SpreadIndices {
    let Normalized {
        aspect,
        ws,
        wd,
        bui,
        pc,
        pdf,
        gcf,
        ..
    } = *n;
    let RosParams {
        a,
        b,
        c,
        q,
        bui0,
        be_max,
    } = *params;
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

    SpreadIndices {
        rsz,
        rsf,
        isf,
        sw,
        rsi,
        brsi,
        be,
    }
}

#[cfg(test)]
mod tests {
    use super::calc_sf;

    #[test]
    fn sf_caps_at_slope_70() {
        assert_eq!(calc_sf(70.0), 10.0);
        assert_eq!(calc_sf(150.0), 10.0);
        assert_eq!(calc_sf(0.0), 1.0);
    }

    #[test]
    fn sf_propagates_nan() {
        assert!(calc_sf(f64::NAN).is_nan());
    }
}
