//! Input normalization — `inputs._verify_inputs` plus `invert_wind_aspect`.

use crate::fbp::FbpInput;
use crate::field::Field;

/// The clamped / defaulted scalar inputs the rest of the chain reads.
#[derive(Clone, Copy)]
pub(crate) struct Normalized {
    pub slope: f64,
    pub aspect: f64,
    pub ws: f64,
    pub wd: f64,
    pub ffmc: f64,
    pub bui: f64,
    pub pc: f64,
    pub pdf: f64,
    pub gfl: f64,
    pub gcf: f64,
}

/// `invert_wind_aspect`: flip wind direction and aspect by 180 degrees.
pub(crate) fn invert_wind_aspect(wd: f64, aspect: f64) -> (f64, f64) {
    let wd = if wd > 180.0 { wd - 180.0 } else { wd + 180.0 };
    let aspect = if aspect > 180.0 {
        aspect - 180.0
    } else {
        aspect + 180.0
    };
    (wd, aspect)
}

/// `inputs._verify_inputs` normalization.
pub(crate) fn normalize(input: &FbpInput) -> Normalized {
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
    let wd = input.wd;
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
    Normalized {
        slope,
        aspect,
        ws,
        wd,
        ffmc,
        bui,
        pc,
        pdf,
        gfl,
        gcf,
    }
}

/// The normalisation [`normalize`] gives one input field, for inputs stored
/// one field at a time (`long` is stored as its magnitude, as the stages read
/// it). Fields `normalize` passes through unchanged (`lat`, `elevation`, the
/// wind direction, ...) are returned as given. Idempotent.
pub(crate) fn normalize_input(field: Field, value: f64) -> f64 {
    // the same expressions as `normalize`, one field at a time
    let clamp0 = |x: f64| if x < 0.0 { 0.0 } else { x };
    match field {
        Field::Slope
        | Field::Ws
        | Field::Ffmc
        | Field::Bui
        | Field::Pc
        | Field::Pdf
        | Field::Gfl => clamp0(value),
        Field::AspectIn | Field::Aspect => {
            if value < 0.0 {
                270.0
            } else {
                value
            }
        }
        Field::Gcf => {
            if value == 0.0 {
                0.1
            } else {
                value
            }
        }
        Field::Long => value.abs(),
        _ => value,
    }
}

#[cfg(test)]
mod tests {
    use super::{invert_wind_aspect, normalize, normalize_input};
    use crate::fbp::FbpInput;
    use crate::field::Field;

    // ---- invert tests ----
    #[test]
    fn invert_at_180_boundary() {
        // exactly 180 is not "> 180", so it adds 180
        assert_eq!(invert_wind_aspect(180.0, 180.0), (360.0, 360.0));
        assert_eq!(invert_wind_aspect(180.5, 270.0), (0.5, 90.0));
        assert_eq!(invert_wind_aspect(0.0, 90.0), (180.0, 270.0));
    }

    #[test]
    fn invert_propagates_nan() {
        let (wd, asp) = invert_wind_aspect(f64::NAN, f64::NAN);
        assert!(wd.is_nan() && asp.is_nan());
    }

    // ---- other tests ----
    #[test]
    fn normalize_input_equals_normalize_field_by_field_and_is_idempotent() {
        let values = [
            f64::NAN,
            f64::NEG_INFINITY,
            -400.0,
            -1.0,
            -0.0,
            0.0,
            0.05,
            0.1,
            1.0,
            180.0,
            270.0,
            365.5,
            f64::INFINITY,
        ];
        let same = |a: f64, b: f64| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
        for v in values {
            let mut input = FbpInput::new(2, 20_230_615);
            (input.slope_pct, input.aspect_deg, input.ws, input.wd) = (v, v, v, v);
            (input.ffmc, input.bui, input.pc, input.pdf) = (v, v, v, v);
            (input.gfl, input.gcf, input.long, input.lat) = (v, v, v, v);
            let n = normalize(&input);
            for (f, want) in [
                (Field::Slope, n.slope),
                (Field::AspectIn, n.aspect),
                (Field::Ws, n.ws),
                (Field::WdIn, n.wd),
                (Field::Ffmc, n.ffmc),
                (Field::Bui, n.bui),
                (Field::Pc, n.pc),
                (Field::Pdf, n.pdf),
                (Field::Gfl, n.gfl),
                (Field::Gcf, n.gcf),
                (Field::Long, input.long.abs()),
                (Field::Lat, input.lat),
            ] {
                let got = normalize_input(f, v);
                assert!(same(got, want), "{f:?}({v}): {got} vs {want}");
                assert!(
                    same(normalize_input(f, got), got),
                    "{f:?}({v}) not idempotent"
                );
            }
        }
    }
}
