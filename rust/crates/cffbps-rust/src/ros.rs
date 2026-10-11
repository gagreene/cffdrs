//! Head/backing/surface rate of spread (`calc_ros`) and the crown and head ROS
//! of the deterministic C-6 blend.

use crate::fuel::FuelType;

/// Head, backing and surface rates of spread. `sros` is only populated for C-6.
pub(crate) struct Ros {
    pub hros: f64,
    pub bros: f64,
    pub sros: f64,
}

/// `calc_ros`, including the setParams({'hros': ...}) injection.
pub(crate) fn calc_ros(
    ft: FuelType,
    rsi: f64,
    brsi: f64,
    be: f64,
    bui: f64,
    hros_override: Option<f64>,
) -> Ros {
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
    if let Some(v) = hros_override {
        hros = v;
    }
    Ros { hros, bros, sros }
}

// The deterministic C-6 blend is four stages: SROS-derived CFB
// (`crown::cfb_from_ros(sros, rso)`) -> CFC (`crown::calc_cfc`, which is
// `blend_cfb * cfl` for C-6) -> CROS (`c6_cros`) -> blended HROS (`c6_hros`).
// The blend CFB is temporary; it is not the CFB used downstream.

/// `calc_c6_cros`: C-6 crown ROS from the blend CFC; zero when the blend
/// consumes no crown fuel.
pub(crate) fn c6_cros(blend_cfc: f64, isi: f64, fme: f64) -> f64 {
    if blend_cfc == 0.0 {
        0.0
    } else {
        60.0 * (1.0 - (-0.0497 * isi).exp()) * (fme / 0.778_237)
    }
}

/// `calc_c6_hros`: C-6 head ROS blended from surface and crown ROS by the
/// blend CFB.
pub(crate) fn c6_hros(sros: f64, blend_cfb: f64, cros: f64) -> f64 {
    sros + blend_cfb * (cros - sros)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crown::{calc_cfc, cfb_from_ros};

    /// The four C-6 stage functions composed in pipeline order.
    fn composed(sros: f64, rso: f64, cfl: f64, isi: f64, fme: f64) -> (f64, f64) {
        let blend_cfb = cfb_from_ros(sros, rso);
        let blend_cfc = calc_cfc(FuelType::C6, blend_cfb, cfl, 50.0, 35.0);
        let cros = c6_cros(blend_cfc, isi, fme);
        (cros, c6_hros(sros, blend_cfb, cros))
    }

    /// The single-function blend this split replaced, verbatim.
    fn old_blend(sros: f64, rso: f64, cfl: f64, isi: f64, fme: f64) -> (f64, f64) {
        let blend_cfb = cfb_from_ros(sros, rso);
        let blend_cfc = blend_cfb * cfl;
        let cros = if blend_cfc == 0.0 {
            0.0
        } else {
            60.0 * (1.0 - (-0.0497 * isi).exp()) * (fme / 0.778_237)
        };
        (cros, sros + blend_cfb * (cros - sros))
    }

    #[test]
    fn the_four_stage_functions_equal_the_old_blend_bit_for_bit() {
        let nan = f64::NAN;
        for sros in [0.0, 0.5, 3.0, 12.0, 40.0, nan] {
            for rso in [0.0, 2.0, 9.0, nan] {
                for cfl in [0.0, 1.8, nan] {
                    for (isi, fme) in [(0.0, 0.0), (8.0, 0.9), (25.0, 1.1), (nan, 0.9), (8.0, nan)]
                    {
                        let (c1, h1) = composed(sros, rso, cfl, isi, fme);
                        let (c2, h2) = old_blend(sros, rso, cfl, isi, fme);
                        let same = |a: f64, b: f64| {
                            a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
                        };
                        assert!(same(c1, c2), "cros {sros} {rso} {cfl} {isi} {fme}");
                        assert!(same(h1, h2), "hros {sros} {rso} {cfl} {isi} {fme}");
                    }
                }
            }
        }
    }

    #[test]
    fn c6_cros_is_zero_without_blend_crown_consumption() {
        assert_eq!(c6_cros(0.0, 20.0, 1.0), 0.0);
        assert!(c6_cros(0.5, 20.0, 1.0) > 0.0);
    }

    #[test]
    fn c6_hros_blends_surface_and_crown_by_the_blend_cfb() {
        assert_eq!(c6_hros(4.0, 0.0, 30.0), 4.0);
        assert_eq!(c6_hros(4.0, 1.0, 30.0), 30.0);
        assert_eq!(c6_hros(4.0, 0.5, 30.0), 17.0);
    }
}
