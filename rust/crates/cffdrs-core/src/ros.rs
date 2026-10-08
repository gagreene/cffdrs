//! Head/backing/surface rate of spread (`calc_ros`) and the deterministic C-6
//! crown blend.

use crate::fuel::FuelType;

/// Head, backing and surface rates of spread. `sros` is only populated for C-6.
pub(crate) struct Ros {
    pub hros: f64,
    pub bros: f64,
    pub sros: f64,
}

/// Cross-over blend output for C-6: crown ROS and the blended head ROS.
pub(crate) struct C6Blend {
    pub cros: f64,
    pub hros: f64,
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

/// Deterministic C-6 blend: SROS-derived CFB -> CFC -> CROS -> blended HROS.
/// `blend_cfb` is the temporary CFB derived from `sros`; it is not the CFB
/// used downstream.
pub(crate) fn calc_c6_blend(sros: f64, blend_cfb: f64, cfl: f64, isi: f64, fme: f64) -> C6Blend {
    let blend_cfc = blend_cfb * cfl;
    let cros = if blend_cfc == 0.0 {
        0.0
    } else {
        60.0 * (1.0 - (-0.0497 * isi).exp()) * (fme / 0.778_237)
    };
    let hros = sros + blend_cfb * (cros - sros);
    C6Blend { cros, hros }
}
