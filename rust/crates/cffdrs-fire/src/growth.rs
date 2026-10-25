//! Fire acceleration parameter (`calc_accel_param`). The percentile growth
//! model itself lives in `percentile`.

use crate::fuel::FuelType;

/// `calc_accel_param`: acceleration parameter by fuel class and CFB.
pub(crate) fn calc_accel_param(ft: FuelType, cfb: f64) -> f64 {
    if ft.is_open() {
        0.115
    } else if ft.is_modeled() {
        0.115 - 18.8 * cfb.powf(2.5) * (-8.0 * cfb).exp()
    } else {
        0.0
    }
}
