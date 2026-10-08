//! Crown fire initiation and fraction burned: CSFI, RSO, CFB, fire type and
//! crown fuel consumption.

use crate::fuel::FuelType;

/// `calc_csfi`: critical surface fire intensity.
pub(crate) fn calc_csfi(ft: FuelType, cbh: f64, fmc: f64) -> f64 {
    // Unknown negative codes take this branch on purpose (decision D3, see
    // `FuelType::has_crown_initiation`); the oracle pins it.
    if ft.has_crown_initiation() {
        (0.01 * cbh * (460.0 + 25.9 * fmc)).powf(1.5)
    } else {
        0.0
    }
}

/// `calc_rso`: critical surface spread rate for crowning. A NaN `sfc` (a
/// missing input upstream) gives NaN, as Python's masked `where` does.
pub(crate) fn calc_rso(sfc: f64, csfi: f64) -> f64 {
    if sfc.is_nan() {
        f64::NAN
    } else if sfc > 0.0 {
        csfi / (300.0 * sfc)
    } else {
        0.0
    }
}

/// Crown fraction burned. The equation is the same for every crowning fuel
/// (including C-6); only the ROS it is applied to differs per step.
pub(crate) fn cfb_from_ros(ros: f64, rso: f64) -> f64 {
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
}

/// CFB for a directional ROS: zero for fuels that cannot crown.
pub(crate) fn directional_cfb(ft: FuelType, ros: f64, rso: f64) -> f64 {
    let crowning = ft.is_modeled() && !ft.is_non_crowning();
    if crowning {
        cfb_from_ros(ros, rso)
    } else {
        0.0
    }
}

/// Final CFB from the percentile-adjusted head ROS. A NaN that appears
/// only at the percentile step from a NaN percentile is an unmasked
/// non-finite value in Python, which the CFB sanitiser zeroes. A NaN that was
/// already there, or that comes from a missing `rso` (a masked regime CFB
/// masks the percentile result), is a masked cell and stays NaN.
pub(crate) fn final_cfb(ft: FuelType, hros: f64, hros_before_percentile: f64, rso: f64) -> f64 {
    if hros.is_nan() && !hros_before_percentile.is_nan() && !rso.is_nan() {
        0.0
    } else {
        directional_cfb(ft, hros, rso)
    }
}

/// `calc_fire_type`: 1 surface, 2 passive crown, 3 active crown.
pub(crate) fn calc_fire_type(ft: FuelType, cfb: f64) -> f64 {
    // Unknown negative codes take this branch on purpose (decision D3, see
    // `FuelType::is_fuel`); the oracle pins it.
    if ft.is_fuel() {
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
    }
}

/// `calc_cfc`: crown fuel consumption.
pub(crate) fn calc_cfc(ft: FuelType, cfb: f64, cfl: f64, pc: f64, pdf: f64) -> f64 {
    match ft {
        FuelType::M1 | FuelType::M2 => cfb * cfl * pc / 100.0,
        FuelType::M3 | FuelType::M4 => cfb * cfl * pdf / 100.0,
        _ => cfb * cfl,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuel::FuelType;

    #[test]
    fn fire_type_boundaries() {
        let ft = FuelType::C2;
        assert_eq!(calc_fire_type(ft, 0.0), 1.0);
        assert_eq!(calc_fire_type(ft, 0.1), 1.0);
        assert_eq!(calc_fire_type(ft, 0.1001), 2.0);
        assert_eq!(calc_fire_type(ft, 0.8999), 2.0);
        assert_eq!(calc_fire_type(ft, 0.9), 3.0);
        assert_eq!(calc_fire_type(ft, 1.0), 3.0);
    }

    #[test]
    fn rso_propagates_missing_sfc() {
        // Python: mask.where(sfc > 0, ...) keeps a masked sfc masked
        assert!(calc_rso(f64::NAN, 500.0).is_nan());
        assert_eq!(calc_rso(0.0, 500.0), 0.0);
        assert_eq!(calc_rso(2.0, 600.0), 1.0);
    }

    #[test]
    fn final_cfb_keeps_a_missing_rso_missing_after_the_percentile_step() {
        // A NaN hros after the percentile step is masked in Python when the
        // regime CFB was masked (missing rso): the final CFB stays NaN for
        // crowning fuels and 0 for fuels that cannot crown.
        assert!(final_cfb(FuelType::C2, f64::NAN, 10.0, f64::NAN).is_nan());
        assert_eq!(final_cfb(FuelType::D1, f64::NAN, 10.0, f64::NAN), 0.0);
        // A NaN percentile with a finite rso is an unmasked NaN: zeroed.
        assert_eq!(final_cfb(FuelType::C2, f64::NAN, 10.0, 2.0), 0.0);
    }

    #[test]
    fn fire_type_nan_and_non_fuel_are_zero() {
        assert_eq!(calc_fire_type(FuelType::C2, f64::NAN), 0.0);
        assert_eq!(calc_fire_type(FuelType::from_code(19), 0.5), 0.0);
        assert_eq!(calc_fire_type(FuelType::from_code(20), 0.95), 0.0);
    }
}
