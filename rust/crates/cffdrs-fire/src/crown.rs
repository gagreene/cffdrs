//! Crown fire determination: CSFI, RSO and CFB.

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

/// Final CFB from the percentile-adjusted head ROS. `growth_made_nan` says
/// that the percentile step itself turned a non-NaN head ROS into NaN (a NaN
/// percentile): that is an unmasked non-finite value in Python, which the CFB
/// sanitiser zeroes. A NaN that was already there (or was injected), or that
/// comes from a missing `rso` (a masked regime CFB masks the percentile
/// result), is a masked cell and stays NaN.
pub(crate) fn final_cfb(ft: FuelType, hros: f64, growth_made_nan: bool, rso: f64) -> f64 {
    if growth_made_nan && hros.is_nan() && !rso.is_nan() {
        0.0
    } else {
        directional_cfb(ft, hros, rso)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuel::FuelType;

    // ---- final tests ----
    #[test]
    fn final_cfb_keeps_a_missing_rso_missing_after_the_percentile_step() {
        // A NaN hros after the percentile step is masked in Python when the
        // regime CFB was masked (missing rso): the final CFB stays NaN for
        // crowning fuels and 0 for fuels that cannot crown.
        assert!(final_cfb(FuelType::C2, f64::NAN, true, f64::NAN).is_nan());
        assert_eq!(final_cfb(FuelType::D1, f64::NAN, true, f64::NAN), 0.0);
    }

    #[test]
    fn final_cfb_zeroes_a_nan_made_by_the_percentile_step() {
        // A NaN percentile with a finite rso is an unmasked NaN: zeroed.
        assert_eq!(final_cfb(FuelType::C2, f64::NAN, true, 2.0), 0.0);
    }

    #[test]
    fn final_cfb_keeps_a_nan_head_ros_the_percentile_step_did_not_make() {
        // An injected NaN hros (or one already NaN before the step) is a
        // masked cell: NaN for crowning fuels.
        assert!(final_cfb(FuelType::C2, f64::NAN, false, 2.0).is_nan());
        assert_eq!(final_cfb(FuelType::D1, f64::NAN, false, 2.0), 0.0);
    }

    #[test]
    fn final_cfb_without_the_flag_is_the_directional_cfb() {
        for (ft, hros, rso) in [
            (FuelType::C2, 12.0, 2.0),
            (FuelType::C2, 1.0, 9.0),
            (FuelType::C6, 30.0, 4.0),
            (FuelType::D1, 12.0, 2.0),
        ] {
            assert_eq!(
                final_cfb(ft, hros, false, rso).to_bits(),
                directional_cfb(ft, hros, rso).to_bits()
            );
        }
    }

    // ---- other tests ----
    #[test]
    fn rso_propagates_missing_sfc() {
        // Python: mask.where(sfc > 0, ...) keeps a masked sfc masked
        assert!(calc_rso(f64::NAN, 500.0).is_nan());
        assert_eq!(calc_rso(0.0, 500.0), 0.0);
        assert_eq!(calc_rso(2.0, 600.0), 1.0);
    }
}
