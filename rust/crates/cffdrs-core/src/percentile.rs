//! ROS percentile growth (Han & Braun 2014), mirroring
//! `cffdrs.cffbps.equations.growth.calc_ros_percentile_growth`.
//!
//! `percentile_growth` is a percentile of the ROS distribution (model error
//! only), not a percent change: at percentile `p`, about `(100 - p)%` of fires
//! under the same inputs spread at least as fast. 50 is the unadjusted ROS.
//!
//! Surface-regime ROS is scaled by `exp(z * sigma_s)`; crown-regime ROS uses the
//! Box-Cox power law `(R^0.6 + z * sigma_c)^(1/0.6)`. Both sigmas are the
//! paper's pooled conifer estimates. Project choices (not from the paper):
//! the `cfb < 0.1` regime rule, the zero result for a negative crown radicand,
//! the C-1..C-7 fuel scope, and the percentile cap.

use crate::fuel::FuelType;
use crate::quantile::t_quantile_large_df;

const SURFACE_SIGMA: f64 = 0.923;
const CROWN_SIGMA: f64 = 1.637;
const CROWN_DELTA: f64 = 0.6;

/// Percentiles outside this range are capped so 0 and 100 give finite ROS.
pub const MIN_PERCENTILE: f64 = 0.001;
pub const MAX_PERCENTILE: f64 = 99.999;

/// Degrees of freedom used by the Python reference's Student-t quantile.
const T_DF: f64 = 9_999_999.0;

/// Wind-speed decay `k(w)` applied to the backing-fire noise term.
pub fn wind_decay(wsv: f64) -> f64 {
    if wsv < 40.0 {
        (-0.10078 * wsv).exp()
    } else {
        (-0.05039 * wsv).exp() / (12.0 * (1.0 - (-0.0818 * (wsv - 28.0)).exp()))
    }
}

/// Student-t quantile for a percentile in 0-100, capped to
/// `[MIN_PERCENTILE, MAX_PERCENTILE]`. NaN stays NaN.
pub fn percentile_tinv(percentile_growth: f64) -> f64 {
    let capped = if percentile_growth.is_nan() {
        f64::NAN
    } else {
        percentile_growth.clamp(MIN_PERCENTILE, MAX_PERCENTILE)
    };
    t_quantile_large_df(capped / 100.0, T_DF)
}

/// Adjust one directional ROS. `regime_cfb` is that direction's pre-percentile
/// CFB; `noise_scale` is 1.0 for head fire and `wind_decay(wsv)` for backing.
pub fn percentile_ros(
    fuel_type: FuelType,
    ros: f64,
    regime_cfb: f64,
    tinv: f64,
    noise_scale: f64,
) -> f64 {
    if regime_cfb < 0.1 {
        if fuel_type.has_surface_regime() {
            ros * (tinv * SURFACE_SIGMA * noise_scale).exp()
        } else {
            ros
        }
    } else if fuel_type.has_crown_regime() {
        let radicand = ros.powf(CROWN_DELTA) + tinv * CROWN_SIGMA * noise_scale;
        if radicand.is_nan() {
            f64::NAN
        } else if radicand < 0.0 {
            0.0
        } else {
            radicand.powf(1.0 / CROWN_DELTA)
        }
    } else {
        ros
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_tinv_is_zero() {
        assert_eq!(percentile_tinv(50.0), 0.0);
    }

    #[test]
    fn tails_are_capped_to_the_bounds() {
        let (lo, hi) = (
            percentile_tinv(MIN_PERCENTILE),
            percentile_tinv(MAX_PERCENTILE),
        );
        for p in [0.0, -5.0, -1e9] {
            assert_eq!(percentile_tinv(p), lo, "p={p}");
        }
        for p in [100.0, 150.0, 1e9] {
            assert_eq!(percentile_tinv(p), hi, "p={p}");
        }
        assert!(lo.is_finite() && hi.is_finite());
    }

    #[test]
    fn nan_percentile_propagates() {
        let tinv = percentile_tinv(f64::NAN);
        assert!(tinv.is_nan());
        assert!(percentile_ros(FuelType::from_code(2), 7.5, 0.0, tinv, 1.0).is_nan());
        assert!(percentile_ros(FuelType::from_code(2), 7.5, 0.9, tinv, 1.0).is_nan());
    }

    #[test]
    fn crown_result_is_non_decreasing_in_ros_at_low_percentiles() {
        // C-2, percentile 5: the old fallback jumped up just below the
        // radicand-zero threshold (R=5.2 -> 0.352, R=5.3 -> 0.0025).
        let tinv = percentile_tinv(5.0);
        let mut prev = -1.0;
        for i in 0..=2000 {
            let ros = i as f64 * 0.005;
            let out = percentile_ros(FuelType::from_code(2), ros, 0.9, tinv, 1.0);
            assert!(out >= prev - 1e-12, "ros={ros}: {out} < {prev}");
            prev = out;
        }
        assert_eq!(
            percentile_ros(FuelType::from_code(2), 5.2, 0.9, tinv, 1.0),
            0.0
        );
    }

    #[test]
    fn out_of_scope_fuels_are_unchanged_in_both_regimes() {
        let tinv = percentile_tinv(90.0);
        for fuel in [8, 12, 14, 19, 20] {
            for cfb in [0.0, 0.9] {
                assert_eq!(
                    percentile_ros(FuelType::from_code(fuel), 5.0, cfb, tinv, 1.0),
                    5.0,
                    "fuel {fuel} cfb {cfb}"
                );
            }
        }
    }

    #[test]
    fn c1_is_crown_only_and_c5_is_surface_only() {
        let tinv = percentile_tinv(90.0);
        assert_eq!(
            percentile_ros(FuelType::from_code(1), 5.0, 0.0, tinv, 1.0),
            5.0
        );
        assert_ne!(
            percentile_ros(FuelType::from_code(1), 5.0, 0.9, tinv, 1.0),
            5.0
        );
        assert_ne!(
            percentile_ros(FuelType::from_code(5), 5.0, 0.0, tinv, 1.0),
            5.0
        );
        assert_eq!(
            percentile_ros(FuelType::from_code(5), 5.0, 0.9, tinv, 1.0),
            5.0
        );
    }
}
