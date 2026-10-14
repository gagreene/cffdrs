//! Total fuel consumption, head fire intensity and the intensity class.

/// Fuel consumption that follows whether the fire spreads: 0 where the head
/// rate of spread is exactly 0, NaN where it is NaN, otherwise `value`
/// unchanged. Applied in `calcTFC` (after the crown calculations used `sfc`) to
/// `ffc`, `wfc`, `sfc` and `cfc`; mirrors `consumption.zero_without_spread`.
pub(crate) fn without_spread(hros: f64, value: f64) -> f64 {
    if hros == 0.0 {
        0.0
    } else if hros.is_nan() {
        f64::NAN
    } else {
        value
    }
}

/// `calc_tfc`: total fuel consumption.
pub(crate) fn calc_tfc(sfc: f64, cfc: f64) -> f64 {
    sfc + cfc
}

/// `calc_hfi`: head fire intensity.
pub(crate) fn calc_hfi(hros: f64, tfc: f64) -> f64 {
    300.0 * hros * tfc
}

/// `calc_fire_intensity_class`.
pub(crate) fn calc_fire_intensity_class(hfi: f64) -> f64 {
    if hfi > 10000.0 {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_spread_consumes_no_fuel_and_a_missing_rate_gives_missing_consumption() {
        assert_eq!(without_spread(0.0, 1.5), 0.0);
        assert_eq!(without_spread(0.0, f64::NAN), 0.0);
        assert!(without_spread(f64::NAN, 1.5).is_nan());
        assert_eq!(without_spread(2.0, 1.5), 1.5);
        assert!(without_spread(2.0, f64::NAN).is_nan());
    }

    #[test]
    fn fire_intensity_class_boundaries() {
        let cases = [
            (0.0, -99.0),
            (1e-9, 1.0),
            (10.0, 1.0),
            (10.0001, 2.0),
            (500.0, 2.0),
            (500.0001, 3.0),
            (2000.0, 3.0),
            (2000.0001, 4.0),
            (4000.0, 4.0),
            (4000.0001, 5.0),
            (10000.0, 5.0),
            (10000.0001, 6.0),
            (-1.0, -99.0),
        ];
        for (hfi, want) in cases {
            assert_eq!(calc_fire_intensity_class(hfi), want, "hfi = {hfi}");
        }
    }

    #[test]
    fn fire_intensity_class_nan_is_current_sentinel() {
        // Pins current behaviour: every comparison is false for NaN.
        assert_eq!(calc_fire_intensity_class(f64::NAN), -99.0);
    }
}
