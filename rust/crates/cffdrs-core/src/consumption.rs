//! Total fuel consumption, head fire intensity and the intensity class.

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
