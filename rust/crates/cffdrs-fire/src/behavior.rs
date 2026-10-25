//! Fire behavior calculated from the other results: fire type, head fire intensity and the intensity class.

use crate::fuel::FuelType;

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

/// `calc_hfi`: head fire intensity.
pub(crate) fn calc_hfi(hros: f64, tfc: f64) -> f64 {
    300.0 * hros * tfc
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- fire tests ----
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
    fn fire_type_nan_and_non_fuel_are_zero() {
        assert_eq!(calc_fire_type(FuelType::C2, f64::NAN), 0.0);
        assert_eq!(calc_fire_type(FuelType::from_code(19), 0.5), 0.0);
        assert_eq!(calc_fire_type(FuelType::from_code(20), 0.95), 0.0);
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
