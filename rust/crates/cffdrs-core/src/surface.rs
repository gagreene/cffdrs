//! Surface fuel consumption (`calc_sfc` and the fine/woody fuel split).

use crate::fuel::FuelType;
use crate::normalize::Normalized;

/// Surface fuel consumption with its fine/woody components. `ffc`/`wfc` stay
/// NaN where the Python package masks them (fuels without a fine/woody split):
/// the GRID path surfaces masked cells as NaN.
pub(crate) struct SurfaceFuel {
    pub sfc: f64,
    pub ffc: f64,
    pub wfc: f64,
}

/// `calc_sfc`: surface fuel consumption by fuel type.
pub(crate) fn calc_sfc(ft: FuelType, n: &Normalized) -> SurfaceFuel {
    let (ffmc, bui, pc, gfl) = (n.ffmc, n.bui, n.pc, n.gfl);
    let mut ffc = f64::NAN;
    let mut wfc = f64::NAN;
    let sfc = match ft {
        FuelType::C1 => {
            if ffmc > 84.0 {
                0.75 + 0.75 * (1.0 - (-0.23 * (ffmc - 84.0)).exp()).sqrt()
            } else {
                0.75 - 0.75 * (1.0 - (0.23 * (ffmc - 84.0)).exp()).sqrt()
            }
        }
        FuelType::C2 | FuelType::M3 | FuelType::M4 => 5.0 * (1.0 - (-0.0115 * bui).exp()),
        FuelType::C3 | FuelType::C4 => 5.0 * (1.0 - (-0.0164 * bui).exp()).powf(2.24),
        FuelType::C5 | FuelType::C6 => 5.0 * (1.0 - (-0.0149 * bui).exp()).powf(2.48),
        FuelType::C7 => {
            // `where(x < 0, 0, x)`: floors at 0 but keeps NaN (f64::max would
            // turn a missing ffmc into 0)
            let raw = 2.0 * (1.0 - (-0.104 * (ffmc - 70.0)).exp());
            ffc = if raw < 0.0 { 0.0 } else { raw };
            wfc = 1.5 * (1.0 - (-0.0201 * bui).exp());
            ffc + wfc
        }
        FuelType::D1 | FuelType::D2 => 1.5 * (1.0 - (-0.0183 * bui).exp()),
        FuelType::M1 | FuelType::M2 => {
            let c2_sfc = 5.0 * (1.0 - (-0.0115 * bui).exp());
            let d1_sfc = 1.5 * (1.0 - (-0.0183 * bui).exp());
            (pc / 100.0) * c2_sfc + ((100.0 - pc) / 100.0) * d1_sfc
        }
        FuelType::O1a | FuelType::O1b => gfl,
        FuelType::S1 => {
            ffc = 4.0 * (1.0 - (-0.025 * bui).exp());
            wfc = 4.0 * (1.0 - (-0.034 * bui).exp());
            ffc + wfc
        }
        FuelType::S2 => {
            ffc = 10.0 * (1.0 - (-0.013 * bui).exp());
            wfc = 6.0 * (1.0 - (-0.06 * bui).exp());
            ffc + wfc
        }
        FuelType::S3 => {
            ffc = 12.0 * (1.0 - (-0.0166 * bui).exp());
            wfc = 20.0 * (1.0 - (-0.021 * bui).exp());
            ffc + wfc
        }
        _ => f64::NAN,
    };
    SurfaceFuel { sfc, ffc, wfc }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalized(ffmc: f64, bui: f64) -> Normalized {
        Normalized {
            slope: 10.0,
            aspect: 270.0,
            ws: 20.0,
            wd: 0.0,
            ffmc,
            bui,
            pc: 50.0,
            pdf: 35.0,
            gfl: 0.35,
            gcf: 80.0,
        }
    }

    #[test]
    fn c7_missing_ffmc_gives_nan_ffc_and_sfc() {
        let s = calc_sfc(FuelType::C7, &normalized(f64::NAN, 76.0));
        assert!(s.ffc.is_nan(), "ffc {}", s.ffc);
        assert!(s.sfc.is_nan(), "sfc {}", s.sfc);
        assert!(s.wfc.is_finite(), "wfc {}", s.wfc);
    }

    #[test]
    fn c7_low_ffmc_floors_ffc_at_zero() {
        let s = calc_sfc(FuelType::C7, &normalized(50.0, 76.0));
        assert_eq!(s.ffc, 0.0);
        assert_eq!(s.sfc, s.wfc);
    }

    #[test]
    fn missing_bui_gives_nan_sfc_for_bui_driven_fuels() {
        for code in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18] {
            let s = calc_sfc(FuelType::from_code(code), &normalized(91.0, f64::NAN));
            assert!(s.sfc.is_nan(), "fuel {code}: sfc {}", s.sfc);
        }
        let c1 = calc_sfc(FuelType::C1, &normalized(91.0, f64::NAN));
        assert!(c1.sfc.is_finite());
    }
}
