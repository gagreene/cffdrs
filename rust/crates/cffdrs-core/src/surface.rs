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
        FuelType::C2 => 5.0 * (1.0 - (-0.0115 * bui).exp()),
        FuelType::C3 | FuelType::C4 => 5.0 * (1.0 - (-0.0164 * bui).exp()).powf(2.24),
        FuelType::C5 | FuelType::C6 => 5.0 * (1.0 - (-0.0149 * bui).exp()).powf(2.48),
        FuelType::C7 => {
            ffc = (2.0 * (1.0 - (-0.104 * (ffmc - 70.0)).exp())).max(0.0);
            wfc = 1.5 * (1.0 - (-0.0201 * bui).exp());
            ffc + wfc
        }
        FuelType::D1 | FuelType::D2 => 1.5 * (1.0 - (-0.0183 * bui).exp()),
        FuelType::M1 | FuelType::M2 => {
            let c2_sfc = 5.0 * (1.0 - (-0.0115 * bui).exp());
            let d1_sfc = 1.5 * (1.0 - (-0.0183 * bui).exp());
            (pc / 100.0) * c2_sfc + ((100.0 - pc) / 100.0) * d1_sfc
        }
        FuelType::M3 | FuelType::M4 => 5.0 * (1.0 - (-0.0115 * bui).exp()),
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
