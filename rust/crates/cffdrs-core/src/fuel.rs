//! CFFBPS fuel types and their per-fuel parameter tables.

/// CFFBPS fuel type (mirrors the numeric codes in the Python `constants`
/// module). Codes 1-18 are modeled fuels; 19 and 20 are non-fuel and
/// water; any other code is carried as `Unknown` so behaviour is unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FuelType {
    /// C-1 Spruce-lichen woodland (code 1).
    C1,
    /// C-2 Boreal spruce (code 2).
    C2,
    /// C-3 Mature jack or lodgepole pine (code 3).
    C3,
    /// C-4 Immature jack or lodgepole pine (code 4).
    C4,
    /// C-5 Red and white pine (code 5).
    C5,
    /// C-6 Conifer plantation (code 6).
    C6,
    /// C-7 Ponderosa pine / Douglas-fir (code 7).
    C7,
    /// D-1 Leafless aspen (code 8).
    D1,
    /// D-2 Green aspen (code 9).
    D2,
    /// M-1 Boreal mixedwood, leafless (code 10).
    M1,
    /// M-2 Boreal mixedwood, green (code 11).
    M2,
    /// M-3 Dead balsam fir mixedwood, leafless (code 12).
    M3,
    /// M-4 Dead balsam fir mixedwood, green (code 13).
    M4,
    /// O-1a Matted grass (code 14).
    O1a,
    /// O-1b Standing grass (code 15).
    O1b,
    /// S-1 Jack or lodgepole pine slash (code 16).
    S1,
    /// S-2 White spruce / balsam slash (code 17).
    S2,
    /// S-3 Coastal cedar / hemlock / Douglas-fir slash (code 18).
    S3,
    /// Non-fuel (code 19).
    NonFuel,
    /// Water (code 20).
    Water,
    /// Any code outside 1..=20, preserved verbatim.
    ///
    /// Build this variant through [`FuelType::from_code`] rather than by hand:
    /// `from_code` is the canonical constructor, so a hand-built
    /// `Unknown(5)` (a valid code that has its own variant) is non-canonical
    /// and would not compare equal to [`FuelType::C5`].
    Unknown(i32),
}

/// Surface ROS parameters (a, b, c, q, bui0, be_max) — `constants.rosParams`.
/// `None` entries in the Python table surface as NaN, matching numpy
/// assignment semantics.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RosParams {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub q: f64,
    pub bui0: f64,
    pub be_max: f64,
}

/// Crown base height and foliar load — `constants.fbpCBH_CFL_HT_LUT`
/// (height column unused here).
#[derive(Debug, Clone, Copy)]
pub(crate) struct CrownFuel {
    pub cbh: f64,
    pub cfl: f64,
}

const fn rp(a: f64, b: f64, c: f64, q: f64, bui0: f64, be_max: f64) -> RosParams {
    RosParams {
        a,
        b,
        c,
        q,
        bui0,
        be_max,
    }
}

impl FuelType {
    /// Convert an integer fuel code to a `FuelType`. Total: codes outside
    /// 1..=20 become `Unknown(code)`.
    pub const fn from_code(code: i32) -> Self {
        match code {
            1 => Self::C1,
            2 => Self::C2,
            3 => Self::C3,
            4 => Self::C4,
            5 => Self::C5,
            6 => Self::C6,
            7 => Self::C7,
            8 => Self::D1,
            9 => Self::D2,
            10 => Self::M1,
            11 => Self::M2,
            12 => Self::M3,
            13 => Self::M4,
            14 => Self::O1a,
            15 => Self::O1b,
            16 => Self::S1,
            17 => Self::S2,
            18 => Self::S3,
            19 => Self::NonFuel,
            20 => Self::Water,
            other => Self::Unknown(other),
        }
    }

    /// The integer fuel code; the inverse of [`FuelType::from_code`].
    pub const fn code(self) -> i32 {
        match self {
            Self::C1 => 1,
            Self::C2 => 2,
            Self::C3 => 3,
            Self::C4 => 4,
            Self::C5 => 5,
            Self::C6 => 6,
            Self::C7 => 7,
            Self::D1 => 8,
            Self::D2 => 9,
            Self::M1 => 10,
            Self::M2 => 11,
            Self::M3 => 12,
            Self::M4 => 13,
            Self::O1a => 14,
            Self::O1b => 15,
            Self::S1 => 16,
            Self::S2 => 17,
            Self::S3 => 18,
            Self::NonFuel => 19,
            Self::Water => 20,
            Self::Unknown(code) => code,
        }
    }

    /// True for the modeled fuels (codes 1..=18).
    pub const fn is_modeled(self) -> bool {
        matches!(self.code(), 1..=18)
    }

    /// True for open fuels (`constants.open_fuel_types`).
    pub const fn is_open(self) -> bool {
        matches!(
            self,
            Self::C1 | Self::C7 | Self::D2 | Self::O1a | Self::O1b | Self::S1 | Self::S2 | Self::S3
        )
    }

    /// True for fuels that never crown (`constants.non_crowning_fuels`).
    pub const fn is_non_crowning(self) -> bool {
        matches!(
            self,
            Self::D1 | Self::D2 | Self::O1a | Self::O1b | Self::S1 | Self::S2 | Self::S3
        )
    }

    /// Surface-regime fuels for percentile growth (C-2..C-7).
    pub(crate) const fn has_surface_regime(self) -> bool {
        matches!(
            self,
            Self::C2 | Self::C3 | Self::C4 | Self::C5 | Self::C6 | Self::C7
        )
    }

    /// Crown-regime fuels for percentile growth (C-1..C-4, C-6, C-7).
    pub(crate) const fn has_crown_regime(self) -> bool {
        matches!(
            self,
            Self::C1 | Self::C2 | Self::C3 | Self::C4 | Self::C6 | Self::C7
        )
    }

    /// Surface ROS parameters; codes without a table entry fall back to
    /// `(0, 0, 0, 0, 1, 1)`, as `ros_params.get(ftype, (0, 0, 0, 0, 1, 1))`.
    pub(crate) fn ros_params(self) -> RosParams {
        const NAN: f64 = f64::NAN;
        match self {
            Self::C1 => rp(90.0, 0.0649, 4.5, 0.9, 72.0, 1.076),
            Self::C2 => rp(110.0, 0.0282, 1.5, 0.7, 64.0, 1.321),
            Self::C3 => rp(110.0, 0.0444, 3.0, 0.75, 62.0, 1.261),
            Self::C4 => rp(110.0, 0.0293, 1.5, 0.8, 66.0, 1.184),
            Self::C5 => rp(30.0, 0.0697, 4.0, 0.8, 56.0, 1.220),
            Self::C6 => rp(30.0, 0.08, 3.0, 0.8, 62.0, 1.197),
            Self::C7 => rp(45.0, 0.0305, 2.0, 0.85, 106.0, 1.134),
            Self::D1 => rp(30.0, 0.0232, 1.6, 0.9, 32.0, 1.179),
            Self::D2 => rp(30.0, 0.0232, 1.6, 0.9, 32.0, 1.179),
            Self::M1 => rp(NAN, NAN, NAN, 0.8, 50.0, 1.250),
            Self::M2 => rp(NAN, NAN, NAN, 0.8, 50.0, 1.250),
            Self::M3 => rp(120.0, 0.0572, 1.4, 0.8, 50.0, 1.250),
            Self::M4 => rp(100.0, 0.0404, 1.48, 0.8, 50.0, 1.250),
            Self::O1a => rp(190.0, 0.0310, 1.4, 1.0, NAN, 1.0),
            Self::O1b => rp(250.0, 0.0350, 1.7, 1.0, NAN, 1.0),
            Self::S1 => rp(75.0, 0.0297, 1.3, 0.75, 38.0, 1.460),
            Self::S2 => rp(40.0, 0.0438, 1.7, 0.75, 63.0, 1.256),
            Self::S3 => rp(55.0, 0.0829, 3.2, 0.75, 31.0, 1.590),
            Self::NonFuel | Self::Water | Self::Unknown(_) => rp(0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
        }
    }

    /// Crown base height and foliar load; NaN for fuels without an entry.
    pub(crate) fn crown_fuel(self) -> CrownFuel {
        let (cbh, cfl) = match self {
            Self::C1 => (2.0, 0.75),
            Self::C2 => (3.0, 0.8),
            Self::C3 => (8.0, 1.15),
            Self::C4 => (4.0, 1.2),
            Self::C5 => (18.0, 1.2),
            Self::C6 => (7.0, 1.8),
            Self::C7 => (10.0, 0.5),
            Self::D1 | Self::D2 | Self::O1a | Self::O1b | Self::S1 | Self::S2 | Self::S3 => {
                (0.0, 0.0)
            }
            Self::M1 | Self::M2 | Self::M3 | Self::M4 => (6.0, 0.8),
            Self::NonFuel | Self::Water | Self::Unknown(_) => (f64::NAN, f64::NAN),
        };
        CrownFuel { cbh, cfl }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verbatim copies of the pre-refactor integer logic from fbp.rs, kept as
    // the reference the enum must reproduce for every code.

    fn old_ros_params(fuel_type: i32) -> (f64, f64, f64, f64, f64, f64) {
        const NAN: f64 = f64::NAN;
        match fuel_type {
            1 => (90.0, 0.0649, 4.5, 0.9, 72.0, 1.076),
            2 => (110.0, 0.0282, 1.5, 0.7, 64.0, 1.321),
            3 => (110.0, 0.0444, 3.0, 0.75, 62.0, 1.261),
            4 => (110.0, 0.0293, 1.5, 0.8, 66.0, 1.184),
            5 => (30.0, 0.0697, 4.0, 0.8, 56.0, 1.220),
            6 => (30.0, 0.08, 3.0, 0.8, 62.0, 1.197),
            7 => (45.0, 0.0305, 2.0, 0.85, 106.0, 1.134),
            8 => (30.0, 0.0232, 1.6, 0.9, 32.0, 1.179),
            9 => (30.0, 0.0232, 1.6, 0.9, 32.0, 1.179),
            10 => (NAN, NAN, NAN, 0.8, 50.0, 1.250),
            11 => (NAN, NAN, NAN, 0.8, 50.0, 1.250),
            12 => (120.0, 0.0572, 1.4, 0.8, 50.0, 1.250),
            13 => (100.0, 0.0404, 1.48, 0.8, 50.0, 1.250),
            14 => (190.0, 0.0310, 1.4, 1.0, NAN, 1.0),
            15 => (250.0, 0.0350, 1.7, 1.0, NAN, 1.0),
            16 => (75.0, 0.0297, 1.3, 0.75, 38.0, 1.460),
            17 => (40.0, 0.0438, 1.7, 0.75, 63.0, 1.256),
            18 => (55.0, 0.0829, 3.2, 0.75, 31.0, 1.590),
            // ros_params.get(ftype, (0, 0, 0, 0, 1, 1)) fallback
            _ => (0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
        }
    }

    /// (cbh, cfl) — `constants.fbpCBH_CFL_HT_LUT` (height column unused here).
    fn old_cbh_cfl(fuel_type: i32) -> (f64, f64) {
        const NAN: f64 = f64::NAN;
        match fuel_type {
            1 => (2.0, 0.75),
            2 => (3.0, 0.8),
            3 => (8.0, 1.15),
            4 => (4.0, 1.2),
            5 => (18.0, 1.2),
            6 => (7.0, 1.8),
            7 => (10.0, 0.5),
            8 | 9 | 14 | 15 | 16 | 17 | 18 => (0.0, 0.0),
            10 | 11 => (6.0, 0.8),
            12 | 13 => (6.0, 0.8),
            _ => (NAN, NAN),
        }
    }

    fn old_is_modeled(fuel_type: i32) -> bool {
        (1..=18).contains(&fuel_type)
    }

    /// `constants.open_fuel_types`
    fn old_is_open_fuel(fuel_type: i32) -> bool {
        matches!(fuel_type, 1 | 7 | 9 | 14 | 15 | 16 | 17 | 18)
    }

    /// `constants.non_crowning_fuels`
    fn old_is_non_crowning(fuel_type: i32) -> bool {
        matches!(fuel_type, 8 | 9 | 14 | 15 | 16 | 17 | 18)
    }

    fn old_has_surface(fuel_type: i32) -> bool {
        matches!(fuel_type, 2..=7)
    }

    fn old_has_crown(fuel_type: i32) -> bool {
        matches!(fuel_type, 1 | 2 | 3 | 4 | 6 | 7)
    }

    fn same(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    #[test]
    fn code_round_trips_for_every_code() {
        for code in -3..=25 {
            assert_eq!(FuelType::from_code(code).code(), code, "code {code}");
        }
        assert_eq!(FuelType::from_code(1), FuelType::C1);
        assert_eq!(FuelType::from_code(10), FuelType::M1);
        assert_eq!(FuelType::from_code(14), FuelType::O1a);
        assert_eq!(FuelType::from_code(18), FuelType::S3);
        assert_eq!(FuelType::from_code(19), FuelType::NonFuel);
        assert_eq!(FuelType::from_code(20), FuelType::Water);
        assert_eq!(FuelType::from_code(21), FuelType::Unknown(21));
        assert_eq!(FuelType::from_code(-1), FuelType::Unknown(-1));
    }

    #[test]
    fn predicates_match_old_integer_logic() {
        for code in -3..=25 {
            let ft = FuelType::from_code(code);
            assert_eq!(ft.is_modeled(), old_is_modeled(code), "modeled {code}");
            assert_eq!(ft.is_open(), old_is_open_fuel(code), "open {code}");
            assert_eq!(
                ft.is_non_crowning(),
                old_is_non_crowning(code),
                "non-crowning {code}"
            );
        }
    }

    #[test]
    fn ros_params_match_old_table_including_fallback() {
        for code in -3..=25 {
            let old = old_ros_params(code);
            let p = FuelType::from_code(code).ros_params();
            let new = (p.a, p.b, p.c, p.q, p.bui0, p.be_max);
            assert!(
                same(old.0, new.0)
                    && same(old.1, new.1)
                    && same(old.2, new.2)
                    && same(old.3, new.3)
                    && same(old.4, new.4)
                    && same(old.5, new.5),
                "code {code}: {old:?} vs {new:?}"
            );
        }
    }

    #[test]
    fn crown_fuel_matches_old_table_including_fallback() {
        for code in -3..=25 {
            let old = old_cbh_cfl(code);
            let cf = FuelType::from_code(code).crown_fuel();
            assert!(
                same(old.0, cf.cbh) && same(old.1, cf.cfl),
                "code {code}: {old:?} vs ({}, {})",
                cf.cbh,
                cf.cfl
            );
        }
    }

    #[test]
    fn code_comparisons_match_old_thresholds() {
        // fbp.rs keeps `ft < 14` (csfi) and `ft < 19` (fire type) on the code.
        for code in -3..=25 {
            let ft = FuelType::from_code(code);
            assert_eq!(ft.code() < 14, code < 14);
            assert_eq!(ft.code() < 19, code < 19);
        }
    }

    #[test]
    fn percentile_regime_sets_match_old_integer_sets() {
        for code in -3..=25 {
            let ft = FuelType::from_code(code);
            assert_eq!(ft.has_surface_regime(), old_has_surface(code), "{code}");
            assert_eq!(ft.has_crown_regime(), old_has_crown(code), "{code}");
        }
    }
}
