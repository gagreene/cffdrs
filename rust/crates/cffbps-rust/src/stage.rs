//! The 23 named pipeline stages of the Python `FBP.runFBP`, with the fields
//! each one reads and writes.
//!
//! [`Stage::ALL`] is the canonical order (the `runFBP` order); running every
//! stage in that order on a [`CellState`](crate::state::CellState) is
//! [`fbp::run`](crate::fbp::run). [`Stage::reads`] and [`Stage::writes`] are the
//! data dependencies: a stage depends on no field outside `reads()` and changes
//! no field outside `writes()` (the stage-state tests check both).

use crate::field::Field;

/// One step of the FBP pipeline, named after the Python `FBP` method it
/// mirrors ([`Stage::name`]).
///
/// The enum is `#[non_exhaustive]`: match it with a wildcard arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Stage {
    /// `invertWindAspect`: flip wind direction and aspect by 180 degrees.
    InvertWindAspect,
    /// `calcSF`: slope factor.
    CalcSf,
    /// `calcISZ`: fine fuel moisture function and zero-wind, zero-slope ISI.
    CalcIsz,
    /// `calcFMC`: foliar moisture content and effect.
    CalcFmc,
    /// `calcISI_RSI_BE`: slope/wind ISI, initial spread rates, buildup effect.
    CalcIsiRsiBe,
    /// `calcROS`: head, backing and (C-6) surface rate of spread.
    CalcRos,
    /// `calcSFC`: surface fuel consumption.
    CalcSfc,
    /// `getCBH_CFL`: crown base height and crown fuel load.
    GetCbhCfl,
    /// `calcCSFI`: critical surface fire intensity.
    CalcCsfi,
    /// `calcRSO`: critical surface ROS for crowning.
    CalcRso,
    /// `calcC6BlendCFB`: temporary SROS-derived CFB of the C-6 blend.
    CalcC6BlendCfb,
    /// `calcC6BlendCFC`: temporary CFC of the C-6 blend.
    CalcC6BlendCfc,
    /// `calcC6CROS`: C-6 crown ROS.
    CalcC6Cros,
    /// `calcC6HROS`: C-6 blended head ROS.
    CalcC6Hros,
    /// `calcPercentileCFB`: directional CFB that selects the percentile regime.
    CalcPercentileCfb,
    /// `calcRosPercentileGrowth`: percentile growth of head and backing ROS.
    CalcRosPercentileGrowth,
    /// `calcCFB`: final directional CFB.
    CalcCfb,
    /// `calcAccelParam`: acceleration parameter.
    CalcAccelParam,
    /// `calcFireType`: fire type.
    CalcFireType,
    /// `calcCFC`: crown fuel consumption.
    CalcCfc,
    /// `calcTFC`: total fuel consumption.
    CalcTfc,
    /// `calcHFI`: head fire intensity.
    CalcHfi,
    /// `calcFireIntensityClass`: fire intensity class.
    CalcFireIntensityClass,
}

impl Stage {
    /// Every stage in the canonical (`runFBP`) order.
    pub const ALL: [Stage; 23] = [
        Stage::InvertWindAspect,
        Stage::CalcSf,
        Stage::CalcIsz,
        Stage::CalcFmc,
        Stage::CalcIsiRsiBe,
        Stage::CalcRos,
        Stage::CalcSfc,
        Stage::GetCbhCfl,
        Stage::CalcCsfi,
        Stage::CalcRso,
        Stage::CalcC6BlendCfb,
        Stage::CalcC6BlendCfc,
        Stage::CalcC6Cros,
        Stage::CalcC6Hros,
        Stage::CalcPercentileCfb,
        Stage::CalcRosPercentileGrowth,
        Stage::CalcCfb,
        Stage::CalcAccelParam,
        Stage::CalcFireType,
        Stage::CalcCfc,
        Stage::CalcTfc,
        Stage::CalcHfi,
        Stage::CalcFireIntensityClass,
    ];

    /// Number of stages.
    pub(crate) const COUNT: usize = Stage::ALL.len();

    /// Position of this stage in [`Stage::ALL`].
    #[must_use]
    pub(crate) const fn index(self) -> usize {
        self as usize
    }

    /// The Python `FBP` method name (`"calcSF"`, `"getCBH_CFL"`, ...).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Stage::InvertWindAspect => "invertWindAspect",
            Stage::CalcSf => "calcSF",
            Stage::CalcIsz => "calcISZ",
            Stage::CalcFmc => "calcFMC",
            Stage::CalcIsiRsiBe => "calcISI_RSI_BE",
            Stage::CalcRos => "calcROS",
            Stage::CalcSfc => "calcSFC",
            Stage::GetCbhCfl => "getCBH_CFL",
            Stage::CalcCsfi => "calcCSFI",
            Stage::CalcRso => "calcRSO",
            Stage::CalcC6BlendCfb => "calcC6BlendCFB",
            Stage::CalcC6BlendCfc => "calcC6BlendCFC",
            Stage::CalcC6Cros => "calcC6CROS",
            Stage::CalcC6Hros => "calcC6HROS",
            Stage::CalcPercentileCfb => "calcPercentileCFB",
            Stage::CalcRosPercentileGrowth => "calcRosPercentileGrowth",
            Stage::CalcCfb => "calcCFB",
            Stage::CalcAccelParam => "calcAccelParam",
            Stage::CalcFireType => "calcFireType",
            Stage::CalcCfc => "calcCFC",
            Stage::CalcTfc => "calcTFC",
            Stage::CalcHfi => "calcHFI",
            Stage::CalcFireIntensityClass => "calcFireIntensityClass",
        }
    }

    /// The stage with this Python method name; `None` for an unknown name.
    /// Names are case sensitive.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Stage> {
        Stage::ALL.into_iter().find(|s| s.name() == name)
    }

    /// Every field the stage's result can depend on. Besides these, a stage
    /// reads only the fixed per-fuel tables and the scalar parameters
    /// (`wx_date` for `calcFMC`, `percentile_growth` for
    /// `calcRosPercentileGrowth`). A field that is also in
    /// [`Stage::writes`] is kept where the stage does not replace it (for
    /// example `cros` for fuels other than C-6).
    #[must_use]
    pub const fn reads(self) -> &'static [Field] {
        use Field as F;
        match self {
            Stage::InvertWindAspect => &[F::WdIn, F::AspectIn],
            Stage::CalcSf => &[F::Slope],
            Stage::CalcIsz => &[F::Ffmc],
            Stage::CalcFmc => &[F::Lat, F::Long, F::Elevation, F::D0Override, F::DjOverride],
            Stage::CalcIsiRsiBe => &[
                F::FuelType,
                F::Gcf,
                F::Isz,
                F::Sf,
                F::Pc,
                F::Pdf,
                F::Bui,
                F::FF,
                F::Wd,
                F::Aspect,
                F::Ws,
            ],
            Stage::CalcRos => &[F::Rsi, F::Brsi, F::Be, F::Bui, F::Sros, F::FuelType],
            Stage::CalcSfc => &[F::FuelType, F::Ffmc, F::Bui, F::Pc, F::Gfl],
            Stage::GetCbhCfl => &[F::FuelType, F::Cbh, F::Cfl],
            Stage::CalcCsfi => &[F::FuelType, F::Cbh, F::Fmc],
            Stage::CalcRso => &[F::Sfc, F::Csfi],
            Stage::CalcC6BlendCfb => &[F::FuelType, F::Sros, F::Rso],
            Stage::CalcC6BlendCfc => &[F::FuelType, F::C6BlendCfb, F::Cfl, F::Pc, F::Pdf],
            Stage::CalcC6Cros => &[F::FuelType, F::C6BlendCfc, F::Isi, F::Fme, F::Cros],
            // hros_growth_nan: kept, like hros, for fuels other than C-6
            Stage::CalcC6Hros => &[
                F::FuelType,
                F::Sros,
                F::Cros,
                F::C6BlendCfb,
                F::Hros,
                F::HrosGrowthNan,
            ],
            Stage::CalcPercentileCfb => &[F::FuelType, F::Rso, F::Hros, F::Bros],
            Stage::CalcRosPercentileGrowth => &[
                F::FuelType,
                F::PercentileCfb,
                F::PercentileBrosCfb,
                F::Wsv,
                F::Hros,
                F::Bros,
            ],
            Stage::CalcCfb => &[F::FuelType, F::Rso, F::Hros, F::HrosGrowthNan, F::Bros],
            Stage::CalcAccelParam => &[F::FuelType, F::Cfb, F::Accel],
            Stage::CalcFireType => &[F::FuelType, F::Cfb],
            Stage::CalcCfc => &[F::FuelType, F::Cfb, F::Cfl, F::Pc, F::Pdf],
            Stage::CalcTfc => &[F::Sfc, F::Cfc],
            Stage::CalcHfi => &[F::Hros, F::Tfc],
            Stage::CalcFireIntensityClass => &[F::Hfi],
        }
    }

    /// Every field the stage may change; all other fields are left exactly
    /// as they were.
    #[must_use]
    pub const fn writes(self) -> &'static [Field] {
        use Field as F;
        match self {
            Stage::InvertWindAspect => &[F::Wd, F::Aspect],
            Stage::CalcSf => &[F::Sf],
            Stage::CalcIsz => &[F::M, F::FF, F::Isz],
            Stage::CalcFmc => &[F::Latn, F::D0, F::Dj, F::Nd, F::Fmc, F::Fme],
            Stage::CalcIsiRsiBe => &[
                F::A,
                F::B,
                F::C,
                F::Q,
                F::Bui0,
                F::BeMax,
                F::Rsz,
                F::Rsf,
                F::Isf,
                F::Rsi,
                F::Brsi,
                F::Be,
                F::Wse1,
                F::Wse2,
                F::Wse,
                F::Wsx,
                F::Wsy,
                F::Wsv,
                F::Raz,
                F::FW,
                F::Bfw,
                F::Isi,
                F::Bisi,
            ],
            Stage::CalcRos => &[F::Hros, F::Bros, F::Sros, F::HrosGrowthNan],
            Stage::CalcSfc => &[F::Ffc, F::Wfc, F::Sfc],
            Stage::GetCbhCfl => &[F::Cbh, F::Cfl],
            Stage::CalcCsfi => &[F::Csfi],
            Stage::CalcRso => &[F::Rso],
            Stage::CalcC6BlendCfb => &[F::C6BlendCfb],
            Stage::CalcC6BlendCfc => &[F::C6BlendCfc],
            Stage::CalcC6Cros => &[F::Cros],
            Stage::CalcC6Hros => &[F::Hros, F::HrosGrowthNan],
            Stage::CalcPercentileCfb => &[F::PercentileCfb, F::PercentileBrosCfb],
            Stage::CalcRosPercentileGrowth => &[F::Hros, F::Bros, F::HrosGrowthNan],
            Stage::CalcCfb => &[F::Cfb, F::BrosCfb],
            Stage::CalcAccelParam => &[F::Accel],
            Stage::CalcFireType => &[F::FireType],
            Stage::CalcCfc => &[F::Cfc],
            Stage::CalcTfc => &[F::Tfc],
            Stage::CalcHfi => &[F::Hfi],
            Stage::CalcFireIntensityClass => &[F::FiClass],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Stage;
    use std::collections::HashSet;

    /// The stage and field reference: (Python method name, reads, writes), in
    /// `runFBP` order. Typed independently of `Stage::reads`/`Stage::writes`
    /// (which use `Field` variants) so a typo in either is caught.
    const REFERENCE: [(&str, &[&str], &[&str]); 23] = [
        (
            "invertWindAspect",
            &["wd_in", "aspect_in"],
            &["wd", "aspect"],
        ),
        ("calcSF", &["slope"], &["sf"]),
        ("calcISZ", &["ffmc"], &["m", "fF", "isz"]),
        (
            "calcFMC",
            &["lat", "long", "elevation", "d0_override", "dj_override"],
            &["latn", "d0", "dj", "nd", "fmc", "fme"],
        ),
        (
            "calcISI_RSI_BE",
            &[
                "fuel_type",
                "gcf",
                "isz",
                "sf",
                "pc",
                "pdf",
                "bui",
                "fF",
                "wd",
                "aspect",
                "ws",
            ],
            &[
                "a", "b", "c", "q", "bui0", "be_max", "rsz", "rsf", "isf", "rsi", "brsi", "be",
                "wse1", "wse2", "wse", "wsx", "wsy", "wsv", "raz", "fW", "bfw", "isi", "bisi",
            ],
        ),
        (
            "calcROS",
            &["rsi", "brsi", "be", "bui", "sros", "fuel_type"],
            &["hros", "bros", "sros", "hros_growth_nan"],
        ),
        (
            "calcSFC",
            &["fuel_type", "ffmc", "bui", "pc", "gfl"],
            &["ffc", "wfc", "sfc"],
        ),
        ("getCBH_CFL", &["fuel_type", "cbh", "cfl"], &["cbh", "cfl"]),
        ("calcCSFI", &["fuel_type", "cbh", "fmc"], &["csfi"]),
        ("calcRSO", &["sfc", "csfi"], &["rso"]),
        (
            "calcC6BlendCFB",
            &["fuel_type", "sros", "rso"],
            &["c6_blend_cfb"],
        ),
        (
            "calcC6BlendCFC",
            &["fuel_type", "c6_blend_cfb", "cfl", "pc", "pdf"],
            &["c6_blend_cfc"],
        ),
        (
            "calcC6CROS",
            &["fuel_type", "c6_blend_cfc", "isi", "fme", "cros"],
            &["cros"],
        ),
        (
            "calcC6HROS",
            &[
                "fuel_type",
                "sros",
                "cros",
                "c6_blend_cfb",
                "hros",
                "hros_growth_nan",
            ],
            &["hros", "hros_growth_nan"],
        ),
        (
            "calcPercentileCFB",
            &["fuel_type", "rso", "hros", "bros"],
            &["percentile_cfb", "percentile_bros_cfb"],
        ),
        (
            "calcRosPercentileGrowth",
            &[
                "fuel_type",
                "percentile_cfb",
                "percentile_bros_cfb",
                "wsv",
                "hros",
                "bros",
            ],
            &["hros", "bros", "hros_growth_nan"],
        ),
        (
            "calcCFB",
            &["fuel_type", "rso", "hros", "hros_growth_nan", "bros"],
            &["cfb", "bros_cfb"],
        ),
        ("calcAccelParam", &["fuel_type", "cfb", "accel"], &["accel"]),
        ("calcFireType", &["fuel_type", "cfb"], &["fire_type"]),
        (
            "calcCFC",
            &["fuel_type", "cfb", "cfl", "pc", "pdf"],
            &["cfc"],
        ),
        ("calcTFC", &["sfc", "cfc"], &["tfc"]),
        ("calcHFI", &["hros", "tfc"], &["hfi"]),
        ("calcFireIntensityClass", &["hfi"], &["fi_class"]),
    ];

    #[test]
    fn all_has_23_unique_stages() {
        assert_eq!(Stage::ALL.len(), 23);
        let unique: HashSet<Stage> = Stage::ALL.iter().copied().collect();
        assert_eq!(unique.len(), 23, "duplicate stage in Stage::ALL");
    }

    #[test]
    fn names_are_the_python_method_names_in_run_fbp_order() {
        let names: Vec<&str> = Stage::ALL.iter().map(|s| s.name()).collect();
        let expected: Vec<&str> = REFERENCE.iter().map(|(n, _, _)| *n).collect();
        assert_eq!(names, expected);
    }

    #[test]
    fn index_is_the_position_in_all() {
        assert_eq!(Stage::COUNT, 23);
        for (k, s) in Stage::ALL.into_iter().enumerate() {
            assert_eq!(s.index(), k, "{s:?}");
        }
    }

    #[test]
    fn from_name_round_trips() {
        for s in Stage::ALL {
            assert_eq!(Stage::from_name(s.name()), Some(s), "{s:?}");
        }
        assert_eq!(Stage::from_name("runFBP"), None);
        assert_eq!(Stage::from_name("calcsf"), None, "names are case sensitive");
    }

    #[test]
    fn reads_and_writes_follow_the_reference_table() {
        for (stage, (name, reads, writes)) in Stage::ALL.iter().zip(REFERENCE) {
            let got_reads: Vec<&str> = stage.reads().iter().map(|f| f.name()).collect();
            let got_writes: Vec<&str> = stage.writes().iter().map(|f| f.name()).collect();
            assert_eq!(got_reads, reads, "{name} reads");
            assert_eq!(got_writes, writes, "{name} writes");
        }
    }

    #[test]
    fn every_stage_writes_something_and_no_field_twice() {
        for s in Stage::ALL {
            assert!(!s.writes().is_empty(), "{s:?}");
            let w: HashSet<_> = s.writes().iter().collect();
            assert_eq!(w.len(), s.writes().len(), "{s:?} writes a field twice");
            let r: HashSet<_> = s.reads().iter().collect();
            assert_eq!(r.len(), s.reads().len(), "{s:?} reads a field twice");
        }
    }
}
