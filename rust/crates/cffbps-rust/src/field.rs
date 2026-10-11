//! The per-cell quantities the pipeline stages read and write.
//!
//! [`Field`] names every value a [`Stage`](crate::stage::Stage) reads or writes:
//! the 54 outputs of [`FbpResult::named_values`](crate::fbp::FbpResult::named_values)
//! (in that order), then 12 input-only fields, then 9 internal fields that are
//! not outputs. [`Field::name`] is the Python package's attribute name.

/// One per-cell quantity of the FBP pipeline.
///
/// The variants are the 54 outputs (in [`FbpResult::named_values`] order), the
/// 12 input-only fields (`lat` .. `aspect_in`; `wd_in` and `aspect_in` are the
/// raw wind direction and aspect, `d0_override` and `dj_override` the explicit
/// foliar moisture date overrides) and 9 internal fields (`isz` .. `aspect`;
/// `aspect` is the working, inverted aspect). `ws`, `wd`, `ffmc`, `bui` and
/// `fuel_type` are inputs that are also outputs (their processed values; `wd`
/// is the inverted wind direction).
///
/// The enum is `#[non_exhaustive]`: match it with a wildcard arm.
///
/// [`FbpResult::named_values`]: crate::fbp::FbpResult::named_values
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Field {
    // the 54 outputs
    /// `ws`: wind speed as used (normalised input), km/h.
    Ws,
    /// `wd`: wind direction inverted by 180 degrees (the raw input until
    /// `invertWindAspect` runs), compass degrees.
    Wd,
    /// `wse`: slope-equivalent wind speed, km/h.
    Wse,
    /// `wse1`: original slope-equivalent wind speed, km/h.
    Wse1,
    /// `wse2`: revised slope-equivalent wind speed, km/h.
    Wse2,
    /// `wsx`: net vectorised wind speed, x (east) component, km/h.
    Wsx,
    /// `wsy`: net vectorised wind speed, y (north) component, km/h.
    Wsy,
    /// `wsv`: net vectorised wind speed, km/h.
    Wsv,
    /// `raz`: net spread direction, compass degrees.
    Raz,
    /// `m`: fine fuel moisture content, percent.
    M,
    /// `fF`: fine fuel moisture function.
    FF,
    /// `fW`: wind function.
    FW,
    /// `ffmc`: Fine Fuel Moisture Code as used (normalised input).
    Ffmc,
    /// `isi`: final ISI.
    Isi,
    /// `bui`: Buildup Index as used (normalised input).
    Bui,
    /// `a`: surface ROS parameter.
    A,
    /// `b`: surface ROS parameter.
    B,
    /// `c`: surface ROS parameter.
    C,
    /// `q`: proportion of maximum ROS at BUI 50.
    Q,
    /// `bui0`: average BUI for the fuel type.
    Bui0,
    /// `be`: buildup effect.
    Be,
    /// `be_max`: maximum buildup effect.
    BeMax,
    /// `sf`: slope factor.
    Sf,
    /// `rsz`: surface ROS at zero wind on flat ground, m/min.
    Rsz,
    /// `rsf`: surface ROS with the slope effect at zero wind, m/min.
    Rsf,
    /// `isf`: ISI with the slope effect at zero wind.
    Isf,
    /// `rsi`: initial spread rate without buildup effect, m/min.
    Rsi,
    /// `latn`: normalised latitude of the foliar moisture date.
    Latn,
    /// `dj`: Julian date of the fire.
    Dj,
    /// `d0`: Julian date of minimum foliar moisture.
    D0,
    /// `nd`: days from the date of minimum foliar moisture.
    Nd,
    /// `fmc`: foliar moisture content, percent.
    Fmc,
    /// `fme`: foliar moisture effect.
    Fme,
    /// `ffc`: forest-floor fuel consumption, kg/m^2.
    Ffc,
    /// `wfc`: woody fuel consumption, kg/m^2.
    Wfc,
    /// `sfc`: surface fuel consumption, kg/m^2.
    Sfc,
    /// `cfl`: crown fuel load, kg/m^2.
    Cfl,
    /// `cfc`: crown fuel consumption, kg/m^2.
    Cfc,
    /// `tfc`: total fuel consumption, kg/m^2.
    Tfc,
    /// `cbh`: crown base height, m.
    Cbh,
    /// `csfi`: critical surface fire intensity, kW/m.
    Csfi,
    /// `rso`: critical surface ROS for crowning, m/min.
    Rso,
    /// `cfb`: crown fraction burned.
    Cfb,
    /// `fire_type`: fire type code.
    FireType,
    /// `hros`: head fire rate of spread, m/min.
    Hros,
    /// `sros`: surface head fire rate of spread (C-6), m/min.
    Sros,
    /// `cros`: crown fire rate of spread (C-6), m/min.
    Cros,
    /// `bfw`: backing fire wind function.
    Bfw,
    /// `bisi`: backing fire ISI.
    Bisi,
    /// `bros`: backing fire rate of spread, m/min.
    Bros,
    /// `hfi`: head fire intensity, kW/m.
    Hfi,
    /// `fi_class`: fire intensity class.
    FiClass,
    /// `accel`: acceleration parameter.
    Accel,
    /// `fuel_type`: fuel type code.
    FuelType,
    // the 12 input-only fields
    /// `lat`: latitude, decimal degrees.
    Lat,
    /// `long`: magnitude of the longitude, decimal degrees.
    Long,
    /// `elevation`: elevation, m.
    Elevation,
    /// `slope`: ground slope as used (normalised input), percent.
    Slope,
    /// `pc`: percent conifer as used (normalised input).
    Pc,
    /// `pdf`: percent dead fir as used (normalised input).
    Pdf,
    /// `gfl`: grass fuel load as used (normalised input), kg/m^2.
    Gfl,
    /// `gcf`: grass curing factor as used (normalised input), percent.
    Gcf,
    /// `d0_override`: explicit date of minimum foliar moisture (NaN = masked).
    D0Override,
    /// `dj_override`: explicit Julian date (NaN = masked).
    DjOverride,
    /// `wd_in`: wind direction input, compass degrees.
    WdIn,
    /// `aspect_in`: aspect input as used (normalised), compass degrees.
    AspectIn,
    // the 9 internal fields
    /// `isz`: ISI at zero wind and slope.
    Isz,
    /// `brsi`: backing initial spread rate, m/min.
    Brsi,
    /// `c6_blend_cfb`: temporary SROS-derived CFB of the C-6 blend.
    C6BlendCfb,
    /// `c6_blend_cfc`: temporary CFC of the C-6 blend.
    C6BlendCfc,
    /// `percentile_cfb`: head CFB that selects the percentile-growth regime.
    PercentileCfb,
    /// `percentile_bros_cfb`: backing CFB that selects the percentile-growth regime.
    PercentileBrosCfb,
    /// `bros_cfb`: final backing CFB.
    BrosCfb,
    /// `hros_growth_nan`: 1.0 when the percentile step itself turned a
    /// non-NaN head ROS into NaN (a NaN percentile), otherwise 0.0.
    HrosGrowthNan,
    /// `aspect`: aspect inverted by 180 degrees (the input until
    /// `invertWindAspect` runs), compass degrees.
    Aspect,
}

impl Field {
    /// Number of fields.
    pub const COUNT: usize = 75;

    /// Every field, in declaration order (outputs, input-only, internal).
    pub const ALL: [Field; Field::COUNT] = [
        Field::Ws,
        Field::Wd,
        Field::Wse,
        Field::Wse1,
        Field::Wse2,
        Field::Wsx,
        Field::Wsy,
        Field::Wsv,
        Field::Raz,
        Field::M,
        Field::FF,
        Field::FW,
        Field::Ffmc,
        Field::Isi,
        Field::Bui,
        Field::A,
        Field::B,
        Field::C,
        Field::Q,
        Field::Bui0,
        Field::Be,
        Field::BeMax,
        Field::Sf,
        Field::Rsz,
        Field::Rsf,
        Field::Isf,
        Field::Rsi,
        Field::Latn,
        Field::Dj,
        Field::D0,
        Field::Nd,
        Field::Fmc,
        Field::Fme,
        Field::Ffc,
        Field::Wfc,
        Field::Sfc,
        Field::Cfl,
        Field::Cfc,
        Field::Tfc,
        Field::Cbh,
        Field::Csfi,
        Field::Rso,
        Field::Cfb,
        Field::FireType,
        Field::Hros,
        Field::Sros,
        Field::Cros,
        Field::Bfw,
        Field::Bisi,
        Field::Bros,
        Field::Hfi,
        Field::FiClass,
        Field::Accel,
        Field::FuelType,
        Field::Lat,
        Field::Long,
        Field::Elevation,
        Field::Slope,
        Field::Pc,
        Field::Pdf,
        Field::Gfl,
        Field::Gcf,
        Field::D0Override,
        Field::DjOverride,
        Field::WdIn,
        Field::AspectIn,
        Field::Isz,
        Field::Brsi,
        Field::C6BlendCfb,
        Field::C6BlendCfc,
        Field::PercentileCfb,
        Field::PercentileBrosCfb,
        Field::BrosCfb,
        Field::HrosGrowthNan,
        Field::Aspect,
    ];

    /// Number of output fields (the first 54 variants).
    const OUTPUTS: usize = 54;

    /// Position of this field in [`Field::ALL`].
    #[must_use]
    pub(crate) const fn index(self) -> usize {
        self as usize
    }

    /// The Python package's name for this field (`"fF"`, `"fW"`,
    /// `"fire_type"`, `"bfw"`, `"wd_in"`, ...).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Field::Ws => "ws",
            Field::Wd => "wd",
            Field::Wse => "wse",
            Field::Wse1 => "wse1",
            Field::Wse2 => "wse2",
            Field::Wsx => "wsx",
            Field::Wsy => "wsy",
            Field::Wsv => "wsv",
            Field::Raz => "raz",
            Field::M => "m",
            Field::FF => "fF",
            Field::FW => "fW",
            Field::Ffmc => "ffmc",
            Field::Isi => "isi",
            Field::Bui => "bui",
            Field::A => "a",
            Field::B => "b",
            Field::C => "c",
            Field::Q => "q",
            Field::Bui0 => "bui0",
            Field::Be => "be",
            Field::BeMax => "be_max",
            Field::Sf => "sf",
            Field::Rsz => "rsz",
            Field::Rsf => "rsf",
            Field::Isf => "isf",
            Field::Rsi => "rsi",
            Field::Latn => "latn",
            Field::Dj => "dj",
            Field::D0 => "d0",
            Field::Nd => "nd",
            Field::Fmc => "fmc",
            Field::Fme => "fme",
            Field::Ffc => "ffc",
            Field::Wfc => "wfc",
            Field::Sfc => "sfc",
            Field::Cfl => "cfl",
            Field::Cfc => "cfc",
            Field::Tfc => "tfc",
            Field::Cbh => "cbh",
            Field::Csfi => "csfi",
            Field::Rso => "rso",
            Field::Cfb => "cfb",
            Field::FireType => "fire_type",
            Field::Hros => "hros",
            Field::Sros => "sros",
            Field::Cros => "cros",
            Field::Bfw => "bfw",
            Field::Bisi => "bisi",
            Field::Bros => "bros",
            Field::Hfi => "hfi",
            Field::FiClass => "fi_class",
            Field::Accel => "accel",
            Field::FuelType => "fuel_type",
            Field::Lat => "lat",
            Field::Long => "long",
            Field::Elevation => "elevation",
            Field::Slope => "slope",
            Field::Pc => "pc",
            Field::Pdf => "pdf",
            Field::Gfl => "gfl",
            Field::Gcf => "gcf",
            Field::D0Override => "d0_override",
            Field::DjOverride => "dj_override",
            Field::WdIn => "wd_in",
            Field::AspectIn => "aspect_in",
            Field::Isz => "isz",
            Field::Brsi => "brsi",
            Field::C6BlendCfb => "c6_blend_cfb",
            Field::C6BlendCfc => "c6_blend_cfc",
            Field::PercentileCfb => "percentile_cfb",
            Field::PercentileBrosCfb => "percentile_bros_cfb",
            Field::BrosCfb => "bros_cfb",
            Field::HrosGrowthNan => "hros_growth_nan",
            Field::Aspect => "aspect",
        }
    }

    /// The field with this Python name; `None` for an unknown name. Names are
    /// case sensitive (`"fF"`, not `"ff"`).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Field> {
        Field::ALL.into_iter().find(|f| f.name() == name)
    }

    /// True for the 54 outputs of
    /// [`FbpResult::named_values`](crate::fbp::FbpResult::named_values).
    #[must_use]
    pub const fn is_output(self) -> bool {
        self.index() < Field::OUTPUTS
    }

    /// The value a field holds before anything sets or computes it, as in the
    /// Python `FBP.initialize`: 0.0 for computed fields (including `fi_class`
    /// and `hros_growth_nan`), NaN for `ffc`, `wfc` and `sfc`, and NaN
    /// (missing) for inputs. `wd` and `aspect` read as their input until
    /// `invertWindAspect` runs, so they also start missing.
    #[must_use]
    pub const fn template_value(self) -> f64 {
        match self {
            Field::Ffc
            | Field::Wfc
            | Field::Sfc
            | Field::Ws
            | Field::Wd
            | Field::Ffmc
            | Field::Bui
            | Field::FuelType
            | Field::Lat
            | Field::Long
            | Field::Elevation
            | Field::Slope
            | Field::Pc
            | Field::Pdf
            | Field::Gfl
            | Field::Gcf
            | Field::D0Override
            | Field::DjOverride
            | Field::WdIn
            | Field::AspectIn
            | Field::Aspect => f64::NAN,
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Field;
    use crate::fbp::FbpResult;
    use std::collections::HashSet;

    /// The 75 field names of the stage and field reference, in its order: the
    /// 54 outputs, the 12 input-only fields, the 9 internal fields. Typed
    /// independently of `Field::name` so a typo in either is caught.
    const REFERENCE_NAMES: [&str; 75] = [
        "ws",
        "wd",
        "wse",
        "wse1",
        "wse2",
        "wsx",
        "wsy",
        "wsv",
        "raz",
        "m",
        "fF",
        "fW",
        "ffmc",
        "isi",
        "bui",
        "a",
        "b",
        "c",
        "q",
        "bui0",
        "be",
        "be_max",
        "sf",
        "rsz",
        "rsf",
        "isf",
        "rsi",
        "latn",
        "dj",
        "d0",
        "nd",
        "fmc",
        "fme",
        "ffc",
        "wfc",
        "sfc",
        "cfl",
        "cfc",
        "tfc",
        "cbh",
        "csfi",
        "rso",
        "cfb",
        "fire_type",
        "hros",
        "sros",
        "cros",
        "bfw",
        "bisi",
        "bros",
        "hfi",
        "fi_class",
        "accel",
        "fuel_type",
        // input-only
        "lat",
        "long",
        "elevation",
        "slope",
        "pc",
        "pdf",
        "gfl",
        "gcf",
        "d0_override",
        "dj_override",
        "wd_in",
        "aspect_in",
        // internal
        "isz",
        "brsi",
        "c6_blend_cfb",
        "c6_blend_cfc",
        "percentile_cfb",
        "percentile_bros_cfb",
        "bros_cfb",
        "hros_growth_nan",
        "aspect",
    ];

    #[test]
    fn all_has_75_unique_fields_with_unique_names() {
        assert_eq!(Field::ALL.len(), Field::COUNT);
        assert_eq!(Field::COUNT, 75);
        let unique: HashSet<Field> = Field::ALL.iter().copied().collect();
        assert_eq!(unique.len(), 75, "duplicate field in Field::ALL");
        let names: HashSet<&str> = Field::ALL.iter().map(|f| f.name()).collect();
        assert_eq!(names.len(), 75, "duplicate field name");
    }

    #[test]
    fn names_follow_the_reference_order() {
        let names: Vec<&str> = Field::ALL.iter().map(|f| f.name()).collect();
        assert_eq!(names, REFERENCE_NAMES);
    }

    #[test]
    fn the_output_fields_are_exactly_the_named_values() {
        let outputs: Vec<&str> = Field::ALL
            .iter()
            .filter(|f| f.is_output())
            .map(|f| f.name())
            .collect();
        let named: Vec<&str> = FbpResult::default()
            .named_values()
            .iter()
            .map(|(n, _)| *n)
            .collect();
        assert_eq!(outputs.len(), 54);
        assert_eq!(outputs, named);
    }

    #[test]
    fn from_name_round_trips() {
        for f in Field::ALL {
            assert_eq!(Field::from_name(f.name()), Some(f), "{f:?}");
        }
        assert_eq!(Field::from_name("nope"), None);
        assert_eq!(Field::from_name("f_f"), None, "the Python name is fF");
    }

    #[test]
    fn template_values_follow_the_python_initialize() {
        for f in Field::ALL {
            let t = f.template_value();
            match f.name() {
                // ffc / wfc / sfc, then the inputs (and wd / aspect, which
                // read as their input until invertWindAspect runs): missing
                "ffc" | "wfc" | "sfc" | "ws" | "wd" | "ffmc" | "bui" | "fuel_type" | "lat"
                | "long" | "elevation" | "slope" | "pc" | "pdf" | "gfl" | "gcf" | "d0_override"
                | "dj_override" | "wd_in" | "aspect_in" | "aspect" => assert!(t.is_nan(), "{f:?}"),
                _ => assert_eq!(t.to_bits(), 0.0_f64.to_bits(), "{f:?}"),
            }
        }
    }
}
