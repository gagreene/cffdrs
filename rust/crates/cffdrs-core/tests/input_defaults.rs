//! `FbpInput::new`: the Python `initialize` defaults, with the environment
//! inputs left missing (NaN) so that forgetting one propagates instead of
//! silently defaulting.

use cffdrs_core::fbp::{run, FbpInput};

#[test]
fn new_has_the_python_initialize_defaults() {
    let input = FbpInput::new(2, 20_230_615);
    assert_eq!(input.fuel_type, 2);
    assert_eq!(input.wx_date, 20_230_615);
    assert_eq!(input.pc, 50.0);
    assert_eq!(input.pdf, 35.0);
    assert_eq!(input.gfl, 0.35);
    assert_eq!(input.gcf, 80.0);
    assert_eq!(input.percentile_growth, 50.0);
    assert_eq!(input.d0_override, None);
    assert_eq!(input.dj_override, None);
    assert_eq!(input.fmc_override, None);
    assert_eq!(input.hros_override, None);
    for (name, v) in [
        ("lat", input.lat),
        ("long", input.long),
        ("elevation", input.elevation),
        ("slope_pct", input.slope_pct),
        ("aspect_deg", input.aspect_deg),
        ("ws", input.ws),
        ("wd", input.wd),
        ("ffmc", input.ffmc),
        ("bui", input.bui),
    ] {
        assert!(v.is_nan(), "{name} should start missing (NaN), got {v}");
    }
}

#[test]
fn unset_environment_propagates_as_missing_spread() {
    let result = run(&FbpInput::new(2, 20_230_615));
    assert!(result.hros.is_nan());
}
