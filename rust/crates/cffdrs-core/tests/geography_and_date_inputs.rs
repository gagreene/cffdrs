//! Missing geography and calendar validation.
//!
//! A NaN latitude, longitude or elevation is a masked cell in the Python
//! package: foliar moisture (and everything crown-related that depends on it)
//! is masked too, never a valid default. Dates are validated as real
//! YYYYMMDD calendar dates, as `datetime.strptime` does.

use cffdrs_core::fbp::{is_valid_wx_date, run, FbpInput};

fn c6() -> FbpInput {
    FbpInput {
        fuel_type: 6,
        wx_date: 20230615,
        lat: 55.0,
        long: -110.0,
        elevation: 500.0,
        slope_pct: 10.0,
        aspect_deg: 270.0,
        ws: 20.0,
        wd: 0.0,
        ffmc: 91.0,
        bui: 76.0,
        pc: 50.0,
        pdf: 35.0,
        gfl: 0.35,
        gcf: 80.0,
        percentile_growth: 50.0,
        d0_override: None,
        dj_override: None,
        fmc_override: None,
        hros_override: None,
    }
}

#[test]
fn finite_geography_gives_finite_c6_behaviour() {
    let r = run(&c6());
    assert!(r.fmc.is_finite() && r.hros.is_finite() && r.hfi.is_finite());
}

#[test]
fn nan_geography_makes_foliar_moisture_and_c6_behaviour_nan() {
    for (name, mutate) in [
        (
            "lat",
            (|i: &mut FbpInput| i.lat = f64::NAN) as fn(&mut FbpInput),
        ),
        ("long", |i| i.long = f64::NAN),
        ("elevation", |i| i.elevation = f64::NAN),
    ] {
        let mut input = c6();
        mutate(&mut input);
        let r = run(&input);
        assert!(r.fmc.is_nan(), "{name}: fmc must be NaN, got {}", r.fmc);
        assert!(r.fme.is_nan(), "{name}: fme must be NaN, got {}", r.fme);
        assert!(
            r.hros.is_nan(),
            "{name}: C-6 hros must be NaN, got {}",
            r.hros
        );
        assert!(r.hfi.is_nan(), "{name}: C-6 hfi must be NaN, got {}", r.hfi);
    }
}

#[test]
fn injected_dates_fully_determine_fmc_despite_missing_elevation() {
    // With both d0 and dj supplied by the caller, foliar moisture depends on
    // neither geography value (Python: nd = |dj - d0| of the overrides), so a
    // missing elevation does not mask it.
    let mut input = c6();
    input.d0_override = Some(150.0);
    input.dj_override = Some(165.0);
    input.elevation = f64::NAN;
    let r = run(&input);
    assert!(r.fmc.is_finite(), "overridden d0/dj fully determine fmc");
}

#[test]
fn calendar_validation() {
    for ok in [20230615, 20240229, 20231231, 20230101, 20000229, 20230228] {
        assert!(is_valid_wx_date(ok), "{ok} should be valid");
    }
    for bad in [
        20231301, 20230230, 20230431, 20230229, 20231232, 20230001, 20230100, 19000229, 0, -1,
        2023615, 123456789, 20230632,
    ] {
        assert!(!is_valid_wx_date(bad), "{bad} should be invalid");
    }
}
