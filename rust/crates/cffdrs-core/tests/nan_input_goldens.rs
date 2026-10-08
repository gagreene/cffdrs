//! NaN-input semantics: the Python package's normalization uses
//! `where(x < 0, 0, x)` / `clip(x, 0, None)`, both of which PRESERVE NaN —
//! a NaN ws/ffmc/bui/slope must propagate to NaN spread outputs (an
//! unspreadable cell), never be silently coerced to zero. Nodata holes in
//! weather rasters reach the core as NaN, so laundering NaN into "calm
//! wind" would let fire spread through missing data. Goldens captured from
//! the Python package.

use cffdrs_core::fbp::{run, FbpInput};

fn base() -> FbpInput {
    FbpInput {
        fuel_type: 2,
        wx_date: 20_230_615,
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
fn nan_gcf_propagates_for_grass_fuels() {
    // Grid-truth: O-1a with NaN gcf yields NaN hros — the _coerce default
    // (80) applies to a wholly-missing input, not to per-cell NaN, which
    // arrives masked and stays masked.
    let mut input = base();
    input.fuel_type = 14;
    input.gcf = f64::NAN;
    let r = run(&input);
    assert!(
        r.hros.is_nan(),
        "O-1a NaN gcf: hros must be NaN, got {}",
        r.hros
    );
}

#[test]
fn masked_cell_observables_match_grid_truth() {
    // Grid-truth captured from the Python package (1x4 grid, ws=NaN cell):
    // cfb and accel are NaN; fire_type surfaces as 0; fi_class as -99.
    let mut input = base();
    input.ws = f64::NAN;
    let r = run(&input);
    assert!(r.cfb.is_nan(), "cfb must be NaN, got {}", r.cfb);
    assert!(r.accel.is_nan(), "accel must be NaN, got {}", r.accel);
    assert_eq!(
        r.fire_type, 0.0,
        "fire_type observable is 0 for masked cells"
    );
    assert_eq!(
        r.fi_class, -99.0,
        "fi_class observable is -99 for masked cells"
    );
    assert!(
        r.wsv.is_nan() && r.raz.is_nan() && r.bros.is_nan(),
        "wind chain must be NaN"
    );
}

#[test]
fn nan_bui_leaves_wind_chain_intact() {
    // Grid-truth: NaN bui poisons the BE chain (hros/bros/hfi NaN) but the
    // wind vectoring is bui-independent and stays finite.
    let mut input = base();
    input.bui = f64::NAN;
    let r = run(&input);
    assert!(r.hros.is_nan() && r.bros.is_nan() && r.hfi.is_nan());
    assert!(
        (r.wsv - 20.251_831_110_187).abs() < 1e-9,
        "wsv finite: got {}",
        r.wsv
    );
    assert!(
        (r.raz - 170.954_944_453_942).abs() < 1e-9,
        "raz finite: got {}",
        r.raz
    );
}

/// (mutator, sfc expectation; NaN means sfc must be NaN) for one NaN-input scenario.
type NanCase = (fn(&mut FbpInput), f64);

#[test]
fn nan_weather_inputs_propagate_to_nan_spread() {
    // (mutator, sfc expectation): sfc is BUI/FFMC-driven per fuel, so it
    // pins that NaN reaches exactly the fields Python's masks let through.
    let cases: [NanCase; 4] = [
        (|i| i.ws = f64::NAN, 2.913_604_549_006_544_5),
        (|i| i.ffmc = f64::NAN, 2.913_604_549_006_544_5),
        // C-2 sfc is BUI-driven: a missing bui gives NaN sfc (Python agrees).
        (|i| i.bui = f64::NAN, f64::NAN),
        (|i| i.slope_pct = f64::NAN, 2.913_604_549_006_544_5),
    ];
    for (idx, (mutate, sfc)) in cases.iter().enumerate() {
        let mut input = base();
        mutate(&mut input);
        let r = run(&input);
        assert!(
            r.hros.is_nan(),
            "case {idx}: hros must be NaN, got {}",
            r.hros
        );
        assert!(r.hfi.is_nan(), "case {idx}: hfi must be NaN, got {}", r.hfi);
        let e = *sfc;
        if e.is_nan() {
            assert!(r.sfc.is_nan(), "case {idx}: sfc must be NaN, got {}", r.sfc);
        } else {
            assert!(
                (r.sfc - e).abs() <= e.abs() * 1e-9,
                "case {idx}: sfc expected {e}, got {}",
                r.sfc
            );
        }
    }
}
