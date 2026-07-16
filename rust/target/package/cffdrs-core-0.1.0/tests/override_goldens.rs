//! Override-mode validation: the FBP chain with fmc and hros injected via
//! setParams — the sequence fire-growth engines use to recompute output
//! quantities from an engine-supplied ROS:
//!
//!   initialize -> invertWindAspect -> calcSF -> calcISZ -> setParams(fmc)
//!   -> calcISI_RSI_BE -> calcSFC -> getCBH_CFL -> calcROS -> setParams(hros)
//!   -> calcCSFI -> calcRSO -> calcCFB -> calcRosPercentileGrowth
//!   -> calcFireType -> calcCFC -> calcC6hros -> calcTFC -> calcHFI
//!   -> calcFireIntensityClass
//!
//! Notable spec behaviours these goldens pin (captured verbatim from the
//! Python package running that exact sequence):
//! - calcFMC never runs, so fme keeps its zero template value — C-6 crown
//!   ROS is therefore 0 and the C-6 blend collapses to sros * (1 - cfb),
//!   OVERWRITING the injected hros (case 3).
//! - C-6 cfb comes from sros (not the injected hros); other crowning fuels
//!   use the injected hros.
//! - hfi = 0 yields fi_class -99 (case 5).

use cffdrs_core::fbp::{run, FbpInput};

struct Case {
    fuel_type: i32,
    ws: f64,
    wd: f64,
    ffmc: f64,
    bui: f64,
    fmc_override: f64,
    hros_override: f64,
    // csfi, rso, cfb, fire_type, cfc, hros, tfc, hfi, fi_class, sfc, fmc
    expected: [f64; 11],
}

const CASES: [Case; 5] = [
    Case {
        fuel_type: 2, ws: 20.0, wd: 0.0, ffmc: 91.0, bui: 76.0,
        fmc_override: 97.5, hros_override: 4.0,
        expected: [
            847.5258291304367, 0.9696189203386331, 0.5019165512987145, 2.0,
            0.4015332410389716, 4.0, 3.315137790045516, 3978.1653480546192,
            4.0, 2.9136045490065445, 97.5,
        ],
    },
    Case {
        fuel_type: 2, ws: 20.0, wd: 0.0, ffmc: 91.0, bui: 76.0,
        fmc_override: 97.5, hros_override: 60.0,
        expected: [
            847.5258291304367, 0.9696189203386331, 0.9999987306272135, 3.0,
            0.7999989845017708, 60.0, 3.7136035335083153, 66844.86360314967,
            6.0, 2.9136045490065445, 97.5,
        ],
    },
    Case {
        fuel_type: 6, ws: 25.0, wd: 90.0, ffmc: 92.5, bui: 85.0,
        fmc_override: 105.0, hros_override: 10.0,
        expected: [
            3320.360999080089, 5.030716309746868, 0.9114986320304777, 3.0,
            1.6406975376548598, 1.3782364289505988, 3.8407560015936735,
            1588.0409508321136, 3.0, 2.2000584639388134, 105.0,
        ],
    },
    Case {
        fuel_type: 14, ws: 30.0, wd: 180.0, ffmc: 90.0, bui: 60.0,
        fmc_override: 120.0, hros_override: 25.0,
        expected: [0.0, 0.0, 0.0, 1.0, 0.0, 25.0, 0.35, 2625.0, 4.0, 0.35, 120.0],
    },
    Case {
        fuel_type: 10, ws: 15.0, wd: 270.0, ffmc: 88.0, bui: 70.0,
        fmc_override: 99.0, hros_override: 0.0,
        expected: [
            2444.111969224106, 4.234514865780585, 0.0, 1.0, 0.0, 0.0,
            1.9239606320078113, 0.0, -99.0, 1.9239606320078113, 99.0,
        ],
    },
];

#[test]
fn override_chain_matches_python_sequence() {
    for (i, case) in CASES.iter().enumerate() {
        let result = run(&FbpInput {
            fuel_type: case.fuel_type,
            wx_date: 20230615,
            lat: 55.0,
            long: -110.0,
            elevation: 500.0,
            slope_pct: 10.0,
            aspect_deg: 270.0,
            ws: case.ws,
            wd: case.wd,
            ffmc: case.ffmc,
            bui: case.bui,
            pc: 50.0,
            pdf: 35.0,
            gfl: 0.35,
            gcf: 80.0,
            percentile_growth: 50.0,
            d0_override: None,
            dj_override: None,
            fmc_override: Some(case.fmc_override),
            hros_override: Some(case.hros_override),
        });
        let actual = [
            result.csfi, result.rso, result.cfb, result.fire_type, result.cfc,
            result.hros, result.tfc, result.hfi, result.fi_class, result.sfc,
            result.fmc,
        ];
        for (name, (a, e)) in ["csfi", "rso", "cfb", "fire_type", "cfc", "hros", "tfc", "hfi", "fi_class", "sfc", "fmc"]
            .iter()
            .zip(actual.iter().zip(case.expected.iter()))
        {
            let ok = (a - e).abs() <= e.abs().max(1e-12) * 1e-9;
            assert!(ok, "case {i} field {name}: expected {e}, got {a}");
        }
        // fme keeps the zero template in override mode
        assert_eq!(result.fme, 0.0, "case {i}: fme must stay 0 in override mode");
    }
}
