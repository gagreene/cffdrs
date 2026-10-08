//! Override-mode validation: the FBP chain with fmc and hros injected via
//! setParams — the sequence fire-growth engines use to recompute output
//! quantities from an engine-supplied ROS:
//!
//!   initialize -> invertWindAspect -> calcSF -> calcISZ -> setParams(fmc)
//!   -> `calcISI_RSI_BE` -> calcSFC -> `getCBH_CFL` -> calcROS -> setParams(hros)
//!   -> calcCSFI -> calcRSO -> calcC6BlendCFB -> calcC6BlendCFC -> calcC6CROS
//!   -> calcC6HROS -> calcPercentileCFB -> calcRosPercentileGrowth -> calcCFB
//!   -> calcFireType -> calcCFC -> calcTFC -> calcHFI -> calcFireIntensityClass
//!
//! Notable spec behaviours these goldens pin (captured verbatim from the
//! Python package running that exact sequence):
//! - calcFMC never runs, so fme keeps its zero template value — C-6 crown
//!   ROS is therefore 0 and the C-6 blend collapses to sros * (1 - cfb),
//!   OVERWRITING the injected hros (case 3).
//! - C-6 blends from an SROS-derived temporary cfb, then the FINAL cfb is
//!   recomputed from the blended hros (case 2: hros 1.378 is far below rso, so
//!   cfb is 0 and the cell is a surface fire); other crowning fuels use the
//!   injected hros.
//! - hfi = 0 yields `fi_class` -99 (case 5).

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
        fuel_type: 2,
        ws: 20.0,
        wd: 0.0,
        ffmc: 91.0,
        bui: 76.0,
        fmc_override: 97.5,
        hros_override: 4.0,
        expected: [
            847.525_829_130_436_7,
            0.969_618_920_338_633_1,
            0.501_916_551_298_714_5,
            2.0,
            0.401_533_241_038_971_6,
            4.0,
            3.315_137_790_045_516,
            3_978.165_348_054_619_2,
            4.0,
            2.913_604_549_006_544_5,
            97.5,
        ],
    },
    Case {
        fuel_type: 2,
        ws: 20.0,
        wd: 0.0,
        ffmc: 91.0,
        bui: 76.0,
        fmc_override: 97.5,
        hros_override: 60.0,
        expected: [
            847.525_829_130_436_7,
            0.969_618_920_338_633_1,
            0.999_998_730_627_213_5,
            3.0,
            0.799_998_984_501_770_8,
            60.0,
            3.713_603_533_508_315_3,
            66_844.863_603_149_67,
            6.0,
            2.913_604_549_006_544_5,
            97.5,
        ],
    },
    Case {
        fuel_type: 6,
        ws: 25.0,
        wd: 90.0,
        ffmc: 92.5,
        bui: 85.0,
        fmc_override: 105.0,
        hros_override: 10.0,
        expected: [
            3_320.360_999_080_089,
            5.030_716_309_746_868,
            0.0,
            1.0,
            0.0,
            1.378_236_428_950_598_8,
            2.200_058_463_938_813_4,
            909.660_216_246_471,
            3.0,
            2.200_058_463_938_813_4,
            105.0,
        ],
    },
    Case {
        fuel_type: 14,
        ws: 30.0,
        wd: 180.0,
        ffmc: 90.0,
        bui: 60.0,
        fmc_override: 120.0,
        hros_override: 25.0,
        expected: [
            0.0, 0.0, 0.0, 1.0, 0.0, 25.0, 0.35, 2625.0, 4.0, 0.35, 120.0,
        ],
    },
    Case {
        fuel_type: 10,
        ws: 15.0,
        wd: 270.0,
        ffmc: 88.0,
        bui: 70.0,
        fmc_override: 99.0,
        hros_override: 0.0,
        expected: [
            2_444.111_969_224_106,
            4.234_514_865_780_585,
            0.0,
            1.0,
            0.0,
            0.0,
            1.923_960_632_007_811_3,
            0.0,
            -99.0,
            1.923_960_632_007_811_3,
            99.0,
        ],
    },
];

#[test]
fn override_chain_matches_python_sequence() {
    for (i, case) in CASES.iter().enumerate() {
        let result = run(&FbpInput {
            fuel_type: case.fuel_type,
            wx_date: 20_230_615,
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
            result.csfi,
            result.rso,
            result.cfb,
            result.fire_type,
            result.cfc,
            result.hros,
            result.tfc,
            result.hfi,
            result.fi_class,
            result.sfc,
            result.fmc,
        ];
        for (name, (a, e)) in [
            "csfi",
            "rso",
            "cfb",
            "fire_type",
            "cfc",
            "hros",
            "tfc",
            "hfi",
            "fi_class",
            "sfc",
            "fmc",
        ]
        .iter()
        .zip(actual.iter().zip(case.expected.iter()))
        {
            let ok = (a - e).abs() <= e.abs().max(1e-12) * 1e-9;
            assert!(ok, "case {i} field {name}: expected {e}, got {a}");
        }
        // fme keeps the zero template in override mode
        assert_eq!(
            result.fme, 0.0,
            "case {i}: fme must stay 0 in override mode"
        );
    }
}
