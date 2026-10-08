//! Typed grid API: validation errors, empty grids, per-cell parity with `run()`.

use cffdrs_core::fbp::{run, FbpInput};
use cffdrs_core::grid::{run_grid, BehaviourGrids, GridError, GridInput};

// C-2 and M-4, plus fuels that read the per-cell `pct_conifer` (M-1/M-2 =
// 10/11), `grass_curing` and the scalar `grass_fuel_load` (O-1a/O-1b = 14/15)
// and the scalar `pct_dead_fir` (M-3 = 12). Every per-cell input differs from
// cell to cell and from the other inputs, so a swapped field in
// `GridInput::cell` changes some output.
const FUEL: [i32; 7] = [2, 13, 10, 11, 14, 15, 12];
const N: usize = FUEL.len();
const LAT: [f64; N] = [50.0, 52.5, 49.0, 55.5, 51.0, 53.0, 58.0];
const LONG: [f64; N] = [-115.0, -118.25, -112.0, -120.5, -110.0, -116.0, -122.0];
const ELEV: [f64; N] = [1200.0, 800.0, 650.0, 1500.0, 300.0, 950.0, 400.0];
const SLOPE: [f64; N] = [10.0, 25.0, 5.0, 40.0, 0.5, 15.0, 30.0];
const ASPECT: [f64; N] = [180.0, 45.0, 135.0, 300.0, 225.0, 90.0, 10.0];
const PC: [f64; N] = [0.0, 50.0, 25.0, 70.0, 45.0, 85.0, 15.0];
const GC: [f64; N] = [60.0, 80.0, 65.0, 75.0, 92.0, 88.0, 55.0];
const WS: [f64; N] = [15.0, 30.0, 12.0, 22.0, 18.0, 27.0, 9.0];
const WD: [f64; N] = [270.0, 90.0, 200.0, 315.0, 160.0, 20.0, 250.0];

type Setter = for<'a> fn(&mut GridInput<'a>, &'a [f64]);

fn input<'a>() -> GridInput<'a> {
    GridInput {
        fuel_type: &FUEL,
        lat: &LAT,
        long: &LONG,
        elevation: &ELEV,
        slope_pct: &SLOPE,
        aspect_deg: &ASPECT,
        pct_conifer: &PC,
        grass_curing: &GC,
        ws: &WS,
        wd: &WD,
        wx_date: 20_230_715,
        ffmc: 90.0,
        bui: 60.0,
        pct_dead_fir: 30.0,
        grass_fuel_load: 0.35,
        percentile_growth: 50.0,
    }
}

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn fields(g: &BehaviourGrids) -> [(&'static str, &Vec<f64>); 10] {
    [
        ("hros", &g.hros),
        ("bros", &g.bros),
        ("raz", &g.raz),
        ("wsv", &g.wsv),
        ("hfi", &g.hfi),
        ("rso", &g.rso),
        ("sros", &g.sros),
        ("sfc", &g.sfc),
        ("fmc", &g.fmc),
        ("accel", &g.accel),
    ]
}

#[test]
fn mismatched_length_is_reported_for_each_slice() {
    let short = [1.0_f64; 1];
    let cases: [(&str, Setter); 9] = [
        ("lat", |i, s| i.lat = s),
        ("long", |i, s| i.long = s),
        ("elevation", |i, s| i.elevation = s),
        ("slope_pct", |i, s| i.slope_pct = s),
        ("aspect_deg", |i, s| i.aspect_deg = s),
        ("pct_conifer", |i, s| i.pct_conifer = s),
        ("grass_curing", |i, s| i.grass_curing = s),
        ("ws", |i, s| i.ws = s),
        ("wd", |i, s| i.wd = s),
    ];
    for (name, set) in cases {
        let mut inp = input();
        set(&mut inp, &short);
        assert_eq!(
            run_grid(&inp),
            Err(GridError::LengthMismatch {
                name,
                expected: N,
                found: 1
            }),
            "slice {name}"
        );
    }
}

#[test]
fn invalid_date_is_rejected() {
    let mut inp = input();
    inp.wx_date = 20_230_230;
    assert_eq!(run_grid(&inp), Err(GridError::InvalidDate(20_230_230)));
}

#[test]
fn empty_grid_gives_empty_vectors() {
    let mut inp = input();
    let empty_f: [f64; 0] = [];
    let empty_i: [i32; 0] = [];
    inp.fuel_type = &empty_i;
    inp.lat = &empty_f;
    inp.long = &empty_f;
    inp.elevation = &empty_f;
    inp.slope_pct = &empty_f;
    inp.aspect_deg = &empty_f;
    inp.pct_conifer = &empty_f;
    inp.grass_curing = &empty_f;
    inp.ws = &empty_f;
    inp.wd = &empty_f;
    let g = run_grid(&inp).expect("empty grid is valid");
    for (name, v) in fields(&g) {
        assert!(v.is_empty(), "{name}");
    }
}

#[test]
fn cells_match_scalar_run() {
    let inp = input();
    let g = run_grid(&inp).expect("valid grid");
    for i in 0..N {
        let r = run(&FbpInput {
            fuel_type: FUEL[i],
            wx_date: inp.wx_date,
            lat: LAT[i],
            long: LONG[i],
            elevation: ELEV[i],
            slope_pct: SLOPE[i],
            aspect_deg: ASPECT[i],
            ws: WS[i],
            wd: WD[i],
            ffmc: inp.ffmc,
            bui: inp.bui,
            pc: PC[i],
            pdf: inp.pct_dead_fir,
            gfl: inp.grass_fuel_load,
            gcf: GC[i],
            percentile_growth: inp.percentile_growth,
            d0_override: None,
            dj_override: None,
            fmc_override: None,
            hros_override: None,
        });
        let expected = [
            r.hros, r.bros, r.raz, r.wsv, r.hfi, r.rso, r.sros, r.sfc, r.fmc, r.accel,
        ];
        for ((name, v), e) in fields(&g).into_iter().zip(expected) {
            assert!(same(v[i], e), "{name}[{i}]: {} vs {e}", v[i]);
        }
        // every cell is a live modeled fuel, so the comparison above is not NaN == NaN
        assert!(g.hros[i] > 0.0, "fuel {} hros {}", FUEL[i], g.hros[i]);
    }
}

#[test]
fn invalid_date_is_reported_before_a_length_mismatch() {
    let short = [1.0_f64; 1];
    let mut inp = input();
    inp.wx_date = 20_230_230;
    inp.lat = &short;
    assert_eq!(run_grid(&inp), Err(GridError::InvalidDate(20_230_230)));
}

#[test]
fn first_short_slice_in_field_order_is_reported() {
    let short = [1.0_f64; 1];
    let order: [(&str, Setter); 9] = [
        ("lat", |i, s| i.lat = s),
        ("long", |i, s| i.long = s),
        ("elevation", |i, s| i.elevation = s),
        ("slope_pct", |i, s| i.slope_pct = s),
        ("aspect_deg", |i, s| i.aspect_deg = s),
        ("pct_conifer", |i, s| i.pct_conifer = s),
        ("grass_curing", |i, s| i.grass_curing = s),
        ("ws", |i, s| i.ws = s),
        ("wd", |i, s| i.wd = s),
    ];
    for (a, (first, set_first)) in order.iter().enumerate() {
        for (second, set_second) in &order[a + 1..] {
            let mut inp = input();
            set_first(&mut inp, &short);
            set_second(&mut inp, &short);
            assert_eq!(
                run_grid(&inp),
                Err(GridError::LengthMismatch {
                    name: first,
                    expected: N,
                    found: 1
                }),
                "short {first} and {second}"
            );
        }
    }
}

#[test]
fn non_modeled_cells_are_nan() {
    let codes = [19, 20, 0];
    let a = [1.0_f64; 3];
    let inp = GridInput {
        fuel_type: &codes,
        lat: &a,
        long: &a,
        elevation: &a,
        slope_pct: &a,
        aspect_deg: &a,
        pct_conifer: &a,
        grass_curing: &a,
        ws: &a,
        wd: &a,
        ..input()
    };
    let g = run_grid(&inp).expect("valid grid");
    for (name, v) in fields(&g) {
        assert_eq!(v.len(), 3, "{name}");
        assert!(v.iter().all(|x| x.is_nan()), "{name} not all NaN");
    }
}

#[test]
fn error_display_text() {
    let e = GridError::LengthMismatch {
        name: "lat",
        expected: 4,
        found: 3,
    };
    assert_eq!(e.to_string(), "lat has 3 values but fuel_type has 4");
    assert_eq!(
        GridError::InvalidDate(20_230_230).to_string(),
        "wx_date 20230230 is not a valid YYYYMMDD calendar date"
    );
    let _: &dyn std::error::Error = &e;
}
