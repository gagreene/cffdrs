//! Typed grid API: per-cell or constant inputs, validation errors, empty grids,
//! requested fields, `d0` / `dj` overrides and per-cell parity with `run()`.

use cffbps_rust::fbp::{run, FbpInput};
use cffbps_rust::field::Field;
use cffbps_rust::grid::{
    run_grid, run_grid_fields, BehaviourGrids, GridError, GridFields, GridInput, NonModeled,
    PerCell,
};
use cffbps_rust::stage::Stage;
use cffbps_rust::state::CellState;

// C-2 and M-4, plus fuels that read the per-cell `pct_conifer` (M-1/M-2 =
// 10/11), `grass_curing` and `grass_fuel_load` (O-1a/O-1b = 14/15),
// `pct_dead_fir` (M-3 = 12), the C-6 blend (6) and the C-1 crown-only
// percentile rule (1). Every per-cell input differs from cell to cell and from
// the other inputs, so a swapped field in the grid's cell builder changes some
// output.
const FUEL: [i32; 9] = [2, 13, 10, 11, 14, 15, 12, 6, 1];
const N: usize = FUEL.len();
const LAT: [f64; N] = [50.0, 52.5, 49.0, 55.5, 51.0, 53.0, 58.0, 54.0, 56.5];
const LONG: [f64; N] = [
    -115.0, -118.25, -112.0, -120.5, -110.0, -116.0, -122.0, -105.0, -99.0,
];
const ELEV: [f64; N] = [
    1200.0, 800.0, 650.0, 1500.0, 300.0, 950.0, 400.0, 0.0, 700.0,
];
const SLOPE: [f64; N] = [10.0, 25.0, 5.0, 40.0, 0.5, 15.0, 30.0, 12.0, 3.0];
const ASPECT: [f64; N] = [180.0, 45.0, 135.0, 300.0, 225.0, 90.0, 10.0, 200.0, 330.0];
const PC: [f64; N] = [0.0, 50.0, 25.0, 70.0, 45.0, 85.0, 15.0, 35.0, 65.0];
const PDF: [f64; N] = [5.0, 40.0, 10.0, 20.0, 30.0, 60.0, 35.0, 15.0, 25.0];
const GFL: [f64; N] = [0.3, 0.35, 0.4, 0.2, 0.45, 0.5, 0.25, 0.33, 0.38];
const GC: [f64; N] = [60.0, 80.0, 65.0, 75.0, 92.0, 88.0, 55.0, 70.0, 95.0];
const WS: [f64; N] = [15.0, 30.0, 12.0, 22.0, 18.0, 27.0, 9.0, 33.0, 24.0];
const WD: [f64; N] = [270.0, 90.0, 200.0, 315.0, 160.0, 20.0, 250.0, 120.0, 45.0];
const FFMC: [f64; N] = [90.0, 92.5, 88.0, 91.0, 93.0, 89.5, 94.0, 90.5, 92.0];
const BUI: [f64; N] = [60.0, 85.0, 40.0, 110.0, 70.0, 55.0, 95.0, 120.0, 75.0];

/// The 54 outputs, in `FbpResult::named_values` order.
fn outputs() -> Vec<Field> {
    Field::ALL.into_iter().filter(|f| f.is_output()).collect()
}

/// A modeled-fuel grid with every input per cell.
fn input<'a>() -> GridInput<'a> {
    GridInput {
        fuel_type: &FUEL,
        lat: (&LAT).into(),
        long: (&LONG).into(),
        elevation: (&ELEV).into(),
        slope_pct: (&SLOPE).into(),
        aspect_deg: (&ASPECT).into(),
        pct_conifer: (&PC).into(),
        pct_dead_fir: (&PDF).into(),
        grass_fuel_load: (&GFL).into(),
        grass_curing: (&GC).into(),
        ws: (&WS).into(),
        wd: (&WD).into(),
        ffmc: (&FFMC).into(),
        bui: (&BUI).into(),
        wx_date: 20_230_715,
        percentile_growth: 90.0,
        d0: None,
        dj: None,
        fmc: None,
    }
}

fn at(p: PerCell<'_>, i: usize) -> f64 {
    match p {
        PerCell::Const(v) => v,
        PerCell::Cells(s) => s[i],
    }
}

/// The scalar input of cell `i`; `dj` is the already-resolved override.
fn cell(inp: &GridInput<'_>, i: usize, dj: Option<f64>) -> FbpInput {
    FbpInput {
        fuel_type: inp.fuel_type[i],
        wx_date: inp.wx_date,
        lat: at(inp.lat, i),
        long: at(inp.long, i),
        elevation: at(inp.elevation, i),
        slope_pct: at(inp.slope_pct, i),
        aspect_deg: at(inp.aspect_deg, i),
        ws: at(inp.ws, i),
        wd: at(inp.wd, i),
        ffmc: at(inp.ffmc, i),
        bui: at(inp.bui, i),
        pc: at(inp.pct_conifer, i),
        pdf: at(inp.pct_dead_fir, i),
        gfl: at(inp.grass_fuel_load, i),
        gcf: at(inp.grass_curing, i),
        percentile_growth: inp.percentile_growth,
        d0_override: inp.d0.map(|p| at(p, i)),
        dj_override: dj,
        fmc_override: inp.fmc.map(|p| at(p, i)),
        hros_override: None,
    }
}

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn assert_same_fields(a: &GridFields, b: &GridFields, what: &str) {
    assert_eq!(a.fields.len(), b.fields.len(), "{what}: field count");
    for ((fa, va), (fb, vb)) in a.fields.iter().zip(&b.fields) {
        assert_eq!(fa, fb, "{what}: field order");
        assert_eq!(va.len(), vb.len(), "{what}: {} length", fa.name());
        for (i, (x, y)) in va.iter().zip(vb).enumerate() {
            assert!(same(*x, *y), "{what}: {}[{i}] {x:?} vs {y:?}", fa.name());
        }
    }
    assert_eq!(
        a.dj_override_clipped, b.dj_override_clipped,
        "{what}: clipped count"
    );
}

fn behaviour(g: &BehaviourGrids) -> [(&'static str, &Vec<f64>); 10] {
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

type Setter = for<'a> fn(&mut GridInput<'a>, PerCell<'a>);

/// Every per-cell input of `GridInput`, in field order, with a setter.
const PER_CELL_INPUTS: [(&str, Setter); 15] = [
    ("lat", |i, p| i.lat = p),
    ("long", |i, p| i.long = p),
    ("elevation", |i, p| i.elevation = p),
    ("slope_pct", |i, p| i.slope_pct = p),
    ("aspect_deg", |i, p| i.aspect_deg = p),
    ("pct_conifer", |i, p| i.pct_conifer = p),
    ("pct_dead_fir", |i, p| i.pct_dead_fir = p),
    ("grass_fuel_load", |i, p| i.grass_fuel_load = p),
    ("grass_curing", |i, p| i.grass_curing = p),
    ("ws", |i, p| i.ws = p),
    ("wd", |i, p| i.wd = p),
    ("ffmc", |i, p| i.ffmc = p),
    ("bui", |i, p| i.bui = p),
    ("d0", |i, p| i.d0 = Some(p)),
    ("dj", |i, p| i.dj = Some(p)),
];

// ---------------------------------------------------------------- errors

#[test]
fn mismatched_length_is_reported_for_each_slice() {
    let short = [1.0_f64; 1];
    for (name, set) in PER_CELL_INPUTS {
        let mut inp = input();
        set(&mut inp, PerCell::Cells(&short));
        assert_eq!(
            run_grid(&inp),
            Err(GridError::LengthMismatch {
                name,
                expected: N,
                found: 1
            }),
            "slice {name}"
        );
        assert_eq!(
            run_grid_fields(&inp, &[Field::Hros], 1, NonModeled::PythonCompat),
            Err(GridError::LengthMismatch {
                name,
                expected: N,
                found: 1
            }),
            "slice {name} (run_grid_fields)"
        );
    }
}

#[test]
fn a_constant_never_mismatches() {
    for (name, set) in PER_CELL_INPUTS {
        let mut inp = input();
        set(&mut inp, PerCell::Const(100.0));
        assert!(run_grid(&inp).is_ok(), "{name}");
    }
}

#[test]
fn invalid_date_is_rejected() {
    let mut inp = input();
    inp.wx_date = 20_230_230;
    assert_eq!(run_grid(&inp), Err(GridError::InvalidDate(20_230_230)));
}

#[test]
fn invalid_date_is_reported_before_a_length_mismatch() {
    let short = [1.0_f64; 1];
    let mut inp = input();
    inp.wx_date = 20_230_230;
    inp.lat = (&short).into();
    assert_eq!(run_grid(&inp), Err(GridError::InvalidDate(20_230_230)));
}

#[test]
fn first_short_slice_in_field_order_is_reported() {
    let short = [1.0_f64; 1];
    for (a, (first, set_first)) in PER_CELL_INPUTS.iter().enumerate() {
        for (second, set_second) in &PER_CELL_INPUTS[a + 1..] {
            let mut inp = input();
            set_first(&mut inp, PerCell::Cells(&short));
            set_second(&mut inp, PerCell::Cells(&short));
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
    assert_eq!(
        GridError::InfiniteDj.to_string(),
        "dj override must be finite (it contains infinity)"
    );
    let _: &dyn std::error::Error = &e;
}

// ------------------------------------------------------- engine contract

#[test]
fn empty_grid_gives_empty_vectors() {
    let empty_f: [f64; 0] = [];
    let empty_i: [i32; 0] = [];
    let mut inp = input();
    inp.fuel_type = &empty_i;
    for (_, set) in &PER_CELL_INPUTS[..13] {
        set(&mut inp, PerCell::Cells(&empty_f));
    }
    let g = run_grid(&inp).expect("empty grid is valid");
    for (name, v) in behaviour(&g) {
        assert!(v.is_empty(), "{name}");
    }
    let f = run_grid_fields(&inp, &Field::ALL, 0, NonModeled::PythonCompat).expect("valid");
    assert_eq!(f.fields.len(), Field::COUNT);
    assert!(f.fields.iter().all(|(_, v)| v.is_empty()));
}

#[test]
fn cells_match_scalar_run() {
    let inp = input();
    let g = run_grid(&inp).expect("valid grid");
    for i in 0..N {
        let r = run(&cell(&inp, i, None));
        let expected = [
            r.hros, r.bros, r.raz, r.wsv, r.hfi, r.rso, r.sros, r.sfc, r.fmc, r.accel,
        ];
        for ((name, v), e) in behaviour(&g).into_iter().zip(expected) {
            assert!(same(v[i], e), "{name}[{i}]: {} vs {e}", v[i]);
        }
        // every cell is a live modeled fuel, so the comparison above is not NaN == NaN
        assert!(g.hros[i] > 0.0, "fuel {} hros {}", FUEL[i], g.hros[i]);
    }
}

#[test]
fn non_modeled_cells_are_nan() {
    let codes = [19, 20, 0];
    let a = [1.0_f64; 3];
    let mut inp = GridInput {
        fuel_type: &codes,
        ..input()
    };
    for (_, set) in &PER_CELL_INPUTS[..13] {
        set(&mut inp, PerCell::Cells(&a));
    }
    let g = run_grid(&inp).expect("valid grid");
    for (name, v) in behaviour(&g) {
        assert_eq!(v.len(), 3, "{name}");
        assert!(v.iter().all(|x| x.is_nan()), "{name} not all NaN");
    }
}

// ------------------------------------------------- (a) Const == filled Cells

#[test]
fn a_constant_equals_a_slice_filled_with_it() {
    // A value each input can meaningfully take (a NaN would hide a mix-up).
    let values = [
        53.0, -113.0, 900.0, 20.0, 135.0, 40.0, 25.0, 0.4, 85.0, 18.0, 225.0, 91.0, 80.0, 150.0,
        170.7,
    ];
    for ((name, set), x) in PER_CELL_INPUTS.into_iter().zip(values) {
        let filled = vec![x; N];
        let mut constant = input();
        set(&mut constant, PerCell::Const(x));
        let mut cells = input();
        set(&mut cells, PerCell::Cells(&filled));
        for fields in [outputs(), Field::ALL.to_vec()] {
            for threads in [1, 2] {
                let a = run_grid_fields(&constant, &fields, threads, NonModeled::PythonCompat)
                    .expect("valid");
                let b = run_grid_fields(&cells, &fields, threads, NonModeled::PythonCompat)
                    .expect("valid");
                assert_same_fields(&a, &b, name);
            }
        }
    }
}

// -------------------------------------- (b) all 54 outputs equal run() per cell

#[test]
fn all_outputs_equal_run_for_every_thread_count() {
    let inp = input();
    let fields = outputs();
    for threads in [0, 1, 2, 7] {
        let g = run_grid_fields(&inp, &fields, threads, NonModeled::Nan).expect("valid");
        assert_eq!(g.fields.len(), 54);
        for i in 0..N {
            let r = run(&cell(&inp, i, None)).named_values();
            for ((f, v), (name, e)) in g.fields.iter().zip(r) {
                assert_eq!(f.name(), name);
                assert!(
                    same(v[i], e),
                    "threads {threads} {name}[{i}]: {} vs {e}",
                    v[i]
                );
            }
        }
    }
}

#[test]
fn the_stage_path_gives_the_same_outputs() {
    // Asking for a non-output field computes the cells stage by stage.
    let inp = input();
    let fused = run_grid_fields(&inp, &outputs(), 1, NonModeled::PythonCompat).expect("valid");
    let mut with_internal = outputs();
    with_internal.push(Field::Isz);
    let staged = run_grid_fields(&inp, &with_internal, 1, NonModeled::PythonCompat).expect("valid");
    let staged = GridFieldsView(&staged);
    for (f, v) in &fused.fields {
        let s = staged.get(*f);
        for i in 0..N {
            assert!(same(v[i], s[i]), "{}[{i}]", f.name());
        }
    }
}

struct GridFieldsView<'a>(&'a GridFields);
impl GridFieldsView<'_> {
    fn get(&self, f: Field) -> &[f64] {
        self.0.get(f).expect("requested")
    }
}

// --------------------------------------------- (c) a subset, in request order

#[test]
fn a_subset_comes_back_alone_in_the_requested_order() {
    let inp = input();
    let want = [Field::Hfi, Field::Ws, Field::Hros, Field::FiClass];
    let g = run_grid_fields(&inp, &want, 1, NonModeled::Nan).expect("valid");
    let got: Vec<Field> = g.fields.iter().map(|(f, _)| *f).collect();
    assert_eq!(got, want);
    let all = run_grid_fields(&inp, &outputs(), 1, NonModeled::Nan).expect("valid");
    for (f, v) in &g.fields {
        let full = all.get(*f).expect("output");
        assert!(v.iter().zip(full).all(|(a, b)| same(*a, *b)), "{f:?}");
    }
    let none = run_grid_fields(&inp, &[], 1, NonModeled::Nan).expect("valid");
    assert_eq!(none.fields, []);
}

// --------------------- (e) input-only and internal fields; duplicates collapse

#[test]
fn input_only_and_internal_fields_are_readable() {
    let mut inp = input();
    let d0 = [
        140.0,
        f64::NAN,
        160.0,
        150.0,
        151.0,
        152.0,
        153.0,
        154.0,
        155.0,
    ];
    inp.d0 = Some((&d0).into());
    inp.dj = Some(PerCell::Const(170.6));
    inp.fmc = Some(PerCell::Const(101.5));
    let extra: Vec<Field> = Field::ALL.into_iter().filter(|f| !f.is_output()).collect();
    assert_eq!(extra.len(), 22);
    let g = run_grid_fields(&inp, &extra, 1, NonModeled::PythonCompat).expect("valid");
    for i in 0..N {
        let mut s = CellState::from_input(&cell(&inp, i, Some(170.0)));
        for stage in Stage::ALL {
            s.apply(stage);
        }
        for (f, v) in &g.fields {
            assert!(
                same(v[i], s.get(*f)),
                "{}[{i}]: {} vs {}",
                f.name(),
                v[i],
                s.get(*f)
            );
        }
    }
    let view = GridFieldsView(&g);
    assert_eq!(view.get(Field::Long)[0], 115.0, "long is its magnitude");
    assert_eq!(
        view.get(Field::DjOverride)[0],
        170.0,
        "the resolved dj override"
    );
    assert!(
        view.get(Field::D0Override)[1].is_nan(),
        "a masked d0 override"
    );
    assert_eq!(
        view.get(Field::FmcOverride)[0],
        101.5,
        "the fmc override as supplied"
    );
}

#[test]
fn duplicate_fields_collapse_to_their_first_position() {
    let inp = input();
    let g = run_grid_fields(
        &inp,
        &[Field::Hros, Field::Fmc, Field::Hros, Field::Lat, Field::Fmc],
        1,
        NonModeled::Nan,
    )
    .expect("valid");
    let got: Vec<Field> = g.fields.iter().map(|(f, _)| *f).collect();
    assert_eq!(got, [Field::Hros, Field::Fmc, Field::Lat]);
}

// ----------------------------------------- (h) the dj override rule (D15.6)

/// (supplied, resolved in a common year, clipped there, resolved in a leap
/// year, clipped there): floor, then clip to 1..365 / 1..366.
const DJ_RULE: [(f64, f64, bool, f64, bool); 16] = [
    (0.3, 1.0, true, 1.0, true),
    (0.0, 1.0, true, 1.0, true),
    (-5.0, 1.0, true, 1.0, true),
    (-0.5, 1.0, true, 1.0, true),
    (1.0, 1.0, false, 1.0, false),
    (1.999, 1.0, false, 1.0, false),
    (166.9, 166.0, false, 166.0, false),
    (166.0, 166.0, false, 166.0, false),
    (365.0, 365.0, false, 365.0, false),
    (365.5, 365.0, false, 365.0, false),
    (366.0, 365.0, true, 366.0, false),
    (366.7, 365.0, true, 366.0, false),
    (367.0, 365.0, true, 366.0, true),
    (400.0, 365.0, true, 366.0, true),
    (1e300, 365.0, true, 366.0, true),
    (-1e300, 1.0, true, 1.0, true),
];

#[test]
fn dj_overrides_are_floored_then_clipped_to_the_year() {
    for (wx_date, leap) in [
        (19_990_615, false),
        (20_000_615, true),
        (20_230_615, false),
        (20_240_615, true),
        (21_000_615, false),
    ] {
        let supplied: Vec<f64> = DJ_RULE.iter().map(|r| r.0).chain([f64::NAN]).collect();
        let n = supplied.len();
        let fuel = vec![2; n];
        let mut inp = input();
        inp.fuel_type = &fuel;
        for (_, set) in &PER_CELL_INPUTS[..13] {
            set(&mut inp, PerCell::Const(50.0));
        }
        inp.lat = PerCell::Const(55.0);
        inp.long = PerCell::Const(-110.0);
        inp.wx_date = wx_date;
        inp.dj = Some(PerCell::Cells(&supplied));
        let g = run_grid_fields(
            &inp,
            &[Field::Dj, Field::DjOverride, Field::Fmc, Field::Fme],
            1,
            NonModeled::PythonCompat,
        )
        .expect("valid");
        let view = GridFieldsView(&g);
        let mut clipped = 0;
        for (i, &(v, common, c_clip, leap_day, l_clip)) in DJ_RULE.iter().enumerate() {
            let (want, was_clipped) = if leap {
                (leap_day, l_clip)
            } else {
                (common, c_clip)
            };
            clipped += usize::from(was_clipped);
            assert_eq!(view.get(Field::Dj)[i], want, "{wx_date}: dj {v}");
            assert_eq!(
                view.get(Field::DjOverride)[i],
                want,
                "{wx_date}: override {v}"
            );
        }
        // the NaN cell stays missing and is not counted
        let last = n - 1;
        assert!(view.get(Field::Dj)[last].is_nan());
        assert!(view.get(Field::Fmc)[last].is_nan() && view.get(Field::Fme)[last].is_nan());
        assert_eq!(g.dj_override_clipped, clipped, "{wx_date}");
    }
}

#[test]
fn a_constant_dj_override_is_one_supplied_value() {
    let mut inp = input();
    inp.dj = Some(PerCell::Const(400.0));
    let g = run_grid_fields(&inp, &[Field::Dj], 1, NonModeled::Nan).expect("valid");
    assert_eq!(g.dj_override_clipped, 1);
    assert!(g.fields[0].1.iter().all(|&d| d == 365.0));
    inp.dj = Some(PerCell::Const(200.9));
    let g = run_grid_fields(&inp, &[Field::Dj], 1, NonModeled::Nan).expect("valid");
    assert_eq!(g.dj_override_clipped, 0, "flooring alone is not clipping");
    inp.dj = None;
    let g = run_grid_fields(&inp, &[Field::Dj], 1, NonModeled::Nan).expect("valid");
    assert_eq!(g.dj_override_clipped, 0);
}

#[test]
fn an_infinite_dj_override_is_rejected() {
    for bad in [f64::INFINITY, f64::NEG_INFINITY] {
        let mut inp = input();
        inp.dj = Some(PerCell::Const(bad));
        assert_eq!(run_grid(&inp), Err(GridError::InfiniteDj), "{bad}");
        let mut cells = [170.0; N];
        cells[4] = bad;
        inp.dj = Some((&cells).into());
        assert_eq!(
            run_grid_fields(&inp, &[Field::Fmc], 1, NonModeled::PythonCompat),
            Err(GridError::InfiniteDj),
            "{bad}"
        );
    }
    // an infinite d0 is not a date and passes through (D15.7)
    let mut inp = input();
    inp.d0 = Some(PerCell::Const(f64::INFINITY));
    assert!(run_grid(&inp).is_ok());
}

#[test]
fn a_length_mismatch_is_reported_before_an_infinite_dj() {
    let short = [1.0_f64; 1];
    let mut inp = input();
    inp.lat = (&short).into();
    inp.dj = Some(PerCell::Const(f64::INFINITY));
    assert!(matches!(
        run_grid(&inp),
        Err(GridError::LengthMismatch { name: "lat", .. })
    ));
}

// ------------------------- (g) d0 / dj overrides equal run() with the same values

#[test]
fn overrides_equal_run_with_the_same_scalar_overrides() {
    let d0_cells = [
        140.0,
        f64::NAN,
        160.0,
        150.0,
        400.0,
        0.0,
        153.0,
        154.0,
        155.0,
    ];
    let dj_cells = [
        170.0,
        180.0,
        f64::NAN,
        200.4,
        160.0,
        150.0,
        1.0,
        366.0,
        90.9,
    ];
    let dj_resolved = [
        170.0,
        180.0,
        f64::NAN,
        200.0,
        160.0,
        150.0,
        1.0,
        365.0,
        90.0,
    ];
    let cases: [(Option<PerCell<'_>>, Option<PerCell<'_>>, &str); 7] = [
        (Some(PerCell::Const(150.0)), None, "constant d0"),
        (Some((&d0_cells).into()), None, "per-cell d0"),
        (Some(PerCell::Const(f64::NAN)), None, "masked d0"),
        (None, Some(PerCell::Const(175.0)), "constant dj"),
        (None, Some((&dj_cells).into()), "per-cell dj"),
        (None, Some(PerCell::Const(f64::NAN)), "masked dj"),
        (Some((&d0_cells).into()), Some((&dj_cells).into()), "both"),
    ];
    for (d0, dj, what) in cases {
        let mut inp = input();
        inp.d0 = d0;
        inp.dj = dj;
        for threads in [1, 3] {
            let g = run_grid_fields(&inp, &outputs(), threads, NonModeled::Nan).expect("valid");
            for i in 0..N {
                let resolved = dj.map(|p| match p {
                    PerCell::Const(v) => v,
                    PerCell::Cells(_) => dj_resolved[i],
                });
                let r = run(&cell(&inp, i, resolved)).named_values();
                for ((f, v), (name, e)) in g.fields.iter().zip(r) {
                    assert_eq!(f.name(), name);
                    assert!(same(v[i], e), "{what} {name}[{i}]: {} vs {e}", v[i]);
                }
                let masked =
                    d0.is_some_and(|p| at(p, i).is_nan()) || resolved.is_some_and(f64::is_nan);
                let view = GridFieldsView(&g);
                let (fmc, fme) = (view.get(Field::Fmc)[i], view.get(Field::Fme)[i]);
                assert_eq!(fmc.is_nan(), masked, "{what} fmc[{i}] {fmc}");
                assert_eq!(fme.is_nan(), masked, "{what} fme[{i}] {fme}");
            }
        }
    }
}

// ---------------------------------- (f) thread counts and the parallel path

/// A grid of `n` cells cycling through modeled fuels, non-fuel, water and
/// invalid codes, with NaN holes in several inputs and per-cell overrides.
struct Big {
    fuel: Vec<i32>,
    cols: Vec<Vec<f64>>,
}

const BIG_FUEL: [i32; 13] = [2, 13, 10, 11, 14, 15, 12, 19, 7, 6, 0, 20, -1];

fn big(n: usize) -> Big {
    let fuel: Vec<i32> = (0..n).map(|i| BIG_FUEL[i % BIG_FUEL.len()]).collect();
    let f = |k: usize, scale: f64, off: f64| -> Vec<f64> {
        (0..n)
            .map(|i| off + scale * f64::from(u32::try_from((i * (k + 3)) % 97).unwrap()) / 97.0)
            .collect()
    };
    let mut cols = vec![
        f(0, 12.0, 48.0),    // lat
        f(1, 30.0, -125.0),  // long
        f(2, 1500.0, 100.0), // elevation
        f(3, 60.0, 0.0),     // slope
        f(4, 360.0, 0.0),    // aspect
        f(5, 100.0, 0.0),    // pct_conifer
        f(6, 100.0, 0.0),    // grass_curing
        f(7, 45.0, 0.0),     // ws
        f(8, 360.0, 0.0),    // wd
        f(9, 30.0, 70.0),    // ffmc
        f(10, 150.0, 0.0),   // bui
        f(11, 400.0, -5.0),  // dj override (clips at both ends)
    ];
    for (col, step) in [(0, 311), (7, 509), (9, 701), (11, 97)] {
        for i in (col..n).step_by(step) {
            cols[col][i] = f64::NAN;
        }
    }
    Big { fuel, cols }
}

fn big_fields(b: &Big, threads: usize, non_modeled: NonModeled) -> GridFields {
    let c = &b.cols;
    let input = GridInput {
        fuel_type: &b.fuel,
        lat: (&c[0]).into(),
        long: (&c[1]).into(),
        elevation: (&c[2]).into(),
        slope_pct: (&c[3]).into(),
        aspect_deg: (&c[4]).into(),
        pct_conifer: (&c[5]).into(),
        pct_dead_fir: PerCell::Const(35.0),
        grass_fuel_load: PerCell::Const(0.35),
        grass_curing: (&c[6]).into(),
        ws: (&c[7]).into(),
        wd: (&c[8]).into(),
        ffmc: (&c[9]).into(),
        bui: (&c[10]).into(),
        wx_date: 20_240_615,
        percentile_growth: 90.0,
        d0: Some(PerCell::Const(150.0)),
        dj: Some((&c[11]).into()),
        fmc: None,
    };
    run_grid_fields(&input, &outputs(), threads, non_modeled).expect("valid")
}

#[test]
fn every_thread_count_gives_identical_arrays_on_the_parallel_path() {
    for n in [4_096, 4_097, 20_011, 100_003] {
        let b = big(n);
        for non_modeled in [NonModeled::Nan, NonModeled::PythonCompat] {
            let serial = big_fields(&b, 1, non_modeled);
            let hros = serial.get(Field::Hros).unwrap();
            assert!(hros.iter().any(|v| v.is_finite()), "n={n}: real results");
            assert!(hros.iter().any(|v| v.is_nan()), "n={n}: NaN cells");
            assert!(serial.dj_override_clipped > 0, "n={n}: clipping counted");
            for threads in [0, 2, 7, 64] {
                assert_same_fields(
                    &serial,
                    &big_fields(&b, threads, non_modeled),
                    &format!("n={n} threads={threads} {non_modeled:?}"),
                );
            }
        }
    }
}

#[test]
fn small_grids_run_serially_with_identical_results() {
    for n in [0, 1, 4_095] {
        let b = big(n);
        let serial = big_fields(&b, 1, NonModeled::PythonCompat);
        assert_eq!(serial.fields[0].1.len(), n);
        for threads in [0, 2, 7, 64] {
            assert_same_fields(
                &serial,
                &big_fields(&b, threads, NonModeled::PythonCompat),
                &format!("n={n} threads={threads}"),
            );
        }
    }
}
