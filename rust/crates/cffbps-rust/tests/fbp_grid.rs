//! `FbpGrid`: the pipeline stages over whole grids, freeing arrays once no
//! planned stage needs them, against the fused pass and the scalar `run()`.

use cffbps_rust::fbp::{run, FbpInput};
use cffbps_rust::fbp_grid::{FbpGrid, ParamName, StageError};
use cffbps_rust::field::Field;
use cffbps_rust::grid::{GridError, GridInput, PerCell};
use cffbps_rust::stage::Stage;

/// Every fuel: the 18 modeled codes, 19/20 and invalid codes.
const MIXED: [i32; 23] = [
    2, 13, 10, 11, 14, 15, 12, 6, 1, 7, 19, 20, 0, 99, 3, 4, 5, 8, 9, 16, 17, 18, -1,
];

/// Modeled fuels only, including C-6 (6).
const MODELED: [i32; 9] = [2, 6, 10, 12, 14, 1, 7, 6, 13];

// column indices of `Data::cols`, in `GridInput` order
const LAT: usize = 0;

const LONG: usize = 1;

const ELEV: usize = 2;

const SLOPE: usize = 3;

const ASPECT: usize = 4;

const WS: usize = 9;

const WD: usize = 10;

/// Owned per-cell inputs of a grid.
#[derive(Clone)]
struct Data {
    fuel: Vec<i32>,
    /// lat, long, elevation, slope, aspect, pc, pdf, gfl, gcf, ws, wd, ffmc, bui
    cols: [Vec<f64>; 13],
    wx_date: i64,
}

impl Data {
    /// The scalar input of cell `i`.
    fn cell(&self, i: usize, percentile: f64) -> FbpInput {
        let c = &self.cols;
        FbpInput {
            fuel_type: self.fuel[i],
            wx_date: self.wx_date,
            lat: c[0][i],
            long: c[1][i],
            elevation: c[2][i],
            slope_pct: c[3][i],
            aspect_deg: c[4][i],
            ws: c[9][i],
            wd: c[10][i],
            ffmc: c[11][i],
            bui: c[12][i],
            pc: c[5][i],
            pdf: c[6][i],
            gfl: c[7][i],
            gcf: c[8][i],
            percentile_growth: percentile,
            d0_override: None,
            dj_override: None,
            fmc_override: None,
            hros_override: None,
        }
    }

    fn grid(&self, percentile: f64, out: &[Field]) -> FbpGrid {
        FbpGrid::new(&self.input(percentile), out).unwrap()
    }

    fn input(&self, percentile: f64) -> GridInput<'_> {
        let c = &self.cols;
        GridInput {
            fuel_type: &self.fuel,
            lat: (&c[0]).into(),
            long: (&c[1]).into(),
            elevation: (&c[2]).into(),
            slope_pct: (&c[3]).into(),
            aspect_deg: (&c[4]).into(),
            pct_conifer: (&c[5]).into(),
            pct_dead_fir: (&c[6]).into(),
            grass_fuel_load: (&c[7]).into(),
            grass_curing: (&c[8]).into(),
            ws: (&c[9]).into(),
            wd: (&c[10]).into(),
            ffmc: (&c[11]).into(),
            bui: (&c[12]).into(),
            wx_date: self.wx_date,
            percentile_growth: percentile,
            d0: None,
            dj: None,
            fmc: None,
        }
    }
}

const FMC_FIELDS: [Field; 6] = [
    Field::Latn,
    Field::D0,
    Field::Dj,
    Field::Nd,
    Field::Fmc,
    Field::Fme,
];

// ---- helpers ----
fn assert_fmc_same(a: &[Vec<f64>], b: &[Vec<f64>], what: &str) {
    for ((f, x), y) in FMC_FIELDS.iter().zip(a).zip(b) {
        assert_same(x, y, &format!("{what}: {}", f.name()));
    }
}

/// `fields` of two grids are bit-identical.
fn assert_grids_same(a: &FbpGrid, b: &FbpGrid, fields: &[Field], what: &str) {
    for &f in fields {
        assert_same(
            &a.get(f).unwrap(),
            &b.get(f).unwrap(),
            &format!("{what}: {}", f.name()),
        );
    }
}

fn assert_same(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(same(*x, *y), "{what}[{i}]: {x:?} vs {y:?}");
    }
}

/// The stage engine and the fused pass agree on every output of `g`'s inputs.
fn assert_staged_equals_fused(g: &FbpGrid, what: &str) {
    let mut s = g.clone();
    s.reset();
    s.run_planned(1).unwrap();
    let mut f = g.clone();
    f.run_fused(1).unwrap();
    assert_grids_same(&s, &f, &outputs(), what);
}

// -------------------------------------------------------------------- 11

/// `calcFMC` on `g`, then the six FMC fields.
fn calc_fmc(g: &mut FbpGrid) -> Vec<Vec<f64>> {
    g.run_stage(Stage::CalcFmc, 1).unwrap();
    FMC_FIELDS.iter().map(|f| g.get(*f).unwrap()).collect()
}

/// `n` cells cycling through `fuels`, every input varying from cell to cell
/// (including negative slope, aspect and wind speed, and zero curing, which
/// the normalisation changes), with NaN holes in `lat`, `ws`, `ffmc`, `bui`.
fn data(n: usize, fuels: &[i32]) -> Data {
    let col = |k: usize, scale: f64, off: f64| -> Vec<f64> {
        (0..n)
            .map(|i| off + scale * f64::from(u32::try_from((i * (k + 3)) % 97).unwrap()) / 97.0)
            .collect()
    };
    let mut cols = [
        col(0, 12.0, 48.0),   // lat
        col(1, 30.0, -125.0), // long
        col(2, 1500.0, 0.0),  // elevation (0 at some cells)
        col(3, 65.0, -5.0),   // slope (some negative)
        col(4, 370.0, -10.0), // aspect (some negative)
        col(5, 100.0, 0.0),   // pc
        col(6, 100.0, 0.0),   // pdf
        col(7, 1.0, 0.0),     // gfl
        col(8, 100.0, 0.0),   // gcf (0 at some cells)
        col(9, 47.0, -2.0),   // ws (some negative)
        col(10, 360.0, 0.0),  // wd
        col(11, 26.0, 70.0),  // ffmc
        col(12, 150.0, 0.0),  // bui
    ];
    for (c, first, step) in [(0, 0, 311), (9, 5, 509), (11, 7, 401), (12, 11, 293)] {
        for i in (first..n).step_by(step) {
            cols[c][i] = f64::NAN;
        }
    }
    Data {
        fuel: (0..n).map(|i| fuels[i % fuels.len()]).collect(),
        cols,
        wx_date: 20_230_715,
    }
}

/// Negative override cells ask for a calculation: an array holds NaN in `fmc` and `fme` until
/// `calcFMC` runs and the calculated value afterwards, a constant negative is no override.
fn fmc_calculate_cells(d: &Data) {
    let n = d.fuel.len();
    let all = outputs();
    let mut cells: Vec<f64> = (0..n)
        .map(|i| 90.0 + f64::from(u32::try_from(i % 20).unwrap()))
        .collect();
    for i in (0..n).step_by(3) {
        cells[i] = -1.0;
    }
    cells[1] = -999.0;
    cells[2] = f64::NAN;
    let mut g = d.grid(90.0, &all);
    g.set_override(Field::FmcOverride, Some(PerCell::Cells(&cells)))
        .unwrap();
    let (fmc, fme) = (g.get(Field::Fmc).unwrap(), g.get(Field::Fme).unwrap());
    for i in 0..n {
        if cells[i] < 0.0 || cells[i].is_nan() {
            assert!(
                fmc[i].is_nan() && fme[i].is_nan(),
                "cell {i}: not calculated yet"
            );
        } else {
            assert_eq!(fmc[i], cells[i]);
            assert!(same(
                fme[i],
                1000.0 * (1.5 - 0.00275 * cells[i]).powi(4) / (460.0 + 25.9 * cells[i])
            ));
        }
    }
    // after calcFMC: given cells kept, calculate cells derived, NaN missing; equals the fused run
    let got = calc_fmc(&mut g);
    let plain = calc_fmc(&mut d.grid(90.0, &all));
    for i in 0..n {
        for (k, f) in FMC_FIELDS.iter().take(4).enumerate() {
            assert!(same(got[k][i], plain[k][i]), "{}[{i}]", f.name());
        }
        if cells[i].is_nan() {
            assert!(
                got[4][i].is_nan() && got[5][i].is_nan(),
                "cell {i}: missing"
            );
        } else if cells[i] < 0.0 {
            assert!(
                same(got[4][i], plain[4][i]) && same(got[5][i], plain[5][i]),
                "cell {i}: calculated"
            );
        } else {
            assert_eq!(got[4][i], cells[i], "cell {i}: given");
        }
    }
    assert_staged_equals_fused(&g, "fmc calculate cells");
    // a constant negative is no override at all (templates before, derived after)
    let mut c = d.grid(90.0, &all);
    c.set_override(Field::FmcOverride, Some(PerCell::Const(-1.0)))
        .unwrap();
    assert!(c.get(Field::Fmc).unwrap().iter().all(|v| *v == 0.0));
    assert_fmc_same(&calc_fmc(&mut c), &plain, "constant negative");
    c.set_const(Field::Fmc, -5.0).unwrap();
    assert!(c.get(Field::Fmc).unwrap().iter().all(|v| *v == 0.0));
}

/// (d3) `d0` is used as computed or supplied, never clipped (D15.7).
fn fmc_d0_is_not_range_limited() {
    let all = outputs();
    let one = |lat: f64, long: f64, elev: f64| {
        let mut c = data(1, &[2]);
        c.cols[LAT][0] = lat;
        c.cols[LONG][0] = long;
        c.cols[ELEV][0] = elev;
        c
    };
    let mut g = one(83.0, -52.0, 5_959.0).grid(50.0, &all);
    assert_eq!(calc_fmc(&mut g)[1][0], 370.0);
    assert_staged_equals_fused(&g, "d0 370");
    let mut g = one(0.0, -110.0, 0.0).grid(50.0, &all);
    assert_eq!(calc_fmc(&mut g)[1][0], 0.0);
    for v in [400.0, 0.0] {
        g.set_override(Field::D0Override, Some(PerCell::Const(v)))
            .unwrap();
        assert_eq!(calc_fmc(&mut g)[1][0], v);
        assert_staged_equals_fused(&g, "d0 override");
    }
}

/// (d2) `dj` overrides: floored, then clipped to the year of `wx_date`, and
/// re-resolved when it changes (D15.6).
#[allow(clippy::too_many_lines)]
fn fmc_dj_overrides_are_floored_and_clipped() {
    let all = outputs();
    let supplied = [
        0.3,
        1.0,
        1.999,
        59.5,
        166.9,
        365.0,
        365.7,
        366.0,
        366.9,
        367.0,
        400.0,
        0.0, // a negative value is a request to calculate, see fmc_calculate_cells
        f64::NAN,
    ];
    let mut dj = data(supplied.len(), &[2]);
    let mut g = dj.grid(50.0, &all);
    g.set_override(Field::DjOverride, Some(PerCell::Cells(&supplied)))
        .unwrap();
    for (year, last) in [
        (1999, 365.0),
        (2000, 366.0),
        (2023, 365.0),
        (2024, 366.0),
        (2100, 365.0),
    ] {
        let date = year * 10_000 + 615;
        g.set_param(ParamName::WxDate, f64::from(date)).unwrap();
        let want: Vec<f64> = supplied
            .iter()
            .map(|v: &f64| {
                if v.is_nan() {
                    f64::NAN
                } else {
                    v.floor().clamp(1.0, last)
                }
            })
            .collect();
        let clipped = supplied
            .iter()
            .filter(|v| !v.is_nan() && (v.floor() < 1.0 || v.floor() > last))
            .count();
        assert_same(
            &g.get(Field::DjOverride).unwrap(),
            &want,
            &format!("{year}: resolved"),
        );
        assert_eq!(g.dj_override_clipped(), clipped, "{year}: clipped");
        let got = calc_fmc(&mut g);
        assert_same(&got[2], &want, &format!("{year}: dj"));
        dj.wx_date = i64::from(date);
        let mut fresh = FbpGrid::new(
            &GridInput {
                dj: Some(PerCell::Cells(&supplied)),
                ..dj.input(50.0)
            },
            &all,
        )
        .unwrap();
        assert_eq!(
            fresh.dj_override_clipped(),
            clipped,
            "{year}: fresh clipped"
        );
        assert_fmc_same(&got, &calc_fmc(&mut fresh), &format!("{year}: fresh"));
        assert_staged_equals_fused(&g, &format!("dj {year}"));
    }
    // in-range, NaN, fraction-only and leap-year-valid values are never counted
    g.set_param(ParamName::WxDate, 20_240_615.0).unwrap();
    let valid = [
        1.0,
        166.9,
        365.5,
        f64::NAN,
        366.0,
        366.9,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
    ];
    g.set_override(Field::DjOverride, Some(PerCell::Cells(&valid)))
        .unwrap();
    assert_eq!(g.dj_override_clipped(), 0);
    // the same 366 is clipped once wx_date moves to a non-leap year
    g.set_param(ParamName::WxDate, 20_230_615.0).unwrap();
    assert_eq!(g.dj_override_clipped(), 2);
    for bad in [f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            g.set_override(Field::DjOverride, Some(PerCell::Const(bad))),
            Err(GridError::InfiniteDj)
        );
    }
    let mut inf = supplied.to_vec();
    inf[2] = f64::INFINITY;
    assert_eq!(
        g.set_override(Field::DjOverride, Some(PerCell::Cells(&inf))),
        Err(GridError::InfiniteDj)
    );
}

/// (a) each changed input equals a fresh grid built with it; (g) staged
/// equals fused.
fn fmc_follows_its_inputs(d: &Data) {
    let n = d.fuel.len();
    let all = outputs();
    let new_lat = new_lat(d);
    let new_long: Vec<f64> = (0..n)
        .map(|i| -100.0 - f64::from(u32::try_from(i % 9).unwrap()))
        .collect();
    let new_elev: Vec<f64> = d.cols[ELEV].iter().map(|v| v * 0.5 + 10.0).collect();
    let cases: [(Field, &Vec<f64>); 3] = [
        (Field::Lat, &new_lat),
        (Field::Long, &new_long),
        (Field::Elevation, &new_elev),
    ];
    for (f, values) in cases {
        let mut g = d.grid(90.0, &all);
        calc_fmc(&mut g);
        g.set(f, values).unwrap();
        let got = calc_fmc(&mut g);
        let mut changed = d.clone();
        let k = match f {
            Field::Lat => LAT,
            Field::Long => LONG,
            _ => ELEV,
        };
        changed.cols[k].clone_from(values);
        let mut fresh = changed.grid(90.0, &all);
        assert_fmc_same(&got, &calc_fmc(&mut fresh), &format!("set({})", f.name()));
        assert_staged_equals_fused(&g, &format!("set({})", f.name()));
    }
    // dj follows long: NaN where long was NaN, the day of year once it is valid
    let mut g = d.grid(90.0, &all);
    let before = calc_fmc(&mut g);
    assert!(before[2][3].is_nan());
    g.set(Field::Long, &new_long).unwrap();
    assert_eq!(calc_fmc(&mut g)[2][3], 196.0); // 2023-07-15

    // wx_date
    let mut g = d.grid(90.0, &all);
    calc_fmc(&mut g);
    g.set_param(ParamName::WxDate, 20_240_301.0).unwrap();
    let got = calc_fmc(&mut g);
    let mut leap = d.clone();
    leap.wx_date = 20_240_301;
    assert_fmc_same(&got, &calc_fmc(&mut leap.grid(90.0, &all)), "wx_date");
    assert_eq!(got[2][0], 61.0);
    assert_staged_equals_fused(&g, "wx_date");
}

/// (d) constant, per-cell and masked overrides take priority, persist,
/// are replaced and cleared.
fn fmc_overrides(d: &Data) {
    let n = d.fuel.len();
    let all = outputs();
    let per_cell: Vec<f64> = (0..n)
        .map(|i| 120.0 + f64::from(u32::try_from(i % 50).unwrap()))
        .collect();
    let mut masked = per_cell.clone();
    masked[4] = f64::NAN;
    let overrides: [(&str, PerCell<'_>); 3] = [
        ("constant", PerCell::Const(140.0)),
        ("per cell", PerCell::Cells(&per_cell)),
        ("masked", PerCell::Cells(&masked)),
    ];
    for (what, d0) in overrides {
        for field in [Field::D0Override, Field::DjOverride, Field::FmcOverride] {
            let mut g = d.grid(90.0, &all);
            calc_fmc(&mut g);
            g.set_override(field, Some(d0)).unwrap();
            let got = calc_fmc(&mut g);
            for i in 0..n {
                let mut input = d.cell(i, 90.0);
                let value = match d0 {
                    PerCell::Const(value) => value,
                    PerCell::Cells(cells) => cells[i],
                };
                match field {
                    Field::D0Override => input.d0_override = Some(value),
                    Field::DjOverride => input.dj_override = Some(value), // in range: floor is a no-op
                    _ => input.fmc_override = Some(value),
                }
                let want = run(&input);
                for (f, x) in FMC_FIELDS.iter().zip(&got) {
                    let expected = want.get(f.name()).unwrap();
                    assert!(
                        same(x[i], expected),
                        "{what} {}: {} cell {i}: {} vs {expected}",
                        field.name(),
                        f.name(),
                        x[i]
                    );
                }
            }
            if what == "masked" {
                assert!(
                    got[4][4].is_nan() && got[5][4].is_nan(),
                    "masked cell: fmc/fme NaN"
                );
            }
            // persists across reset and run_fused
            g.reset();
            let after_reset = calc_fmc(&mut g);
            assert_fmc_same(&got, &after_reset, &format!("{what} after reset"));
            assert_staged_equals_fused(&g, &format!("{what} {}", field.name()));
            g.run_fused(1).unwrap();
            assert_fmc_same(
                &got,
                &FMC_FIELDS
                    .iter()
                    .map(|f| g.get(*f).unwrap())
                    .collect::<Vec<_>>(),
                &format!("{what} after run_fused"),
            );
            // replaced by a second override; None returns to derived
            g.reset();
            g.set_override(field, Some(PerCell::Const(150.0))).unwrap();
            let replaced = calc_fmc(&mut g);
            let col = match field {
                Field::D0Override => 1,
                Field::DjOverride => 2,
                _ => 4,
            };
            assert!(
                replaced[col].iter().all(|v| *v == 150.0),
                "{what}: replaced"
            );
            g.set_override(field, None).unwrap();
            assert_fmc_same(
                &calc_fmc(&mut g),
                &calc_fmc(&mut d.grid(90.0, &all)),
                &format!("{what}: cleared"),
            );
        }
    }
    assert_eq!(
        d.grid(90.0, &all)
            .set_override(Field::D0Override, Some(PerCell::Cells(&[1.0]))),
        Err(GridError::LengthMismatch {
            name: "d0",
            expected: n,
            found: 1
        })
    );
    assert_eq!(
        d.grid(90.0, &all).set_override(Field::Fmc, None),
        Err(GridError::NotAnOverride(Field::Fmc))
    );
}

/// (b) an identical repeat does not recompute; (c) a given `fmc` is an
/// override.
fn fmc_repeats_are_cached(d: &Data) {
    let n = d.fuel.len();
    let all = outputs();
    let new_lat = new_lat(d);
    let mut g = d.grid(90.0, &all);
    let first = calc_fmc(&mut g);
    assert_eq!(g.fmc_computations(), 1);
    let second = calc_fmc(&mut g);
    assert_eq!(
        g.fmc_computations(),
        1,
        "an identical repeat is a cache hit"
    );
    assert_fmc_same(&first, &second, "repeat");
    g.set(Field::Lat, &new_lat).unwrap();
    calc_fmc(&mut g);
    assert_eq!(g.fmc_computations(), 2, "a changed input recomputes");

    // (c) a given fmc is an override: calcFMC recomputes with it
    g.set(Field::Fmc, &vec![55.0; n]).unwrap();
    let again = calc_fmc(&mut g);
    assert_eq!(g.fmc_computations(), 3);
    let mut fresh = d.grid(90.0, &all);
    fresh.set(Field::Lat, &new_lat).unwrap();
    fresh
        .set_override(Field::FmcOverride, Some(PerCell::Const(55.0)))
        .unwrap();
    assert_fmc_same(&again, &calc_fmc(&mut fresh), "fmc override");
    assert!(again[4].iter().all(|v| *v == 55.0));
    assert_eq!(g.fmc_computations(), 3);
    calc_fmc(&mut g);
    assert_eq!(g.fmc_computations(), 3, "the override is a cache hit too");
}

/// (e) an invalid `wx_date` is refused; (f) `percentile_growth` changes the
/// next growth stage.
fn fmc_scalar_parameters(d: &Data) {
    let all = outputs();
    let mut g = d.grid(50.0, &all);
    assert_eq!(
        g.set_param(ParamName::WxDate, 20_230_230.0),
        Err(GridError::InvalidDate(20_230_230))
    );
    assert!(matches!(
        g.set_param(ParamName::WxDate, 20_230_615.5),
        Err(GridError::InvalidDate(_))
    ));
    assert!(matches!(
        g.set_param(ParamName::WxDate, f64::NAN),
        Err(GridError::InvalidDate(_))
    ));

    // (f) percentile_growth changes the next growth stage
    let mut g = d.grid(50.0, &all);
    g.set_param(ParamName::PercentileGrowth, 95.0).unwrap();
    g.run_planned(1).unwrap();
    assert_grids_same(&g, &planned(d, 95.0, &all, 1), &all, "percentile 95");
    assert_staged_equals_fused(&g, "percentile 95");
}

/// The stages that are freed (`get` gives `FreedRead`).
fn freed(g: &FbpGrid) -> Vec<Field> {
    Field::ALL
        .into_iter()
        .filter(|f| g.get(*f) == Err(StageError::FreedRead(*f)))
        .collect()
}

fn fused(d: &Data, percentile: f64, out: &[Field], threads: usize) -> FbpGrid {
    let mut g = d.grid(percentile, out);
    g.run_fused(threads).unwrap();
    g
}

/// `set_fuel` persistence; the refused `set` routes.
fn lifecycle_fuel_and_set_routes(d: &Data) {
    let n = d.fuel.len();
    let all = outputs();
    // set_fuel persists across reset and run_fused
    let codes: Vec<i32> = (0..n).map(|i| MODELED[i % MODELED.len()]).collect();
    let mut g = d.grid(90.0, &all);
    g.set_fuel(&codes).unwrap();
    g.reset();
    g.run_fused(1).unwrap();
    let mut with_codes = d.clone();
    with_codes.fuel = codes;
    assert_grids_same(&g, &fused(&with_codes, 90.0, &all, 1), &all, "set_fuel");
    assert_eq!(
        g.set_fuel(&[2]),
        Err(GridError::LengthMismatch {
            name: "fuel_type",
            expected: n,
            found: 1
        })
    );

    // set routes
    assert_eq!(
        g.set(Field::Hros, &[1.0]),
        Err(StageError::LengthMismatch {
            field: Field::Hros,
            expected: n,
            found: 1
        })
    );
    for f in [
        Field::FuelType,
        Field::D0Override,
        Field::DjOverride,
        Field::FmcOverride,
    ] {
        assert_eq!(g.set_const(f, 1.0), Err(StageError::NotSettable(f)));
    }
}

/// Injected values and `run_fused`; the liveness trace after `reset`.
fn lifecycle_injections(d: &Data) {
    let n = d.fuel.len();
    let all = outputs();
    let reference = fused(d, 90.0, &all, 1);
    // an injected computed value followed by run_fused is discarded
    let mut g = d.grid(90.0, &all);
    g.run_stage(Stage::CalcFmc, 1).unwrap();
    g.set(Field::Hros, &vec![77.0; n]).unwrap();
    g.run_fused(1).unwrap();
    assert_grids_same(&g, &reference, &all, "set(hros) then run_fused");

    // a given fmc is an override: it persists through run_fused, and setting
    // fmc is the same as setting the override
    let mut with_override = d.grid(90.0, &all);
    with_override
        .set_override(Field::FmcOverride, Some(PerCell::Const(77.0)))
        .unwrap();
    with_override.run_fused(1).unwrap();
    let mut set_fmc = d.grid(90.0, &all);
    set_fmc.run_stage(Stage::CalcFmc, 1).unwrap();
    set_fmc.set(Field::Fmc, &vec![77.0; n]).unwrap();
    set_fmc.run_fused(1).unwrap();
    assert_grids_same(&set_fmc, &with_override, &all, "set(fmc) then run_fused");
    assert!(with_override
        .get(Field::Fmc)
        .unwrap()
        .iter()
        .all(|v| *v == 77.0));
    assert!(with_override
        .get(Field::Latn)
        .unwrap()
        .iter()
        .any(|v| *v != 0.0));

    // after reset a second run_planned has the same live_bytes trace and frees the same fields
    let trace = |g: &mut FbpGrid| -> (Vec<usize>, Vec<Field>) {
        let mut bytes = vec![g.live_bytes()];
        for s in Stage::ALL {
            g.run_stage(s, 1).unwrap();
            bytes.push(g.live_bytes());
        }
        (bytes, freed(g))
    };
    let mut g = d.grid(90.0, &[Field::Hros, Field::Hfi]);
    let first = trace(&mut g);
    g.reset();
    let second = trace(&mut g);
    assert_eq!(first, second);
    assert!(first.1.contains(&Field::Sfc));
    assert!(
        first.0.windows(2).any(|w| w[1] < w[0]),
        "something is freed"
    );

    // a Fresh grid: injections followed by run_fused are released
    let mut injected = d.grid(90.0, &all);
    injected.set(Field::Hros, &vec![3.0; n]).unwrap();
    injected.run_fused(1).unwrap();
    assert_grids_same(
        &injected,
        &reference,
        &all,
        "fresh injection then run_fused",
    );
    assert_eq!(injected.live_bytes(), reference.live_bytes());
}

/// Repeated and mixed runs, `AfterFused`, `reset` keeping inputs.
fn lifecycle_runs(d: &Data) {
    let n = d.fuel.len();
    let all = outputs();
    let reference = fused(d, 90.0, &all, 1);

    // repeated run_fused; run_planned then run_fused
    let mut g = d.grid(90.0, &all);
    g.run_fused(1).unwrap();
    g.run_fused(1).unwrap();
    assert_grids_same(&g, &reference, &all, "run_fused twice");
    let mut g = planned(d, 90.0, &all, 1);
    g.run_fused(1).unwrap();
    assert_grids_same(&g, &reference, &all, "run_planned then run_fused");

    // run_fused then run_stage
    assert_eq!(g.run_stage(Stage::CalcSf, 1), Err(StageError::AfterFused));
    assert_eq!(g.run_planned(1), Err(StageError::AfterFused));
    g.reset();
    g.run_planned(1).unwrap();
    assert_grids_same(
        &g,
        &planned(d, 90.0, &all, 1),
        &all,
        "run_planned after reset",
    );

    // reset keeps a set() on an input and discards a set() on a computed field
    let slope = vec![33.0; n];
    let mut g = d.grid(90.0, &all);
    g.set(Field::Slope, &slope).unwrap();
    g.set(Field::Sfc, &vec![1.5; n]).unwrap();
    g.reset();
    assert!(g.get(Field::Sfc).unwrap().iter().all(|v| v.is_nan()));
    assert_same(&g.get(Field::Slope).unwrap(), &slope, "slope kept");
    g.run_planned(1).unwrap();
    let mut changed = d.clone();
    changed.cols[SLOPE] = slope;
    assert_grids_same(&g, &planned(&changed, 90.0, &all, 1), &all, "set(slope)");
}

/// `invertWindAspect` and `set(wd)`.
fn lifecycle_wind_inversion(d: &Data) {
    let n = d.fuel.len();
    let all = outputs();
    // invertWindAspect: twice equals once; get(wd) before and after
    let mut g = d.grid(90.0, &all);
    assert_same(
        &g.get(Field::Wd).unwrap(),
        &d.cols[WD],
        "wd before inversion",
    );
    g.run_stage(Stage::InvertWindAspect, 1).unwrap();
    let once = g.get(Field::Wd).unwrap();
    g.run_stage(Stage::InvertWindAspect, 1).unwrap();
    assert_same(&g.get(Field::Wd).unwrap(), &once, "twice equals once");
    for (i, w) in once.iter().enumerate() {
        let raw = d.cols[WD][i];
        let want = if raw > 180.0 {
            raw - 180.0
        } else {
            raw + 180.0
        };
        assert!(same(*w, want), "wd[{i}]");
    }
    // set(wd, v) then the stage equals a grid built with v
    let wd: Vec<f64> = (0..n)
        .map(|i| f64::from(u32::try_from(i * 7 % 360).unwrap()))
        .collect();
    g.set(Field::Wd, &wd).unwrap();
    assert_same(
        &g.get(Field::Wd).unwrap(),
        &wd,
        "set(wd) reads as the input",
    );
    g.run_stage(Stage::InvertWindAspect, 1).unwrap();
    let mut with_wd = d.clone();
    with_wd.cols[WD] = wd;
    let mut h = with_wd.grid(90.0, &all);
    h.run_stage(Stage::InvertWindAspect, 1).unwrap();
    assert_grids_same(&g, &h, &[Field::Wd, Field::Aspect], "set(wd)");
    g.reset();
    g.run_planned(1).unwrap();
    assert_grids_same(
        &g,
        &planned(&with_wd, 90.0, &all, 1),
        &all,
        "set(wd) persists",
    );
}

fn new_lat(d: &Data) -> Vec<f64> {
    d.cols[LAT].iter().map(|v| v + 3.0).collect()
}

/// The 54 outputs.
fn outputs() -> Vec<Field> {
    Field::ALL.into_iter().filter(|f| f.is_output()).collect()
}

fn planned(d: &Data, percentile: f64, out: &[Field], threads: usize) -> FbpGrid {
    let mut g = d.grid(percentile, out);
    g.run_planned(threads).unwrap();
    g
}

// --------------------------------------------------------------------- 8

/// Canonical order with `calcSFC` and `getCBH_CFL` moved before `calcROS`.
fn pyrocell_order() -> Vec<Stage> {
    let mut order: Vec<Stage> = Stage::ALL
        .into_iter()
        .filter(|s| !matches!(s, Stage::CalcSfc | Stage::GetCbhCfl))
        .collect();
    let ros = order.iter().position(|s| *s == Stage::CalcRos).unwrap();
    order.insert(ros, Stage::GetCbhCfl);
    order.insert(ros, Stage::CalcSfc);
    order
}

fn run_all(g: &mut FbpGrid, stages: &[Stage]) {
    for &s in stages {
        g.run_stage(s, 1)
            .unwrap_or_else(|e| panic!("{}: {e}", s.name()));
    }
}

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

// ---- other tests ----

#[test]
fn stepwise_equals_fused() {
    let d = data(1_013, &MIXED);
    for threads in [1, 4] {
        let a = planned(&d, 90.0, &outputs(), threads);
        let b = fused(&d, 90.0, &outputs(), threads);
        assert_grids_same(&a, &b, &outputs(), &format!("threads {threads}"));
    }
}

#[test]
fn freeing_does_not_change_results() {
    let d = data(1_013, &MIXED);
    let n = d.fuel.len();
    let small = [Field::Hros, Field::Hfi, Field::FireType];
    let all = planned(&d, 90.0, &outputs(), 1);
    let part = planned(&d, 90.0, &small, 1);
    assert_grids_same(&all, &part, &small, "out_request of three");
    // 13 input arrays and the fuel codes, plus the three outputs
    assert!(
        part.live_bytes() <= (14 + 3) * n * 8,
        "{}",
        part.live_bytes()
    );
    assert!(
        part.live_bytes() < all.live_bytes() / 2,
        "{} vs {}",
        part.live_bytes(),
        all.live_bytes()
    );
}

#[test]
fn freed_field_read_is_an_error_and_never_computed_is_template() {
    let d = data(50, &MIXED);
    let g = planned(&d, 50.0, &[Field::Hros], 1);
    assert_eq!(g.get(Field::Sfc), Err(StageError::FreedRead(Field::Sfc)));
    assert!(g.get(Field::Hros).is_ok());
    let fresh = d.grid(50.0, &[Field::Hros]);
    assert!(fresh.get(Field::Sfc).unwrap().iter().all(|v| v.is_nan()));
    assert!(fresh.get(Field::Rso).unwrap().iter().all(|v| *v == 0.0));
    assert!(fresh.get(Field::Hros).unwrap().iter().all(|v| *v == 0.0));
}

#[test]
fn injection_matches_the_pyrocell_sequence() {
    let d = data(200, &MODELED);
    let n = d.fuel.len();
    let fmc: Vec<f64> = (0..n)
        .map(|i| 80.0 + f64::from(u32::try_from(i % 40).unwrap()))
        .collect();
    let hros: Vec<f64> = (0..n)
        .map(|i| 3.0 + 0.7 * f64::from(u32::try_from(i % 17).unwrap()))
        .collect();
    for percentile in [50.0, 90.0] {
        let mut g = d.grid(percentile, &outputs());
        run_all(
            &mut g,
            &[Stage::InvertWindAspect, Stage::CalcSf, Stage::CalcIsz],
        );
        g.set(Field::Fmc, &fmc).unwrap();
        run_all(
            &mut g,
            &[
                Stage::CalcIsiRsiBe,
                Stage::CalcSfc,
                Stage::GetCbhCfl,
                Stage::CalcRos,
            ],
        );
        g.set(Field::Hros, &hros).unwrap();
        run_all(
            &mut g,
            &[
                Stage::CalcCsfi,
                Stage::CalcRso,
                Stage::CalcC6BlendCfb,
                Stage::CalcC6BlendCfc,
                Stage::CalcC6Cros,
                Stage::CalcC6Hros,
                Stage::CalcPercentileCfb,
                Stage::CalcRosPercentileGrowth,
                Stage::CalcCfb,
                Stage::CalcFireType,
                Stage::CalcCfc,
                Stage::CalcTfc,
                Stage::CalcHfi,
                Stage::CalcFireIntensityClass,
            ],
        );
        let got: Vec<(Field, Vec<f64>)> = outputs()
            .into_iter()
            .map(|f| (f, g.get(f).unwrap()))
            .collect();
        let mut c6_blended = 0;
        for i in 0..n {
            let mut input = d.cell(i, percentile);
            input.fmc_override = Some(fmc[i]);
            input.hros_override = Some(hros[i]);
            let want = run(&input);
            for (f, v) in &got {
                if *f == Field::Accel {
                    // calcAccelParam is not in the sequence: the template stays
                    assert_eq!(v[i], 0.0);
                    continue;
                }
                if matches!(*f, Field::Latn | Field::D0 | Field::Dj | Field::Nd) {
                    // calcFMC is not in the sequence (the fmc override gives fmc
                    // and fme at once): these four stay at their templates, the
                    // run computes them
                    assert_eq!(v[i], 0.0);
                    continue;
                }
                let w = want.get(f.name()).unwrap();
                assert!(
                    same(v[i], w),
                    "p{percentile} cell {i} (fuel {}) {}: {} vs run {w}",
                    d.fuel[i],
                    f.name(),
                    v[i]
                );
            }
            if d.fuel[i] == 6 {
                // the given fmc gives the matching foliar moisture effect
                let fme = got.iter().find(|(f, _)| *f == Field::Fme).unwrap().1[i];
                let expected = 1000.0 * (1.5 - 0.00275 * fmc[i]).powi(4) / (460.0 + 25.9 * fmc[i]);
                assert!(same(fme, expected), "fme {fme} vs {expected}");
                let h = got.iter().find(|(f, _)| *f == Field::Hros).unwrap().1[i];
                if percentile == 50.0 && !same(h, hros[i]) {
                    c6_blended += 1;
                }
            }
        }
        if percentile == 50.0 {
            assert!(
                c6_blended > 0,
                "the C-6 blend overwrites the injected hros (sros blended with crown ROS)"
            );
        }
    }
}

#[test]
fn wrong_order_is_safe() {
    // modeled fuels: zeros in, zeros out (non-modeled cells get Python's
    // values instead)
    let m = data(40, &MODELED);
    let mut g = m.grid(50.0, &outputs());
    g.run_stage(Stage::CalcHfi, 1).unwrap();
    assert!(g.get(Field::Hfi).unwrap().iter().all(|v| *v == 0.0));
    g.run_stage(Stage::CalcFireIntensityClass, 1).unwrap();
    assert!(g.get(Field::FiClass).unwrap().iter().all(|v| *v == -99.0));
    // any stage, in any order, on any fuel, never panics
    let d = data(40, &MIXED);
    let mut g = d.grid(50.0, &outputs());
    for s in Stage::ALL.iter().rev() {
        g.run_stage(*s, 1).unwrap();
    }

    let mut g = planned(&d, 50.0, &[Field::Hros], 1);
    assert_eq!(
        g.run_stage(Stage::CalcRso, 1),
        Err(StageError::Freed {
            field: Field::Sfc,
            stage: Stage::CalcRso
        })
    );
    // a stage whose inputs are all inputs still runs
    g.run_stage(Stage::CalcSf, 1).unwrap();
}

#[test]
fn thread_counts_are_bit_identical() {
    let out = [
        Field::Hros,
        Field::Bros,
        Field::Raz,
        Field::Hfi,
        Field::Cfb,
        Field::Fmc,
        Field::Dj,
        Field::FireType,
        Field::Sfc,
        Field::Wd,
    ];
    for n in [0, 1, 4_095, 4_096, 4_097, 20_011, 100_003] {
        let d = data(n, &MIXED);
        let staged = planned(&d, 90.0, &out, 1);
        let fused_ref = fused(&d, 90.0, &out, 1);
        if n >= 4_096 {
            assert_grids_same(
                &staged,
                &fused_ref,
                &out,
                &format!("n {n}: staged vs fused"),
            );
        }
        for threads in [0, 2, 7, 64] {
            let what = format!("n {n} threads {threads}");
            assert_grids_same(&staged, &planned(&d, 90.0, &out, threads), &out, &what);
            assert_grids_same(&fused_ref, &fused(&d, 90.0, &out, threads), &out, &what);
        }
    }
}

#[test]
fn normalisation_is_idempotent() {
    let raw = data(500, &MIXED);
    let mut norm = raw.clone();
    let clamp0 = |v: &mut f64| {
        if *v < 0.0 {
            *v = 0.0;
        }
    };
    for k in [SLOPE, 5, 6, 7, WS, 11, 12] {
        norm.cols[k].iter_mut().for_each(clamp0);
    }
    for v in &mut norm.cols[ASPECT] {
        if *v < 0.0 {
            *v = 270.0;
        }
    }
    for v in &mut norm.cols[8] {
        if *v == 0.0 {
            *v = 0.1;
        }
    }
    for v in &mut norm.cols[LONG] {
        *v = v.abs();
    }
    assert!(raw.cols[SLOPE].iter().any(|v| *v < 0.0));
    let mut fields = outputs();
    fields.extend([
        Field::Slope,
        Field::AspectIn,
        Field::Long,
        Field::Gcf,
        Field::Pc,
    ]);
    let a = planned(&raw, 90.0, &outputs(), 1);
    let b = planned(&norm, 90.0, &outputs(), 1);
    assert_grids_same(&a, &b, &fields, "staged");
    assert_grids_same(
        &fused(&raw, 90.0, &outputs(), 1),
        &fused(&norm, 90.0, &outputs(), 1),
        &outputs(),
        "fused",
    );
    // set() normalises too
    let mut c = raw.grid(90.0, &outputs());
    c.set(Field::Slope, &raw.cols[SLOPE]).unwrap();
    assert_same(
        &c.get(Field::Slope).unwrap(),
        &norm.cols[SLOPE],
        "set(slope)",
    );
}

#[test]
fn liveness_is_order_independent() {
    let d = data(300, &MIXED);
    let out = [Field::Hros, Field::Hfi];
    let canonical = planned(&d, 90.0, &out, 1);

    let mut pyro = d.grid(90.0, &out);
    run_all(&mut pyro, &pyrocell_order());
    assert_grids_same(&canonical, &pyro, &out, "PyroCell order");

    // skipping calcFMC (injected fmc) with an exact plan, in the PyroCell order
    let fmc: Vec<f64> = vec![95.0; d.fuel.len()];
    let without_fmc: Vec<Stage> = pyrocell_order()
        .into_iter()
        .filter(|s| *s != Stage::CalcFmc)
        .collect();
    let mut exact = d.grid(90.0, &out);
    exact.plan(&without_fmc, &[]);
    exact.set(Field::Fmc, &fmc).unwrap();
    run_all(&mut exact, &without_fmc);
    // the default plan keeps calcFMC's inputs, and gives the same results
    let mut default_plan = d.grid(90.0, &out);
    default_plan.set(Field::Fmc, &fmc).unwrap();
    run_all(&mut default_plan, &without_fmc);
    assert_grids_same(&exact, &default_plan, &out, "skipped calcFMC");
    assert!(exact.live_bytes() <= default_plan.live_bytes());

    // a skipped planned stage keeps its inputs stored
    let mut skip_accel = d.grid(90.0, &out);
    let without_accel: Vec<Stage> = Stage::ALL
        .into_iter()
        .filter(|s| *s != Stage::CalcAccelParam)
        .collect();
    run_all(&mut skip_accel, &without_accel);
    assert!(
        skip_accel.get(Field::Cfb).is_ok(),
        "calcAccelParam still needs cfb"
    );
    assert_eq!(
        canonical.get(Field::Cfb),
        Err(StageError::FreedRead(Field::Cfb))
    );
    skip_accel.run_stage(Stage::CalcAccelParam, 1).unwrap();
    assert_eq!(
        skip_accel.get(Field::Cfb),
        Err(StageError::FreedRead(Field::Cfb))
    );
}

#[test]
fn injected_nan_hros_is_not_zeroed() {
    let d = data(60, &[2, 3, 4]); // crowning fuels
    let n = d.fuel.len();
    let after_ros = Stage::ALL
        .iter()
        .position(|s| *s == Stage::CalcRos)
        .unwrap();
    let (head, tail) = Stage::ALL.split_at(after_ros + 1);
    let nan = vec![f64::NAN; n];
    let finite_rso = |g: &FbpGrid| g.get(Field::Rso).unwrap();
    let keep = [Field::Cfb, Field::Rso, Field::Hros, Field::HrosGrowthNan];

    // (a) the PyroCell tail, growth included
    let mut with_growth = d.grid(90.0, &keep);
    run_all(&mut with_growth, head);
    with_growth.set(Field::Hros, &nan).unwrap();
    run_all(&mut with_growth, tail);
    // (b) calcCFB without the growth stage (the pre-0.2 PyroCell order)
    let mut without_growth = d.grid(90.0, &keep);
    run_all(&mut without_growth, head);
    without_growth.set(Field::Hros, &nan).unwrap();
    let no_growth: Vec<Stage> = tail
        .iter()
        .copied()
        .filter(|s| *s != Stage::CalcRosPercentileGrowth)
        .collect();
    run_all(&mut without_growth, &no_growth);
    for (g, what) in [
        (&with_growth, "with growth"),
        (&without_growth, "without growth"),
    ] {
        let rso = finite_rso(g);
        let cfb = g.get(Field::Cfb).unwrap();
        for i in 0..n {
            if !rso[i].is_nan() {
                assert!(cfb[i].is_nan(), "{what} cell {i}: cfb {}", cfb[i]);
            }
        }
        assert!(rso.iter().any(|v| !v.is_nan()));
    }

    // a NaN percentile with a finite hros: the sanitiser zeroes cfb
    let mut nan_p = d.grid(f64::NAN, &keep);
    nan_p.run_planned(1).unwrap();
    let (cfb, hros, flag) = (
        nan_p.get(Field::Cfb).unwrap(),
        nan_p.get(Field::Hros).unwrap(),
        nan_p.get(Field::HrosGrowthNan).unwrap(),
    );
    let rso = finite_rso(&nan_p);
    let hros50 = planned(&d, 50.0, &keep, 1).get(Field::Hros).unwrap();
    let mut zeroed = 0;
    for i in 0..n {
        if !rso[i].is_nan() && !hros50[i].is_nan() {
            zeroed += 1;
            assert!(hros[i].is_nan(), "cell {i}");
            assert_eq!(cfb[i], 0.0, "cell {i}");
            assert_eq!(flag[i], 1.0, "cell {i}");
        }
    }
    assert!(zeroed > n / 2, "{zeroed} of {n}");
    // set(hros) after the growth stage resets the provenance
    nan_p.set(Field::Hros, &vec![5.0; n]).unwrap();
    assert!(nan_p
        .get(Field::HrosGrowthNan)
        .unwrap()
        .iter()
        .all(|v| *v == 0.0));
}

#[test]
fn lifecycle() {
    let d = data(400, &MIXED);
    lifecycle_runs(&d);
    lifecycle_wind_inversion(&d);
    lifecycle_injections(&d);
    lifecycle_fuel_and_set_routes(&d);
}

#[test]
fn fmc_contract() {
    let mut d = data(120, &MODELED);
    let n = d.fuel.len();
    for i in (3..n).step_by(17) {
        d.cols[LONG][i] = f64::NAN;
    }
    fmc_follows_its_inputs(&d);
    fmc_repeats_are_cached(&d);
    fmc_overrides(&d);
    fmc_calculate_cells(&d);
    fmc_d0_is_not_range_limited();
    fmc_dj_overrides_are_floored_and_clipped();
    fmc_scalar_parameters(&d);
}

#[test]
fn non_modeled_stages_consume_injected_values() {
    let d = data(3, &[19, 20, 0]);
    let mut g = d.grid(50.0, &[Field::Tfc, Field::Hfi, Field::FiClass]);
    g.set_const(Field::Sfc, 2.0).unwrap();
    g.set_const(Field::Cfc, 3.0).unwrap();
    g.set_const(Field::Hros, 4.0).unwrap();
    g.run_stage(Stage::CalcTfc, 1).unwrap();
    assert_eq!(g.get(Field::Tfc).unwrap(), vec![5.0; 3]);
    g.run_stage(Stage::CalcHfi, 1).unwrap();
    assert_eq!(g.get(Field::Hfi).unwrap(), vec![6000.0; 3]);
    g.run_stage(Stage::CalcFireIntensityClass, 1).unwrap();
    assert_eq!(g.get(Field::FiClass).unwrap(), vec![5.0; 3]);
}

#[test]
fn alternate_acceleration_fuel_uses_current_cfb_and_retention() {
    let d = data(2, &[0, 2]);
    let modeled = data(2, &[2, 2]);
    let mut g = d.grid(50.0, &[Field::Accel]);
    let mut reference = modeled.grid(50.0, &[Field::Accel]);
    g.set_const(Field::Cfb, 0.2).unwrap();
    reference.set_const(Field::Cfb, 0.2).unwrap();
    reference.run_stage(Stage::CalcAccelParam, 1).unwrap();
    g.run_acceleration_for_fuel(&[2, 2], 1).unwrap();
    assert_eq!(
        g.get(Field::Accel).unwrap(),
        reference.get(Field::Accel).unwrap()
    );
    assert_eq!(g.get(Field::FuelType).unwrap(), vec![0.0, 2.0]);
    let mut unretained = d.grid(50.0, &[Field::Hros, Field::Hfi, Field::FireType]);
    unretained.run_acceleration_for_fuel(&[2, 2], 1).unwrap();
    assert_eq!(
        unretained.get(Field::Accel),
        Err(StageError::FreedRead(Field::Accel))
    );
    assert!(matches!(
        g.run_acceleration_for_fuel(&[2], 1),
        Err(StageError::LengthMismatch { .. })
    ));
    g.run_fused(1).unwrap();
    assert_eq!(
        g.run_acceleration_for_fuel(&[2, 2], 1),
        Err(StageError::AfterFused)
    );
}
