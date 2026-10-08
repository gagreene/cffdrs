//! Threaded grid pass: every thread count gives bit-identical results to the
//! serial pass, and `run_grid` itself never starts threads.

use cffdrs_core::grid::{run_grid, run_grid_with_threads, BehaviourGrids, GridInput};

const BASE_FUEL: [i32; 9] = [2, 13, 10, 11, 14, 15, 12, 19, 7];

/// A grid of `n` cells that cycles through modeled fuels, a non-fuel code and
/// varying terrain/wind, with a few NaN inputs so NaN propagation is covered too.
struct Cells {
    fuel: Vec<i32>,
    cols: [Vec<f64>; 8],
}

fn cells(n: usize) -> Cells {
    let fuel: Vec<i32> = (0..n).map(|i| BASE_FUEL[i % BASE_FUEL.len()]).collect();
    let f = |k: usize, scale: f64, off: f64| -> Vec<f64> {
        (0..n)
            .map(|i| off + scale * f64::from(u32::try_from((i * (k + 3)) % 97).unwrap()) / 97.0)
            .collect()
    };
    let mut cols = [
        f(0, 12.0, 48.0),    // lat
        f(1, 30.0, -125.0),  // long
        f(2, 1500.0, 100.0), // elevation
        f(3, 60.0, 0.0),     // slope
        f(4, 360.0, 0.0),    // aspect
        f(5, 100.0, 0.0),    // pct_conifer
        f(6, 100.0, 0.0),    // grass_curing
        f(7, 45.0, 0.0),     // ws
    ];
    for i in (0..n).step_by(311) {
        cols[0][i] = f64::NAN;
    }
    for i in (5..n).step_by(509) {
        cols[7][i] = f64::NAN;
    }
    Cells { fuel, cols }
}

fn grids_for(c: &Cells, threads: Option<usize>) -> BehaviourGrids {
    let n = c.fuel.len();
    let wd: Vec<f64> = (0..n)
        .map(|i| f64::from(u32::try_from(i % 360).unwrap()))
        .collect();
    let input = GridInput {
        fuel_type: &c.fuel,
        lat: &c.cols[0],
        long: &c.cols[1],
        elevation: &c.cols[2],
        slope_pct: &c.cols[3],
        aspect_deg: &c.cols[4],
        pct_conifer: &c.cols[5],
        grass_curing: &c.cols[6],
        ws: &c.cols[7],
        wd: &wd,
        wx_date: 20_230_615,
        ffmc: 91.0,
        bui: 76.0,
        pct_dead_fir: 35.0,
        grass_fuel_load: 0.35,
        percentile_growth: 90.0,
    };
    match threads {
        None => run_grid(&input).unwrap(),
        Some(t) => run_grid_with_threads(&input, t).unwrap(),
    }
}

fn assert_bit_identical(a: &BehaviourGrids, b: &BehaviourGrids, what: &str) {
    let pairs = [
        ("hros", &a.hros, &b.hros),
        ("bros", &a.bros, &b.bros),
        ("raz", &a.raz, &b.raz),
        ("wsv", &a.wsv, &b.wsv),
        ("hfi", &a.hfi, &b.hfi),
        ("rso", &a.rso, &b.rso),
        ("sros", &a.sros, &b.sros),
        ("sfc", &a.sfc, &b.sfc),
        ("fmc", &a.fmc, &b.fmc),
        ("accel", &a.accel, &b.accel),
    ];
    for (name, x, y) in pairs {
        assert_eq!(x.len(), y.len(), "{what}: {name} length");
        for (i, (p, q)) in x.iter().zip(y).enumerate() {
            assert!(
                p.to_bits() == q.to_bits() || (p.is_nan() && q.is_nan()),
                "{what}: {name}[{i}] serial {p:?} vs threaded {q:?}"
            );
        }
    }
}

#[test]
fn every_thread_count_matches_the_serial_pass() {
    // Large enough to take the parallel path; not a multiple of any chunk size.
    let c = cells(20_011);
    let serial = grids_for(&c, None);
    assert!(
        serial.hros.iter().any(|v| v.is_finite()),
        "grid has real results"
    );
    assert!(serial.hros.iter().any(|v| v.is_nan()), "grid has NaN cells");
    for threads in [0, 1, 2, 3, 7, 64] {
        let threaded = grids_for(&c, Some(threads));
        assert_bit_identical(&serial, &threaded, &format!("threads={threads}"));
    }
}

#[test]
fn run_grid_is_the_single_thread_pass() {
    let c = cells(10_000);
    assert_bit_identical(&grids_for(&c, None), &grids_for(&c, Some(1)), "threads=1");
}

#[test]
fn small_and_empty_grids_ignore_the_thread_count() {
    for n in [0, 1, 9, 4_095] {
        let c = cells(n);
        let serial = grids_for(&c, None);
        for threads in [0, 8] {
            assert_bit_identical(&serial, &grids_for(&c, Some(threads)), &format!("n={n}"));
        }
    }
}

#[test]
fn more_threads_than_chunks_is_fine() {
    // 4,096 cells is the smallest grid that goes parallel.
    let c = cells(4_096);
    let serial = grids_for(&c, None);
    assert_bit_identical(&serial, &grids_for(&c, Some(10_000)), "threads=10000");
}
