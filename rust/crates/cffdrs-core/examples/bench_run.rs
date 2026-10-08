//! Release-profile timing of `run()` for before/after comparison:
//!
//!     cargo run --release --manifest-path rust/Cargo.toml -p cffdrs-core --example bench_run

// Timing arithmetic on small, bounded counters; the casts are exact here.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]

use cffdrs_core::fbp::{run, FbpInput};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    const CELLS: usize = 200_000;
    const REPEATS: usize = 9;
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut unit = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64) / ((1_u64 << 53) as f64)
    };
    let inputs: Vec<FbpInput> = (0..CELLS)
        .map(|k| FbpInput {
            fuel_type: 1 + (k % 18) as i32,
            wx_date: 20_230_615,
            lat: 45.0 + 17.0 * unit(),
            long: -125.0 + 35.0 * unit(),
            elevation: 1800.0 * unit(),
            slope_pct: 60.0 * unit(),
            aspect_deg: 360.0 * unit(),
            ws: 40.0 * unit(),
            wd: 360.0 * unit(),
            ffmc: 70.0 + 28.0 * unit(),
            bui: 150.0 * unit(),
            pc: 100.0 * unit(),
            pdf: 100.0 * unit(),
            gfl: 0.35,
            gcf: 100.0 * unit(),
            percentile_growth: if k % 2 == 0 { 50.0 } else { 90.0 },
            d0_override: None,
            dj_override: None,
            fmc_override: None,
            hros_override: None,
        })
        .collect();
    let mut ns_per_cell: Vec<f64> = (0..REPEATS)
        .map(|_| {
            let start = Instant::now();
            let mut acc = 0.0;
            for input in &inputs {
                acc += black_box(run(black_box(input))).hros;
            }
            black_box(acc);
            start.elapsed().as_nanos() as f64 / CELLS as f64
        })
        .collect();
    ns_per_cell.sort_by(f64::total_cmp);
    println!(
        "min {:.1} ns/cell, median {:.1} ns/cell",
        ns_per_cell[0],
        ns_per_cell[REPEATS / 2]
    );
}
