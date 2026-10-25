//! Local refactoring oracle for `cffdrs-fire` (not a CI test).
//!
//! Runs `run()` over a deterministic set of 7,200 inputs and records or compares
//! all 54 outputs of each as raw `f64` bits (NaN canonicalised). Record a baseline
//! on the SAME machine and toolchain before refactoring, then compare after each
//! step.
//!
//! Record (once, before refactoring):
//!
//!     CFFDRS_BASELINE_RECORD=1 \
//!       cargo test --manifest-path rust/Cargo.toml -p cffdrs-fire --test characterization -- --ignored
//!
//! Compare (after each step):
//!
//!     cargo test --manifest-path rust/Cargo.toml -p cffdrs-fire --test characterization -- --ignored
//!
//! Record mode is enabled only when `CFFDRS_BASELINE_RECORD` is exactly `1`; any
//! other value (including `0` or empty) compares, so it cannot overwrite the
//! baseline by accident.
//!
//! The baseline is `rust/target/characterization-baseline.txt` (the workspace
//! `target/` directory, git-ignored), resolved from `CARGO_MANIFEST_DIR` because
//! Cargo runs integration tests from the package directory. Set `CFFDRS_BASELINE`
//! to an absolute path to override it (for example if `CARGO_TARGET_DIR` is set).

#[path = "common/cases.rs"]
mod cases;

use cases::{bits, cases};
use cffdrs_fire::fbp::{run, FbpInput};
use std::path::PathBuf;
use std::{env, fs};

fn baseline_path() -> PathBuf {
    match env::var_os("CFFDRS_BASELINE") {
        Some(p) => PathBuf::from(p), // must be absolute
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/characterization-baseline.txt"),
    }
}

const NAMES: [&str; 54] = [
    "ws",
    "wd",
    "wse",
    "wse1",
    "wse2",
    "wsx",
    "wsy",
    "wsv",
    "raz",
    "m",
    "fF",
    "fW",
    "ffmc",
    "isi",
    "bui",
    "a",
    "b",
    "c",
    "q",
    "bui0",
    "be",
    "be_max",
    "sf",
    "rsz",
    "rsf",
    "isf",
    "rsi",
    "latn",
    "dj",
    "d0",
    "nd",
    "fmc",
    "fme",
    "ffc",
    "wfc",
    "sfc",
    "cfl",
    "cfc",
    "tfc",
    "cbh",
    "csfi",
    "rso",
    "cfb",
    "fire_type",
    "hros",
    "sros",
    "cros",
    "bfw",
    "bisi",
    "bros",
    "hfi",
    "fi_class",
    "accel",
    "fuel_type",
];

// ---- helpers ----
fn row(input: &FbpInput) -> Vec<u64> {
    let r = run(input);
    NAMES
        .iter()
        .map(|n| bits(r.get(n).unwrap_or_else(|| panic!("no field {n}"))))
        .collect()
}

// ---- other tests ----
#[test]
#[ignore = "local refactoring oracle: needs a recorded baseline; see the module docs"]
fn outputs_match_the_recorded_baseline() {
    let path = baseline_path();
    let cases = cases();
    assert_eq!(cases.len(), cases::CASE_COUNT, "input set changed");
    let rows: Vec<Vec<u64>> = cases.iter().map(|(_, _, i)| row(i)).collect();

    // Only the exact value "1" records: `=0`, an empty value or any other
    // value compares, so a stray variable cannot overwrite the baseline.
    if env::var("CFFDRS_BASELINE_RECORD").as_deref() == Ok("1") {
        let text: String = rows
            .iter()
            .map(|r| {
                r.iter()
                    .map(|b| format!("{b:016x}"))
                    .collect::<Vec<_>>()
                    .join(" ")
                    + "\n"
            })
            .collect();
        fs::write(&path, text).expect("write baseline");
        return;
    }

    let baseline = fs::read_to_string(&path).expect("read baseline (record it first)");
    let lines: Vec<&str> = baseline.lines().collect();
    assert_eq!(
        lines.len(),
        rows.len(),
        "baseline has a different number of runs"
    );
    for (n, ((fuel, percentile, _), (now, line))) in
        cases.iter().zip(rows.iter().zip(&lines)).enumerate()
    {
        let before: Vec<u64> = line
            .split(' ')
            .map(|h| {
                u64::from_str_radix(h, 16).unwrap_or_else(|e| {
                    panic!("baseline run {n}: {h:?} is not a 16-digit hex value: {e}")
                })
            })
            .collect();
        assert_eq!(
            before.len(),
            NAMES.len(),
            "baseline run {n} has {} values, expected {}",
            before.len(),
            NAMES.len()
        );
        for (k, (b, a)) in before.iter().zip(now).enumerate() {
            assert!(
                b == a,
                "run {n} (fuel {fuel}, percentile {percentile}) field {}: baseline {} now {}",
                NAMES[k],
                f64::from_bits(*b),
                f64::from_bits(*a),
            );
        }
    }
}
