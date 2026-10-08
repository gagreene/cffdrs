//! Local refactoring oracle for `cffdrs-core` (not a CI test).
//!
//! Runs `run()` over a deterministic set of 7,200 inputs and records or compares
//! all 54 outputs of each as raw `f64` bits (NaN canonicalised). Record a baseline
//! on the SAME machine and toolchain before refactoring, then compare after each
//! step:
//!
//!   CFFDRS_BASELINE_RECORD=1 \
//!     cargo test --manifest-path rust/Cargo.toml -p cffdrs-core --test characterization -- --ignored
//!   cargo test --manifest-path rust/Cargo.toml -p cffdrs-core --test characterization -- --ignored
//!
//! The baseline is `rust/target/characterization-baseline.txt` (the workspace
//! `target/` directory, git-ignored), resolved from `CARGO_MANIFEST_DIR` because
//! Cargo runs integration tests from the package directory. Set `CFFDRS_BASELINE`
//! to an absolute path to override it (for example if `CARGO_TARGET_DIR` is set).

use cffdrs_core::fbp::{run, FbpInput};
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

const CANONICAL_NAN: u64 = 0x7ff8_0000_0000_0000;

fn bits(x: f64) -> u64 {
    if x.is_nan() {
        CANONICAL_NAN
    } else {
        x.to_bits()
    }
}

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1_u64 << 53) as f64)
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// Every run, in a fixed order: (fuel code, percentile, input).
fn cases() -> Vec<(i32, f64, FbpInput)> {
    let mut rng = Lcg(0x5eed_cafe_f00d_d00d);
    let mut out = Vec::new();
    let fuels: Vec<i32> = (-1..=22).chain([99]).collect(); // 25 codes
    for &fuel in &fuels {
        for cell in 0..48_u32 {
            for percentile in [50.0, 5.0, 95.0, f64::NAN, 0.0, 100.0] {
                let mut i = FbpInput {
                    fuel_type: fuel,
                    wx_date: 20_230_000
                        + 100 * (1 + i64::from(cell % 12))
                        + 1
                        + i64::from(cell % 28),
                    lat: rng.range(45.0, 62.0),
                    long: rng.range(-125.0, -90.0),
                    elevation: rng.range(-10.0, 1800.0),
                    slope_pct: rng.range(-5.0, 90.0),
                    aspect_deg: rng.range(-10.0, 360.0),
                    ws: rng.range(-2.0, 50.0),
                    wd: rng.range(0.0, 360.0),
                    ffmc: rng.range(60.0, 98.0),
                    bui: rng.range(0.0, 200.0),
                    pc: rng.range(0.0, 100.0),
                    pdf: rng.range(0.0, 100.0),
                    gfl: rng.range(0.1, 0.6),
                    gcf: rng.range(0.0, 100.0),
                    percentile_growth: percentile,
                    d0_override: None,
                    dj_override: None,
                    fmc_override: None,
                    hros_override: None,
                };
                match cell % 24 {
                    1 => i.lat = f64::NAN,
                    2 => i.long = f64::NAN,
                    3 => i.elevation = f64::NAN,
                    4 => i.slope_pct = f64::NAN,
                    5 => i.aspect_deg = f64::NAN,
                    6 => i.ws = f64::NAN,
                    7 => i.wd = f64::NAN,
                    8 => i.ffmc = f64::NAN,
                    9 => i.bui = f64::NAN,
                    10 => i.pc = f64::NAN,
                    11 => i.pdf = f64::NAN,
                    12 => i.gfl = f64::NAN,
                    13 => i.gcf = f64::NAN,
                    14 => i.gcf = 0.0,
                    15 => {
                        i.d0_override = Some(rng.range(100.0, 200.0));
                        i.dj_override = Some(rng.range(100.0, 250.0));
                    }
                    16 => {
                        i.d0_override = Some(f64::NAN);
                        i.dj_override = Some(f64::NAN);
                    }
                    17 => i.fmc_override = Some(rng.range(80.0, 130.0)),
                    18 => i.fmc_override = Some(f64::NAN),
                    19 => i.hros_override = Some(rng.range(0.0, 40.0)),
                    20 => i.hros_override = Some(f64::NAN),
                    21 => i.wx_date = 20_231_301, // impossible month
                    22 => {
                        i.aspect_deg = -5.0; // flat-terrain rule
                        i.slope_pct = 85.0; // slope-factor cap
                    }
                    23 => i.ws = 60.0, // wind > 40 branch
                    _ => {}
                }
                out.push((fuel, percentile, i));
            }
        }
    }
    out
}

fn row(input: &FbpInput) -> Vec<u64> {
    let r = run(input);
    NAMES
        .iter()
        .map(|n| bits(r.get(n).unwrap_or_else(|| panic!("no field {n}"))))
        .collect()
}

#[test]
#[ignore = "local refactoring oracle: needs a recorded baseline; see the module docs"]
fn outputs_match_the_recorded_baseline() {
    let path = baseline_path();
    let cases = cases();
    assert_eq!(cases.len(), 25 * 48 * 6, "input set changed");
    let rows: Vec<Vec<u64>> = cases.iter().map(|(_, _, i)| row(i)).collect();

    if env::var("CFFDRS_BASELINE_RECORD").is_ok() {
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
            .map(|h| u64::from_str_radix(h, 16).unwrap())
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
