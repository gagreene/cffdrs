//! The bit-level oracle's deterministic input generator and its
//! NaN-canonicalising comparison, shared by `characterization.rs` (the oracle)
//! and `stage_state.rs`. Kept apart from `common/mod.rs` so the golden test
//! binaries, which do not use it, carry no unused code.
//!
//! Include it with `#[path = "common/cases.rs"] mod cases;`.

use cffdrs_fire::fbp::FbpInput;

pub const CANONICAL_NAN: u64 = 0x7ff8_0000_0000_0000;

struct Lcg(u64);

// Top 53 bits of the LCG state scaled to [0, 1): exact in f64.
#[allow(clippy::cast_precision_loss)]
impl Lcg {
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1_u64 << 53) as f64)
    }
}

/// The number of runs of [`cases`].
#[allow(dead_code)]
pub const CASE_COUNT: usize = 25 * 48 * 6 + 25 * 4;

pub fn bits(x: f64) -> u64 {
    if x.is_nan() {
        CANONICAL_NAN
    } else {
        x.to_bits()
    }
}

/// Every run, in a fixed order: (fuel code, percentile, input).
pub fn cases() -> Vec<(i32, f64, FbpInput)> {
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
    // Appended after every original run (so their rows keep their index and their random draws):
    // negative d0 / dj / fmc overrides ask for the derived value, 4 per fuel code, from the plain
    // percentile-50 inputs of cells 0 and 24.
    let per_fuel = 48 * 6;
    for (f, &fuel) in fuels.iter().enumerate() {
        let plain = |cell: usize| out[f * per_fuel + cell * 6].2.clone();
        let (a, b) = (plain(0), plain(24));
        let mut d0 = a.clone();
        d0.d0_override = Some(-1.0);
        let mut dj = a;
        dj.dj_override = Some(-999.0);
        let mut fmc = b.clone();
        fmc.fmc_override = Some(-1.0);
        let mut mixed = b;
        mixed.d0_override = Some(150.0);
        mixed.dj_override = Some(-5.0);
        mixed.fmc_override = Some(-0.5);
        for i in [d0, dj, fmc, mixed] {
            out.push((fuel, 50.0, i));
        }
    }
    out
}

/// Bit-identical, with every NaN equal to every other NaN.
#[allow(dead_code)] // used by stage_state.rs, not by the oracle
pub fn same_bits(a: f64, b: f64) -> bool {
    bits(a) == bits(b)
}
