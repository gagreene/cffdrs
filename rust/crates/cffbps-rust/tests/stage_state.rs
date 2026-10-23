//! The stage engine against the scalar pass, and the reference table of
//! stage reads and writes against the stage bodies.
//!
//! Runs on every input of the bit-level oracle (`characterization.rs`).

#[path = "common/cases.rs"]
mod cases;

use cases::{cases, same_bits};
use cffbps_rust::fbp::{run, FbpInput, FbpResult};
use cffbps_rust::field::Field;
use cffbps_rust::stage::Stage;
use cffbps_rust::state::CellState;

// ---- helpers ----
fn generated_inputs() -> Vec<FbpInput> {
    cases().into_iter().map(|(_, _, input)| input).collect()
}

/// Every stage in canonical order on a fresh state.
fn run_stages(input: &FbpInput) -> FbpResult {
    let mut s = CellState::from_input(input);
    for stage in Stage::ALL {
        s.apply(stage);
    }
    s.into_result()
}

/// Each stage paired with the state just before it ran.
fn run_stages_full(input: &FbpInput) -> Vec<(Stage, CellState)> {
    let mut s = CellState::from_input(input);
    let mut out = Vec::with_capacity(Stage::ALL.len());
    for stage in Stage::ALL {
        out.push((stage, s.clone()));
        s.apply(stage);
    }
    out
}

// ---- stage tests ----
#[test]
fn stage_composition_equals_run_bit_for_bit() {
    for (n, input) in generated_inputs().iter().enumerate() {
        let fused = run(input).named_values();
        let staged = run_stages(input).named_values();
        for ((name, a), (_, b)) in fused.iter().zip(staged.iter()) {
            assert!(
                same_bits(*a, *b),
                "case {n} (fuel {}, percentile {}) {name}: run {a} stages {b}",
                input.fuel_type,
                input.percentile_growth,
            );
        }
    }
}

#[test]
fn every_stage_reads_only_what_it_declares() {
    // Two poison values: NaN (hides in NaN-valued outputs) and a finite
    // sentinel (changes finite outputs).
    for poison in [f64::NAN, 987_654.321] {
        for input in generated_inputs() {
            let reference = run_stages_full(&input);
            for (stage, before) in &reference {
                let mut full = before.clone();
                let mut poisoned = before.clone();
                for f in Field::ALL {
                    if !stage.reads().contains(&f) {
                        poisoned.set(f, poison);
                    }
                }
                full.apply(*stage);
                poisoned.apply(*stage);
                for f in stage.writes() {
                    assert!(
                        same_bits(full.get(*f), poisoned.get(*f)),
                        "{stage:?} wrote {f:?} from an undeclared field (poison {poison}, fuel {})",
                        input.fuel_type,
                    );
                }
            }
        }
    }
}

#[test]
fn every_stage_writes_only_what_it_declares() {
    for input in generated_inputs() {
        for (stage, before) in run_stages_full(&input) {
            let mut after = before.clone();
            after.apply(stage);
            for f in Field::ALL {
                if !stage.writes().contains(&f) {
                    assert_eq!(
                        before.get(f).to_bits(),
                        after.get(f).to_bits(),
                        "{stage:?} changed undeclared {f:?} (fuel {})",
                        input.fuel_type,
                    );
                }
            }
        }
    }
}

// ---- other tests ----
#[test]
fn a_template_holds_the_python_initialize_values() {
    let s = CellState::template(
        2,
        cffbps_rust::state::StageParams {
            wx_date: 20_230_615,
            percentile_growth: 50.0,
            d0: None,
            dj: None,
            fmc: None,
        },
    );
    for f in Field::ALL {
        let want = if f == Field::FuelType {
            2.0
        } else {
            f.template_value()
        };
        assert!(same_bits(s.get(f), want), "{f:?}: {}", s.get(f));
    }
}

#[test]
fn from_input_stores_the_normalised_inputs() {
    let mut input = FbpInput::new(6, 20_230_615);
    input.slope_pct = -3.0;
    input.aspect_deg = -1.0;
    input.ws = -2.0;
    input.wd = 90.0;
    input.long = -115.0;
    input.gcf = 0.0;
    let s = CellState::from_input(&input);
    assert_eq!(s.get(Field::Slope), 0.0);
    assert_eq!(s.get(Field::AspectIn), 270.0);
    assert_eq!(s.get(Field::Ws), 0.0);
    assert_eq!(s.get(Field::Long), 115.0);
    assert_eq!(s.get(Field::Gcf), 0.1);
    assert_eq!(s.get(Field::FuelType), 6.0);
    // wd / aspect read as the input until invertWindAspect runs
    assert_eq!(s.get(Field::Wd), 90.0);
    assert_eq!(s.get(Field::Aspect), 270.0);
    let mut s = s;
    s.apply(Stage::InvertWindAspect);
    s.apply(Stage::InvertWindAspect);
    assert_eq!(
        s.get(Field::Wd),
        270.0,
        "repeating the stage never flips twice"
    );
    assert_eq!(s.get(Field::Aspect), 90.0);
}

#[test]
fn set_fuel_type_changes_the_fuel_the_stages_use() {
    let mut input = FbpInput::new(2, 20_230_615);
    input.ffmc = 90.0;
    input.bui = 80.0;
    let mut s = CellState::from_input(&input);
    s.apply(Stage::CalcSfc);
    let c2 = s.get(Field::Sfc);
    s.set(Field::FuelType, 8.0);
    s.apply(Stage::CalcSfc);
    assert_ne!(c2.to_bits(), s.get(Field::Sfc).to_bits());
    assert_eq!(s.into_result().fuel_type, 8.0);
}
