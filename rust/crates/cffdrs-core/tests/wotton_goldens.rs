//! Scalar-core validation against the SAME golden fixtures the Python suite
//! uses: inputs from `tests/cffbps/data/Inputs_for_Test_Cases_Wotton2009.csv`
//! joined by case id to `tests/cffbps/data/golden/
//! wotton2009_scalar_snapshot.json`. Regenerate the snapshot with
//! `tools/gen_fbp_goldens.py` when the Python spec changes; this suite
//! then holds the Rust core to it.

use cffdrs_core::fbp::{run, FbpResult};
mod common;

use common::parse_inputs;

#[test]
fn scalar_core_matches_wotton_snapshot() {
    let inputs = parse_inputs();
    let snapshot = common::load_snapshot("wotton2009_scalar_snapshot.json");

    let cases = snapshot["cases"].as_array().expect("cases");
    assert!(cases.len() >= 18, "expected the full Wotton case set");

    let mut checked = 0usize;
    for case in cases {
        let id = case["id"].as_i64().expect("case id");
        let code = case["fuel_type_code"].as_str().unwrap_or("?");
        let input = inputs
            .get(&id)
            .unwrap_or_else(|| panic!("no CSV inputs for case {id}"));
        let result = run(input);

        for (name, expected) in case["outputs"].as_object().expect("outputs") {
            let actual = result
                .get(name)
                .unwrap_or_else(|| panic!("FbpResult has no accessor for golden field {name:?}"));
            // null golden = unmasked NaN in the Python export (scalarize maps
            // NaN -> None); masked values export as 0.0 and appear numeric.
            // ffc/wfc: the scalar export maps MASKED -> 0.0 via `.item()`,
            // while the core (the grid pass) reports NaN there — apply the
            // export convention harness-side.
            let actual = if matches!(name.as_str(), "ffc" | "wfc") && actual.is_nan() {
                0.0
            } else {
                actual
            };
            let ok = match expected.as_f64() {
                None => actual.is_nan(),
                Some(e) => (actual - e).abs() <= e.abs().max(1e-12) * 1e-9,
            };
            assert!(
                ok,
                "case {id} ({code}) field {name}: expected {expected}, got {actual}"
            );
            checked += 1;
        }
    }
    // 20 cases x 54 quantities — make silent shrinkage impossible
    assert!(checked >= 1000, "only {checked} golden quantities checked");
}

#[test]
fn field_name_table_matches_snapshot_outputs() {
    let snapshot = common::load_snapshot("wotton2009_scalar_snapshot.json");
    let listed: Vec<&str> = snapshot["outputs"]
        .as_array()
        .expect("outputs list")
        .iter()
        .map(|v| v.as_str().expect("output name"))
        .collect();

    let result = FbpResult::default();
    for name in &listed {
        assert!(
            result.get(name).is_some(),
            "snapshot output {name:?} does not resolve via get()"
        );
    }
    for (name, _) in result.named_values() {
        assert!(
            listed.contains(&name),
            "named_values() entry {name:?} is not in the snapshot outputs"
        );
    }
}
