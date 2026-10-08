//! Full-chain percentile growth against the Python implementation: every
//! Wotton case at each recorded percentile (`tests/cffbps/data/golden/
//! percentile_growth_snapshot.json`). Regenerate with `tools/gen_fbp_goldens.py`
//! when the Python spec changes.

use cffdrs_core::fbp::run;
mod common;

use common::parse_inputs;

#[test]
fn scalar_core_matches_percentile_snapshot() {
    let inputs = parse_inputs();
    let snapshot = common::load_snapshot("percentile_growth_snapshot.json");

    let cases = snapshot["cases"].as_array().expect("cases");
    assert!(cases.len() >= 18, "expected the full Wotton case set");

    let mut checked = 0usize;
    for case in cases {
        let id = case["id"].as_i64().expect("case id");
        let code = case["fuel_type_code"].as_str().unwrap_or("?");
        let base = inputs
            .get(&id)
            .unwrap_or_else(|| panic!("no CSV inputs for case {id}"));
        for (pct_key, outputs) in case["percentiles"].as_object().expect("percentiles") {
            let mut input = base.clone();
            input.percentile_growth = pct_key.parse().expect("percentile key");
            let result = run(&input);
            for (name, expected) in outputs.as_object().expect("outputs") {
                let actual = result.get(name).unwrap_or_else(|| {
                    panic!("FbpResult has no accessor for golden field {name:?}")
                });
                let ok = match expected.as_f64() {
                    None => actual.is_nan(),
                    Some(e) => (actual - e).abs() <= e.abs().max(1e-12) * 1e-9,
                };
                assert!(
                    ok,
                    "case {id} ({code}) percentile {pct_key} field {name}: expected {expected}, got {actual}"
                );
                checked += 1;
            }
        }
    }
    // 20 cases x 9 percentiles x 7 outputs
    assert!(checked >= 1200, "only {checked} golden quantities checked");
}
