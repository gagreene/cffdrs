//! The pure percentile ROS adjustment against direct outputs of the Python
//! `calc_ros_percentile_growth` (`tests/cffbps/data/golden/
//! percentile_ros_function_snapshot.json`). Regenerate with
//! `tools/gen_fbp_goldens.py` when the Python spec changes.

use cffdrs_core::percentile::{percentile_ros, percentile_tinv, wind_decay};
use std::path::PathBuf;

fn load(rel: &str) -> serde_json::Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join(rel);
    serde_json::from_str(&std::fs::read_to_string(path).expect("golden file")).expect("valid json")
}

fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= expected.abs().max(1e-12) * 1e-9
}

#[test]
fn percentile_ros_matches_python_function() {
    let snapshot = load("tests/cffbps/data/golden/percentile_ros_function_snapshot.json");
    let rows = snapshot["rows"].as_array().expect("rows");
    assert!(rows.len() >= 700, "snapshot shrank: {} rows", rows.len());
    for row in rows {
        let fuel = row["fuel_type"].as_i64().unwrap() as i32;
        let (cfb, pct) = (
            row["regime_cfb"].as_f64().unwrap(),
            row["percentile"].as_f64().unwrap(),
        );
        let (ros, wsv) = (row["ros"].as_f64().unwrap(), row["wsv"].as_f64().unwrap());
        let tinv = percentile_tinv(pct);
        let head = percentile_ros(fuel, ros, cfb, tinv, 1.0);
        let back = percentile_ros(fuel, ros, cfb, tinv, wind_decay(wsv));
        let (eh, eb) = (row["hros"].as_f64().unwrap(), row["bros"].as_f64().unwrap());
        assert!(
            close(head, eh),
            "head fuel={fuel} cfb={cfb} p={pct} ros={ros}: expected {eh}, got {head}"
        );
        assert!(
            close(back, eb),
            "back fuel={fuel} cfb={cfb} p={pct} ros={ros}: expected {eb}, got {back}"
        );
    }
}

#[test]
fn wind_decay_matches_python() {
    let snapshot = load("tests/cffbps/data/golden/percentile_ros_function_snapshot.json");
    for row in snapshot["wind_decay"].as_array().expect("wind_decay") {
        let (w, k) = (row["wsv"].as_f64().unwrap(), row["k"].as_f64().unwrap());
        assert!(
            close(wind_decay(w), k),
            "k({w}): expected {k}, got {}",
            wind_decay(w)
        );
    }
}
