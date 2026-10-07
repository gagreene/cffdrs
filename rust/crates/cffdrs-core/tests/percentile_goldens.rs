//! Full-chain percentile growth against the Python implementation: every
//! Wotton case at each recorded percentile (`tests/cffbps/data/golden/
//! percentile_growth_snapshot.json`). Regenerate with `tools/gen_fbp_goldens.py`
//! when the Python spec changes.

use cffdrs_core::fbp::{run, FbpInput};
use std::collections::HashMap;
use std::path::PathBuf;

fn repo_path(rel: &str) -> PathBuf {
    // crates/cffdrs-core -> repo root
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join(rel)
}

fn parse_inputs() -> HashMap<i64, FbpInput> {
    let text = std::fs::read_to_string(repo_path(
        "tests/cffbps/data/Inputs_for_Test_Cases_Wotton2009.csv",
    ))
    .expect("fixture CSV");
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("header").trim().split(',').collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("column {name}"))
    };
    let (
        c_ft,
        c_date,
        c_lat,
        c_long,
        c_elev,
        c_aspect,
        c_slope,
        c_ws,
        c_wd,
        c_ffmc,
        c_bui,
        c_pc,
        c_pdf,
        c_gfl,
        c_gcf,
        c_id,
    ) = (
        col("fuel_type"),
        col("wx_date"),
        col("lat"),
        col("long"),
        col("elevation"),
        col("aspect"),
        col("slope"),
        col("ws"),
        col("wd"),
        col("ffmc"),
        col("bui"),
        col("pc"),
        col("pdf"),
        col("gfl"),
        col("gcf"),
        col("id"),
    );
    let (c_d0, c_dj) = (col("d0"), col("dj"));
    let num = |fields: &[&str], i: usize, default: f64| -> f64 {
        let v = fields[i].trim();
        if v.is_empty() {
            default
        } else {
            v.parse().unwrap_or_else(|_| panic!("bad number {v:?}"))
        }
    };

    let mut out = HashMap::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let id: i64 = f[c_id].trim().parse().expect("id");
        out.insert(
            id,
            FbpInput {
                fuel_type: f[c_ft].trim().parse().expect("fuel_type"),
                wx_date: f[c_date].trim().parse().expect("wx_date"),
                lat: num(&f, c_lat, f64::NAN),
                long: num(&f, c_long, f64::NAN),
                elevation: num(&f, c_elev, f64::NAN),
                slope_pct: num(&f, c_slope, f64::NAN),
                aspect_deg: num(&f, c_aspect, f64::NAN),
                ws: num(&f, c_ws, f64::NAN),
                wd: num(&f, c_wd, f64::NAN),
                ffmc: num(&f, c_ffmc, f64::NAN),
                bui: num(&f, c_bui, f64::NAN),
                // defaults matching FBP.initialize's kwargs for blank columns
                pc: num(&f, c_pc, 50.0),
                pdf: num(&f, c_pdf, 35.0),
                gfl: num(&f, c_gfl, 0.35),
                gcf: num(&f, c_gcf, 80.0),
                percentile_growth: 50.0,
                d0_override: {
                    let v = f[c_d0].trim();
                    if v.is_empty() {
                        None
                    } else {
                        Some(v.parse().expect("d0"))
                    }
                },
                dj_override: {
                    let v = f[c_dj].trim();
                    if v.is_empty() {
                        None
                    } else {
                        Some(v.parse().expect("dj"))
                    }
                },
                fmc_override: None,
                hros_override: None,
            },
        );
    }
    out
}

#[test]
fn scalar_core_matches_percentile_snapshot() {
    let inputs = parse_inputs();
    let snapshot: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(repo_path(
            "tests/cffbps/data/golden/percentile_growth_snapshot.json",
        ))
        .expect("golden snapshot"),
    )
    .expect("valid json");

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
