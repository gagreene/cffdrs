//! Fixture loading shared by the golden test binaries.

use cffdrs_core::fbp::FbpInput;
use std::collections::HashMap;
use std::path::PathBuf;

pub fn repo_path(rel: &str) -> PathBuf {
    // crates/cffdrs-core -> repo root
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join(rel)
}

// One field per CSV column, so the length tracks the fixture schema.
#[allow(clippy::too_many_lines)]
pub fn parse_inputs() -> HashMap<i64, FbpInput> {
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

/// Load a golden snapshot JSON under `tests/cffbps/data/golden/`.
pub fn load_snapshot(file_name: &str) -> serde_json::Value {
    let rel = format!("tests/cffbps/data/golden/{file_name}");
    serde_json::from_str(&std::fs::read_to_string(repo_path(&rel)).expect("golden snapshot"))
        .expect("valid json")
}
