//! Non-modeled cells (fuel codes 19 non-fuel, 20 water, and invalid codes 0,
//! 99, -1): `NonModeled::PythonCompat` gives exactly what the Python package's
//! `FBP.getParams` returns for every output, held to
//! `tests/cffbps/data/golden/non_modeled_snapshot.json`
//! (`tools/gen_fbp_goldens.py --non-modeled-only`); `NonModeled::Nan` keeps the
//! engine contract (every requested field NaN).

mod common;

use cffbps_rust::field::Field;
use cffbps_rust::grid::{run_grid_fields, GridInput, NonModeled, PerCell};
use serde_json::Value;

/// Golden tolerance (the shared fixtures' rtol).
const RTOL: f64 = 1e-9;

fn number(v: &Value) -> f64 {
    // `null` is a NaN (missing) value in the snapshots.
    v.as_f64().unwrap_or(f64::NAN)
}

struct Scenario {
    name: String,
    inputs: serde_json::Map<String, Value>,
}

impl Scenario {
    fn get(&self, key: &str) -> PerCell<'static> {
        PerCell::Const(number(&self.inputs[key]))
    }

    fn override_of(&self, key: &str) -> Option<PerCell<'static>> {
        self.inputs.get(key).map(|v| PerCell::Const(number(v)))
    }
}

fn scenarios(snapshot: &Value) -> Vec<Scenario> {
    snapshot["scenarios"]
        .as_array()
        .expect("scenarios")
        .iter()
        .map(|s| Scenario {
            name: s["name"].as_str().expect("name").to_owned(),
            inputs: s["inputs"].as_object().expect("inputs").clone(),
        })
        .collect()
}

fn grid<'a>(fuel: &'a [i32], s: &Scenario, wx_date: i64) -> GridInput<'a> {
    GridInput {
        fuel_type: fuel,
        lat: s.get("lat"),
        long: s.get("long"),
        elevation: s.get("elevation"),
        slope_pct: s.get("slope"),
        aspect_deg: s.get("aspect"),
        pct_conifer: s.get("pc"),
        pct_dead_fir: s.get("pdf"),
        grass_fuel_load: s.get("gfl"),
        grass_curing: s.get("gcf"),
        ws: s.get("ws"),
        wd: s.get("wd"),
        ffmc: s.get("ffmc"),
        bui: s.get("bui"),
        wx_date,
        percentile_growth: number(&s.inputs["percentile_growth"]),
        d0: s.override_of("d0"),
        dj: s.override_of("dj"),
        fmc: None,
    }
}

fn close(got: f64, want: f64) -> bool {
    if want.is_nan() {
        return got.is_nan();
    }
    got == want || (got - want).abs() <= RTOL * want.abs()
}

#[test]
fn python_compat_equals_the_python_values_for_every_output() {
    let snapshot = common::load_snapshot("non_modeled_snapshot.json");
    let wx_date = snapshot["wx_date"].as_i64().expect("wx_date");
    let names: Vec<&str> = snapshot["outputs"]
        .as_array()
        .expect("outputs")
        .iter()
        .map(|v| v.as_str().expect("name"))
        .collect();
    assert_eq!(names.len(), 54);
    let fields: Vec<Field> = names
        .iter()
        .map(|n| Field::from_name(n).unwrap_or_else(|| panic!("unknown output {n}")))
        .collect();
    let scenarios = scenarios(&snapshot);
    let cases = snapshot["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 5 * scenarios.len());
    let mut failures = Vec::new();
    for case in cases {
        let code = i32::try_from(case["fuel_type"].as_i64().expect("code")).expect("i32");
        let scenario_name = case["scenario"].as_str().expect("scenario");
        let s = scenarios
            .iter()
            .find(|s| s.name == scenario_name)
            .expect("known scenario");
        let fuel = [code];
        let out = run_grid_fields(
            &grid(&fuel, s, wx_date),
            &fields,
            1,
            NonModeled::PythonCompat,
        )
        .expect("valid grid");
        for (name, (field, values)) in names.iter().zip(&out.fields) {
            assert_eq!(field.name(), *name);
            let want = number(&case["outputs"][*name]);
            if !close(values[0], want) {
                failures.push(format!(
                    "code {code} {scenario_name} {name}: got {} want {want}",
                    values[0]
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn nan_mode_keeps_every_requested_field_nan() {
    let snapshot = common::load_snapshot("non_modeled_snapshot.json");
    let wx_date = snapshot["wx_date"].as_i64().expect("wx_date");
    let codes = [19, 20, 0, 99, -1];
    for s in scenarios(&snapshot) {
        let out = run_grid_fields(&grid(&codes, &s, wx_date), &Field::ALL, 1, NonModeled::Nan)
            .expect("valid grid");
        for (field, values) in &out.fields {
            assert!(
                values.iter().all(|v| v.is_nan()),
                "{} {}: {values:?}",
                s.name,
                field.name()
            );
        }
    }
}

#[test]
fn modeled_cells_do_not_depend_on_the_non_modeled_choice() {
    let snapshot = common::load_snapshot("non_modeled_snapshot.json");
    let wx_date = snapshot["wx_date"].as_i64().expect("wx_date");
    let codes = [2, 19, 6, 0];
    for s in scenarios(&snapshot) {
        let input = grid(&codes, &s, wx_date);
        let nan = run_grid_fields(&input, &Field::ALL, 1, NonModeled::Nan).expect("valid");
        let compat =
            run_grid_fields(&input, &Field::ALL, 1, NonModeled::PythonCompat).expect("valid");
        for ((field, a), (_, b)) in nan.fields.iter().zip(&compat.fields) {
            for i in [0, 2] {
                assert!(
                    a[i].to_bits() == b[i].to_bits() || (a[i].is_nan() && b[i].is_nan()),
                    "{} {}[{i}]",
                    s.name,
                    field.name()
                );
            }
        }
    }
}
