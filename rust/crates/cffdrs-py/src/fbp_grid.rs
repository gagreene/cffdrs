//! Owned `NumPy` snapshots and Python lifecycle for the Rust stage grid.

use cffbps_rust::fbp_grid::{FbpGrid as CoreGrid, ParamName, StageError};
use cffbps_rust::field::Field;
use cffbps_rust::grid::{GridInput, PerCell};
use cffbps_rust::stage::Stage;
use numpy::{PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// A scalar remains constant; an array is copied in logical C order.
enum Snapshot {
    Const(f64),
    Cells(Vec<f64>, Vec<usize>),
}

impl Snapshot {
    fn read(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Cast before filling so masked integer inputs can represent NaN.
        let array = py
            .import("numpy.ma")?
            .getattr("asarray")?
            .call1((value, "float64"))?;
        let filled = array.call_method1("filled", (f64::NAN,))?;
        let array = py.import("numpy")?.getattr("asarray")?.call1((filled,))?;
        let readonly = array.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
        let shape = readonly.shape().to_vec();
        let values: Vec<f64> = readonly.as_array().iter().copied().collect();
        if shape.is_empty() {
            Ok(Self::Const(values[0]))
        } else {
            Ok(Self::Cells(values, shape))
        }
    }

    fn per_cell(&self) -> PerCell<'_> {
        match self {
            Self::Const(v) => PerCell::Const(*v),
            Self::Cells(v, _) => PerCell::Cells(v),
        }
    }

    fn check_shape(&self, name: &str, expected: &[usize]) -> PyResult<()> {
        if let Self::Cells(_, shape) = self {
            check_shape(name, shape, expected)?;
        }
        Ok(())
    }
}

fn check_shape(name: &str, found: &[usize], expected: &[usize]) -> PyResult<()> {
    if found != expected {
        return Err(PyValueError::new_err(format!(
            "{name} shape {found:?} != grid shape {expected:?}"
        )));
    }
    Ok(())
}

/// Fuel arrays must already contain the wrapper's int32 codes, including 0
/// for masked/invalid cells. Integer scalars can broadcast over the grid.
fn fuel_snapshot(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<(Vec<i32>, Vec<usize>)> {
    let array = py.import("numpy")?.getattr("asarray")?.call1((value,))?;
    let shape = array.getattr("shape")?.extract::<Vec<usize>>()?;
    if shape.is_empty() {
        return Ok((vec![array.call_method0("item")?.extract::<i32>()?], shape));
    }
    let readonly = array.extract::<PyReadonlyArrayDyn<'_, i32>>()?;
    Ok((readonly.as_array().iter().copied().collect(), shape))
}

fn field(name: &str) -> PyResult<Field> {
    Field::from_name(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown field {name:?}; valid names: {}",
            Field::ALL.map(Field::name).join(", ")
        ))
    })
}

fn stage(name: &str) -> PyResult<Stage> {
    Stage::from_name(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown stage {name:?}; valid names: {}",
            Stage::ALL.map(Stage::name).join(", ")
        ))
    })
}

fn requested_fields(names: Option<Vec<String>>, default: &[Field]) -> PyResult<Vec<Field>> {
    names.map_or_else(
        || Ok(default.to_vec()),
        |names| names.iter().map(|name| field(name)).collect(),
    )
}

const DEFAULT_OUTPUTS: [Field; 10] = [
    Field::Hros,
    Field::Bros,
    Field::Raz,
    Field::Wsv,
    Field::Hfi,
    Field::Rso,
    Field::Sros,
    Field::Sfc,
    Field::Fmc,
    Field::Accel,
];
fn stage_error(error: &StageError) -> PyErr {
    if *error == StageError::AfterFused {
        PyRuntimeError::new_err("call reset() before running stages after run()")
    } else {
        PyValueError::new_err(error.to_string())
    }
}

/// Stateful, owned grid of per-cell FBP values. `NumPy` buffers are copied
/// before Rust work detaches from Python; returned arrays are independent.
#[pyclass(name = "FbpGrid", module = "cffdrs._rust")]
pub(crate) struct FbpGrid {
    grid: CoreGrid,
    shape: Vec<usize>,
    threads: usize,
}

#[pymethods]
impl FbpGrid {
    #[new]
    #[pyo3(signature = (
        fuel_type, lat, long, elevation, slope, aspect, pc, pdf, gfl, gcf,
        ws, wd, ffmc, bui, wx_date, percentile_growth=50.0, d0=None, dj=None,
        out_request=None, stages=None, keep=None, threads=1
    ))]
    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    fn new(
        py: Python<'_>,
        fuel_type: &Bound<'_, PyAny>,
        lat: &Bound<'_, PyAny>,
        long: &Bound<'_, PyAny>,
        elevation: &Bound<'_, PyAny>,
        slope: &Bound<'_, PyAny>,
        aspect: &Bound<'_, PyAny>,
        pc: &Bound<'_, PyAny>,
        pdf: &Bound<'_, PyAny>,
        gfl: &Bound<'_, PyAny>,
        gcf: &Bound<'_, PyAny>,
        ws: &Bound<'_, PyAny>,
        wd: &Bound<'_, PyAny>,
        ffmc: &Bound<'_, PyAny>,
        bui: &Bound<'_, PyAny>,
        wx_date: i64,
        percentile_growth: f64,
        d0: Option<&Bound<'_, PyAny>>,
        dj: Option<&Bound<'_, PyAny>>,
        out_request: Option<Vec<String>>,
        stages: Option<Vec<String>>,
        keep: Option<Vec<String>>,
        threads: i64,
    ) -> PyResult<Self> {
        let threads = usize::try_from(threads).map_err(|_| {
            PyValueError::new_err("threads must be 0 (all cores) or a positive count")
        })?;
        let outputs = requested_fields(out_request, &DEFAULT_OUTPUTS)?;
        let planned = stages.map_or_else(
            || Ok(Stage::ALL.to_vec()),
            |names| {
                names
                    .iter()
                    .map(|name| stage(name))
                    .collect::<PyResult<Vec<_>>>()
            },
        )?;
        let keep = keep
            .unwrap_or_default()
            .iter()
            .map(|name| field(name))
            .collect::<PyResult<Vec<_>>>()?;
        let (mut fuel, fuel_shape) = fuel_snapshot(py, fuel_type)?;
        let inputs = [
            ("lat", lat),
            ("long", long),
            ("elevation", elevation),
            ("slope", slope),
            ("aspect", aspect),
            ("pc", pc),
            ("pdf", pdf),
            ("gfl", gfl),
            ("gcf", gcf),
            ("ws", ws),
            ("wd", wd),
            ("ffmc", ffmc),
            ("bui", bui),
        ]
        .into_iter()
        .map(|(name, value)| Ok((name, Snapshot::read(py, value)?)))
        .collect::<PyResult<Vec<_>>>()?;
        let d0 = d0.map(|v| Snapshot::read(py, v)).transpose()?;
        let dj = dj.map(|v| Snapshot::read(py, v)).transpose()?;
        let shape = if fuel_shape.is_empty() {
            inputs
                .iter()
                .map(|(_, v)| v)
                .chain(d0.iter())
                .chain(dj.iter())
                .find_map(|v| match v {
                    Snapshot::Cells(_, s) => Some(s.clone()),
                    Snapshot::Const(_) => None,
                })
                .unwrap_or_default()
        } else {
            fuel_shape.clone()
        };
        for (name, value) in &inputs {
            value.check_shape(name, &shape)?;
        }
        if let Some(v) = &d0 {
            v.check_shape("d0", &shape)?;
        }
        if let Some(v) = &dj {
            v.check_shape("dj", &shape)?;
        }
        if fuel_shape.is_empty() {
            fuel.resize(shape.iter().product(), fuel[0]);
        }
        let grid = py
            .detach(|| {
                let input = GridInput {
                    fuel_type: &fuel,
                    lat: inputs[0].1.per_cell(),
                    long: inputs[1].1.per_cell(),
                    elevation: inputs[2].1.per_cell(),
                    slope_pct: inputs[3].1.per_cell(),
                    aspect_deg: inputs[4].1.per_cell(),
                    pct_conifer: inputs[5].1.per_cell(),
                    pct_dead_fir: inputs[6].1.per_cell(),
                    grass_fuel_load: inputs[7].1.per_cell(),
                    grass_curing: inputs[8].1.per_cell(),
                    ws: inputs[9].1.per_cell(),
                    wd: inputs[10].1.per_cell(),
                    ffmc: inputs[11].1.per_cell(),
                    bui: inputs[12].1.per_cell(),
                    wx_date,
                    percentile_growth,
                    d0: d0.as_ref().map(Snapshot::per_cell),
                    dj: dj.as_ref().map(Snapshot::per_cell),
                };
                let mut grid = CoreGrid::new(&input, &outputs)?;
                grid.plan(&planned, &keep);
                Ok::<_, cffbps_rust::grid::GridError>(grid)
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            grid,
            shape,
            threads,
        })
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.shape.iter().copied())
    }

    fn run_stage(&mut self, py: Python<'_>, name: &str) -> PyResult<()> {
        let stage = stage(name)?;
        py.detach(|| self.grid.run_stage(stage, self.threads))
            .map_err(|e| stage_error(&e))
    }

    fn run(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.grid.run_fused(self.threads))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn reset(&mut self, py: Python<'_>) {
        py.detach(|| self.grid.reset());
    }

    fn run_planned(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.grid.run_planned(self.threads))
            .map_err(|e| stage_error(&e))
    }

    fn set_param(&mut self, py: Python<'_>, name: &str, value: f64) -> PyResult<()> {
        let param = match name {
            "wx_date" => ParamName::WxDate,
            "percentile_growth" => ParamName::PercentileGrowth,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown parameter {name:?}; valid names: wx_date, percentile_growth"
                )))
            }
        };
        py.detach(|| self.grid.set_param(param, value))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn set_override(
        &mut self,
        py: Python<'_>,
        name: &str,
        values: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let field = match name {
            "d0" => Field::D0Override,
            "dj" => Field::DjOverride,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown override {name:?}; valid names: d0, dj"
                )))
            }
        };
        let values = values.map(|v| Snapshot::read(py, v)).transpose()?;
        if let Some(v) = &values {
            v.check_shape(name, &self.shape)?;
        }
        py.detach(|| {
            self.grid
                .set_override(field, values.as_ref().map(Snapshot::per_cell))
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn set(&mut self, py: Python<'_>, name: &str, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let field = field(name)?;
        let values = Snapshot::read(py, values)?;
        values.check_shape(name, &self.shape)?;
        py.detach(|| match values {
            Snapshot::Const(v) => self.grid.set_const(field, v),
            Snapshot::Cells(v, _) => self.grid.set(field, &v),
        })
        .map_err(|e| stage_error(&e))
    }

    fn set_fuel(&mut self, py: Python<'_>, codes: &Bound<'_, PyAny>) -> PyResult<()> {
        let (mut codes, shape) = fuel_snapshot(py, codes)?;
        if shape.is_empty() {
            codes.resize(self.grid.len(), codes[0]);
        } else {
            check_shape("fuel_type", &shape, &self.shape)?;
        }
        py.detach(|| self.grid.set_fuel(&codes))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn run_acceleration_for_fuel(
        &mut self,
        py: Python<'_>,
        codes: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let (mut codes, shape) = fuel_snapshot(py, codes)?;
        if shape.is_empty() {
            codes.resize(self.grid.len(), codes[0]);
        } else {
            check_shape("fuel_type", &shape, &self.shape)?;
        }
        py.detach(|| self.grid.run_acceleration_for_fuel(&codes, self.threads))
            .map_err(|e| stage_error(&e))
    }
    fn get<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let field = field(name)?;
        let values = py
            .detach(|| self.grid.get(field))
            .map_err(|e| stage_error(&e))?;
        PyArray1::from_vec(py, values).reshape(numpy::ndarray::IxDyn(&self.shape))
    }

    #[getter]
    fn dj_override_clipped(&self, py: Python<'_>) -> usize {
        py.detach(|| self.grid.dj_override_clipped())
    }

    fn live_bytes(&self, py: Python<'_>) -> usize {
        py.detach(|| self.grid.live_bytes())
    }
}
