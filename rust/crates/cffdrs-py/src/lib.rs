use pyo3::prelude::*;

/// Python bindings over cffdrs-core. Kept intentionally thin: the science
/// lives in cffdrs-core; the cffdrs Python package remains the reference
/// implementation and the spec.
#[pymodule]
fn cffdrs_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__core_version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
