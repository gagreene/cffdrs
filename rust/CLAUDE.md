# cffdrs Rust workspace — conventions

Two crates, one version, released together:

- `crates/cffdrs-core` — pure Rust (zero runtime deps): the CFFBPS scalar
  chain and grid pass. Consumable as a cargo dependency by fire-growth
  engines that need the science without a Python runtime in the hot path.
- `crates/cffdrs-py` — thin PyO3 bindings over cffdrs-core; maturin builds
  the `cffdrs-rs` wheel for Python consumers.

## The Python package is the spec

`src/cffdrs/` remains the reference implementation. The Rust core mirrors
it; it never leads. Science changes land in Python first, the Rust port
follows in the same PR, and the shared goldens prove agreement.

## Workflow: types first, red tests before code

1. New components start as structs and `pub fn` signatures with `todo!()`
   bodies — `cargo check` clean before any behavior exists.
2. Every function with behavior gets a failing test before its body is
   written. `todo!()` panics count as red.
3. Test scenarios and expected values come from the shared fixtures, never
   invented: `tests/cffbps/data/Inputs_for_Test_Cases_Wotton2009.csv` +
   `tests/cffbps/data/golden/*.json` — the same files the Python suite
   asserts against. Regenerate with `uv run python tools/gen_fbp_goldens.py`
   after any Python-side science change; the Rust suite then holds the port
   to the new snapshot.

## Numeric expectations

- Everything is f64; match the Python operation order where practical.
- Golden tolerance: rtol 1e-9 (NaN must match NaN). Bit-identity with
  numpy is not a goal — document any field where op-order noise exceeds
  the tolerance rather than loosening it globally.
- Fuel-code conventions follow the Python package: 1..18 modeled,
  19/20 non-fuel (NaN behaviour surfaces), M-1/2 use pc, M-3/4 use pdf,
  O-1a/b use gfl/gcf.

## Building

```bash
cd rust
cargo test                       # core + goldens
# wheel (from crates/cffdrs-py):
#   set PYO3_PYTHON to the target venv's interpreter first — building
#   against whatever python3 is on PATH links symbols the runtime may lack
uv run maturin build --release -m crates/cffdrs-py/Cargo.toml
```
