# Known differences between the Python and Rust backends

The Python backend (`cffdrs.cffbps.python`) is the reference. The Rust backend
(`cffdrs.cffbps.rust`, and `cffdrs.cffbps.FBP` when the extension is installed) must
agree with it, except where this register says otherwise. This file is the single list
of known differences, so effort to close or accept them can be planned in one place.
`docs/CODEBASE.md` ("Outputs and intentional differences") describes the output
contract in prose; this register tracks each difference individually.

## How differences are found

`tools/compare_backends.py` runs both backends on the same scenarios and tabulates,
per output name, every difference of **value** (rtol 1e-9, NaN equal to NaN), **type**
(scalar input), **dtype**, **shape** and **container**:

```
uv run --no-sync python tools/compare_backends.py [--markdown]
```

Scenarios: every fuel code 1-20 plus invalid codes 0 and 99, each input missing in
turn, percentile 50 and 90, scalar / array / mixed (array fuel, scalar weather)
input, and three states (`init`: before any stage; `run`: after `runFBP()`;
`calcFMC`: after the first four stages). Last run (2026-10-11, tree after commit
`0bc0437` plus the fixes below): 4,356 runs, **no value difference**; the shape,
container and dtype differences below are the only ones.

Run the tool after any change to either backend and compare the table with the one
below. A new row, or a changed count, is a new difference to triage.

## Register

Status: **intentional** (decided, documented, tested), **accepted** (low impact, no
plan to change), **open** (should be revisited), **fixed** (kept for history).

### Found by the comparison tool

| ID | Area | Python | Rust | Impact | Status |
|---|---|---|---|---|---|
| BD-01 | Shape with mixed scalar / array inputs | Fields derived only from scalar inputs (`bui`, `ffmc`, `ws`, `wd` before inversion, `fF`, `fmc`, `fme`, `latn`, `nd`, `sf`, `d0`, `dj`, `m`) keep shape `(1,)`; the rest have the grid shape | Every field has the grid shape | Callers that index results get the same values; shapes differ | intentional (D12c); parity tests broadcast |
| BD-02 | `d0` / `dj` before `calcFMC`, array input | A bare `nan` (the getter returns `np.nan` for an unset attribute) | A full-shape array of NaN | None in practice; code that checks `np.isnan(x)` works for both | accepted |
| BD-03 | Dtype of `fire_type` / `fi_class` before the producing stage runs, array input | `fire_type` float64, `fi_class` int8 (leftover templates) | int64 for both | None; after `runFBP()` both are int64 | accepted |

### By design (not seen as a value difference by the tool)

| ID | Area | Python | Rust | Status |
|---|---|---|---|---|
| BD-04 | Numerical agreement | Reference | rtol 1e-9 for floats, exact for integers; not bit identical (operation order) | intentional |
| BD-05 | 0-d inputs | Some outputs are `memoryview` objects | 0-d inputs treated as scalars | intentional (D12c) |
| BD-06 | Python-only operations | `rosParams`, `open_fuel_types`, `non_crowning_fuels`, `fbpCBH_CFL_HT_LUT`, `block=`, `fbpMultiprocessArray`, `getCBH_CFL(ftype, cbh, cfl)` overrides, reading computed attributes such as `fbp.hros` | Not supported: raise with a message naming `backend='python'` (or `getParams([...])`); `AutoFBP` fixes the backend once at `initialize` | intentional (D7, D8) |
| BD-07 | In-place edits of arrays passed to `initialize` | Seen by the next calculation (references are held; the FMC cache is keyed on content) | Not seen: Rust owns copies; use `setParams`, assignment, `set_override` | intentional (D16e) |
| BD-08 | A `calcX` step after `runFBP()` | Allowed (everything is stored) | `RuntimeError`: call `reset()` first (unrequested intermediates were not stored) | intentional (D16a) |
| BD-09 | Freed fields | Never freed | Reading a field outside `out_request` after a run, or after its last planned reader, raises `ValueError` naming the field | intentional (D5) |
| BD-10 | Scalar fuel with grid inputs | Coefficient indexing defect in the reference for this mix | Parity tests broadcast the fuel explicitly | open: a reference defect, not a Rust one |
| BD-11 | Missing-input scalar exports | Masked scalar results export the backing data (zero, float) instead of NaN / the integer sentinel; unmasked NaN stays NaN | Reproduced by `rust/_scalar_masks.py`, a hand-written table of which stage consumes which input mask | accepted, **open as a risk**: see below |
| BD-12 | Threads | Argument ignored | `threads=1` default, 0 = all cores, results independent of the count | intentional (D6) |
| BD-13 | Direct crate API | n/a | `run_grid` and `BehaviourGrids` give all-NaN for non-modeled cells; `PythonCompat` (used by the Python package) reproduces Python | intentional |
| BD-14 | CuPy backend | Experimental, not audited for the NaN leak, not covered by the Rust comparison | n/a | open |

### Fixed during this work

| ID | What | Where |
|---|---|---|
| BD-F1 | Scalar fuels 10-13 (mixedwood) with missing `lat`, `long` or `elevation` and percentile not 50: Python exports `hros` / `bros` 0.0 (masked), Rust gave NaN | `_scalar_masks.py` growth rule now covers fuels 10-13; test `test_scalar_mixedwood_growth_keeps_the_foliar_moisture_mask` |
| BD-F2 | A missing `d0` / `dj` override exported 0.0 before `calcFMC` (found by the final scoped review) | `ScalarMasks.set`; test `test_missing_date_override_exports_nan_before_and_after_calcfmc` |
| BD-F4 | `fmc` became an override like `d0` / `dj` (sticky, `initialize(fmc=...)`, missing cell exports NaN); the CuPy backend has no override and only derives `fme` from a `fmc` given to `setParams` | Python/Rust `set_override('fmc')`; tests `test_fmc_override_*`, `test_override_routes[fmc-*]` |
| BD-F3 | Fuel dtype / mask / wrap, repeated `runFBP`, stale FMC results, stale injected values, unnormalised updates (five owner-authorised reference changes, D13) | Python reference, `docs/CODEBASE.md` |

## Where to focus next

1. **BD-11 is the main risk.** `rust/_scalar_masks.py` re-implements, by hand, which
   input masks reach which output in each stage. Python is the source of truth, so the
   table can drift whenever a Python equation changes. 159 regression tests pin it
   today. The cleaner long-term fix is for the native grid to return a mask with each
   result, so Python needs no parallel table. Until then, treat any change to a Python
   stage as a change to this table too.
2. **Widen the comparison tool** to the cases it does not cover yet: `setParams`
   injections between stages, explicit `d0` / `dj` overrides with masks, stage orders
   other than canonical, float32 inputs, masked arrays with finite data under the mask,
   `percentile_growth` NaN, non-square and large grids, and threads above 1. Each added
   scenario family should either show no difference or add a row here.
3. **Run the tool before each release** and attach the table to the release notes.
4. **BD-10 and BD-14** need decisions from the owner (fix the reference defect; audit
   CuPy), not code in this register.
