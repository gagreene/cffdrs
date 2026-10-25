"""Regenerate the FBP golden regression fixtures from the *current* code.

Run this only from a known-good state of ``cffbps`` — its output becomes the
baseline refactors are checked against. (The tif goldens previously committed here
were stale, predating several behavior fixes — hence these regenerated snapshots.)

Usage:
    python tools/gen_fbp_goldens.py
    python tools/gen_fbp_goldens.py --non-modeled-only   # only non_modeled_snapshot.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# Make the src/ package importable (for `cffdrs`) and the tests helper importable.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests", "cffbps"))

import _wotton_helpers as wh  # noqa: E402

from cffdrs.cffbps.python import FBP, valid_outputs  # noqa: E402

PERCENTILE_FUNCTION_GOLDEN = os.path.join(wh.GOLDEN_DIR, "percentile_ros_function_snapshot.json")
PERCENTILE_PIPELINE_GOLDEN = os.path.join(wh.GOLDEN_DIR, "percentile_growth_snapshot.json")

# Percentiles exercised by the percentile snapshots. 0 and 100 are capped to the
# 0.001 / 99.999 bounds by the model; 50 is the exact no-op and is covered by the
# scalar snapshot.
PERCENTILES = [0, 5, 10, 25, 45, 75, 90, 95, 100]
PERCENTILE_OUTPUTS = ["hros", "bros", "cfb", "cfc", "tfc", "hfi", "fire_type"]

NON_MODELED_GOLDEN = os.path.join(wh.GOLDEN_DIR, "non_modeled_snapshot.json")

# Fuel codes that are not modeled: 19 non-fuel and 20 water (valid codes), then invalid codes
# (masked by the fuel validation, exported as code 0).
NON_MODELED_CODES = [19, 20, 0, 99, -1]
NON_MODELED_WX_DATE = 20230615
_NON_MODELED_BASE = {
    "lat": 55.0, "long": -110.0, "elevation": 500.0, "slope": 10.0, "aspect": 270.0,
    "ws": 20.0, "wd": 0.0, "ffmc": 91.0, "bui": 76.0, "pc": 50.0, "pdf": 35.0, "gfl": 0.35, "gcf": 80.0,
    "percentile_growth": 50.0,
}
# name -> changes to the base cell. ``d0``/``dj`` present = an explicit override (NaN = masked override).
# The first three are the required scenarios; the rest pin the input-dependent rules.
_NON_MODELED_SCENARIOS = {
    "normal": {},
    "nan_bui": {"bui": np.nan},
    "nan_geography": {"lat": np.nan, "long": np.nan, "elevation": np.nan},
    "nan_ffmc": {"ffmc": np.nan},
    "nan_slope": {"slope": np.nan},
    "nan_ws": {"ws": np.nan},
    "nan_long": {"long": np.nan},
    "nan_pc_gcf": {"pc": np.nan, "gcf": np.nan},
    "percentile_90": {"percentile_growth": 90.0},
    "percentile_nan": {"percentile_growth": np.nan},
    "dj_override_nan": {"dj": np.nan},
    "overrides_d0_400_dj_0_3": {"d0": 400.0, "dj": 0.3},
}


def _dump(path: str, snapshot: dict) -> None:
    with open(path, "w") as fh:
        json.dump(snapshot, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _json_number(value: float):
    return None if not np.isfinite(value) else float(value)


def build_array_snapshot() -> dict:
    outputs = list(valid_outputs)
    fbp = FBP()
    fbp.initialize(out_request=outputs, **wh.load_raster_inputs())
    result = fbp.runFBP()
    return {
        "outputs": outputs,
        "wx_date": wh.RASTER_WX_DATE,
        "dj": wh.RASTER_DJ,
        "arrays": {name: wh.arrayize(value) for name, value in zip(outputs, result, strict=True)},
    }


def build_non_modeled_snapshot() -> dict:
    """All outputs for non-modeled fuel codes, one cell per (code, scenario), as ``getParams`` returns them.

    Every cell is a 1-element array grid (the array path), so an invalid code is masked and exported as 0.
    NaN inputs and outputs are written as ``null``.
    """
    outputs = list(valid_outputs)
    scenarios = []
    for name, changes in _NON_MODELED_SCENARIOS.items():
        cell = dict(_NON_MODELED_BASE)
        cell.update(changes)
        scenarios.append({"name": name, "inputs": {k: _json_number(v) for k, v in cell.items()}})
    cases = []
    for code in NON_MODELED_CODES:
        for scenario in scenarios:
            cell = {k: (np.nan if v is None else v) for k, v in scenario["inputs"].items()}
            percentile = cell.pop("percentile_growth")
            overrides = {k: np.array([cell.pop(k)]) for k in ("d0", "dj") if k in cell}
            fbp = FBP()
            fbp.initialize(fuel_type=np.array([code], dtype=np.int8), wx_date=NON_MODELED_WX_DATE,
                           out_request=outputs, percentile_growth=percentile,
                           **{k: np.array([v], dtype=np.float64) for k, v in cell.items()}, **overrides)
            result = fbp.runFBP()
            cases.append({
                "fuel_type": code, "scenario": scenario["name"],
                "outputs": {n: wh.scalarize(v) for n, v in zip(outputs, result, strict=True)},
            })
    return {"outputs": outputs, "wx_date": NON_MODELED_WX_DATE, "scenarios": scenarios, "cases": cases}


def build_percentile_function_snapshot() -> dict:
    """Direct outputs of ``calc_ros_percentile_growth`` over a fuel x regime x percentile x ROS grid.

    Lets a port check the pure ROS adjustment (wind decay included) without the rest of the chain.
    """
    from cffdrs.cffbps.python.equations.growth import _wind_decay, calc_ros_percentile_growth

    wsv = 10.0

    def cell(value):
        return np.ma.array([float(value)], mask=False)

    rows = []
    for fuel in (1, 2, 5, 6, 8, 12, 14):
        for cfb in (0.0, 0.09999, 0.1, 0.9):
            for percentile in PERCENTILES:
                for ros in (0.5, 7.5, 40.0):
                    head, back = calc_ros_percentile_growth(
                        percentile_growth=percentile,
                        fuel_type=np.ma.array([fuel], dtype=np.int8, mask=False),
                        hros_cfb=cell(cfb), bros_cfb=cell(cfb), wsv=cell(wsv),
                        hros=cell(ros), bros=cell(ros),
                    )
                    rows.append({
                        "fuel_type": fuel, "regime_cfb": cfb, "percentile": percentile,
                        "ros": ros, "wsv": wsv,
                        "hros": float(head[0]), "bros": float(back[0]),
                    })
    wind_decay = [{"wsv": w, "k": float(_wind_decay(cell(w))[0])} for w in (0.0, 10.0, 39.999, 40.0, 60.0)]
    return {"rows": rows, "wind_decay": wind_decay}


def build_percentile_pipeline_snapshot() -> dict:
    """Full-chain outputs for each Wotton case at each non-median percentile."""
    df = wh.load_input_frame()
    cases = []
    for _, row in df.iterrows():
        record = {"id": int(row["id"]), "fuel_type_code": str(row["fuel_type_code"]), "percentiles": {}}
        for percentile in PERCENTILES:
            fbp = FBP()
            fbp.initialize(percentile_growth=percentile, **wh.row_to_kwargs(row, PERCENTILE_OUTPUTS))
            result = fbp.runFBP()
            record["percentiles"][str(percentile)] = {
                name: wh.scalarize(value) for name, value in zip(PERCENTILE_OUTPUTS, result, strict=True)
            }
        cases.append(record)
    return {"outputs": PERCENTILE_OUTPUTS, "percentiles": PERCENTILES, "cases": cases}


def build_scalar_snapshot() -> dict:
    df = wh.load_input_frame()
    outputs = list(valid_outputs)
    cases = []
    for _, row in df.iterrows():
        fbp = FBP()
        fbp.initialize(**wh.row_to_kwargs(row, outputs))
        result = fbp.runFBP()
        record = {"id": int(row["id"]), "fuel_type_code": str(row["fuel_type_code"])}
        record["outputs"] = {
            name: wh.scalarize(value) for name, value in zip(outputs, result, strict=True)
        }
        cases.append(record)
    return {"outputs": outputs, "cases": cases}


def main() -> None:
    os.makedirs(wh.GOLDEN_DIR, exist_ok=True)
    if "--non-modeled-only" in sys.argv[1:]:
        # Writes only the non-modeled snapshot and leaves the other snapshots untouched.
        write_non_modeled_snapshot()
        return

    scalar = build_scalar_snapshot()
    _dump(wh.SCALAR_GOLDEN, scalar)
    print(f"Wrote {wh.SCALAR_GOLDEN}: {len(scalar['cases'])} cases "
          f"x {len(scalar['outputs'])} outputs")

    array = build_array_snapshot()
    rows = len(next(iter(array["arrays"].values())))
    cols = len(next(iter(array["arrays"].values()))[0])
    _dump(wh.ARRAY_GOLDEN, array)
    print(f"Wrote {wh.ARRAY_GOLDEN}: {rows}x{cols} grid "
          f"x {len(array['outputs'])} outputs")

    function = build_percentile_function_snapshot()
    _dump(PERCENTILE_FUNCTION_GOLDEN, function)
    print(f"Wrote {PERCENTILE_FUNCTION_GOLDEN}: {len(function['rows'])} rows")

    pipeline = build_percentile_pipeline_snapshot()
    _dump(PERCENTILE_PIPELINE_GOLDEN, pipeline)
    print(f"Wrote {PERCENTILE_PIPELINE_GOLDEN}: {len(pipeline['cases'])} cases "
          f"x {len(pipeline['percentiles'])} percentiles x {len(pipeline['outputs'])} outputs")

    write_non_modeled_snapshot()


def write_non_modeled_snapshot() -> None:
    snapshot = build_non_modeled_snapshot()
    _dump(NON_MODELED_GOLDEN, snapshot)
    print(f"Wrote {NON_MODELED_GOLDEN}: {len(snapshot['cases'])} cases x {len(snapshot['outputs'])} outputs")


if __name__ == "__main__":
    main()
