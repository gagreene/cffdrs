"""Regenerate the FBP golden regression fixtures from the *current* code.

Run this only from a known-good state of ``cffbps`` — its output becomes the
baseline refactors are checked against. (The tif goldens previously committed here
were stale, predating several behavior fixes — hence these regenerated snapshots.)

Usage:
    python tools/gen_fbp_goldens.py
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

from cffdrs.cffbps import FBP, valid_outputs  # noqa: E402


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
            name: wh.scalarize(value) for name, value in zip(outputs, result)
        }
        cases.append(record)
    return {"outputs": outputs, "cases": cases}


def build_array_snapshot() -> dict:
    outputs = list(valid_outputs)
    fbp = FBP()
    fbp.initialize(out_request=outputs, **wh.load_raster_inputs())
    result = fbp.runFBP()
    return {
        "outputs": outputs,
        "wx_date": wh.RASTER_WX_DATE,
        "dj": wh.RASTER_DJ,
        "arrays": {name: wh.arrayize(value) for name, value in zip(outputs, result)},
    }


PERCENTILE_FUNCTION_GOLDEN = os.path.join(wh.GOLDEN_DIR, "percentile_ros_function_snapshot.json")
PERCENTILE_PIPELINE_GOLDEN = os.path.join(wh.GOLDEN_DIR, "percentile_growth_snapshot.json")

# Percentiles exercised by the percentile snapshots. 0 and 100 are capped to the
# 0.001 / 99.999 bounds by the model; 50 is the exact no-op and is covered by the
# scalar snapshot.
PERCENTILES = [0, 5, 10, 25, 45, 75, 90, 95, 100]
PERCENTILE_OUTPUTS = ["hros", "bros", "cfb", "cfc", "tfc", "hfi", "fire_type"]


def build_percentile_function_snapshot() -> dict:
    """Direct outputs of ``calc_ros_percentile_growth`` over a fuel x regime x percentile x ROS grid.

    Lets a port check the pure ROS adjustment (wind decay included) without the rest of the chain.
    """
    from cffdrs.cffbps.equations.growth import _wind_decay, calc_ros_percentile_growth

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
                name: wh.scalarize(value) for name, value in zip(PERCENTILE_OUTPUTS, result)
            }
        cases.append(record)
    return {"outputs": PERCENTILE_OUTPUTS, "percentiles": PERCENTILES, "cases": cases}


def _dump(path: str, snapshot: dict) -> None:
    with open(path, "w") as fh:
        json.dump(snapshot, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main() -> None:
    os.makedirs(wh.GOLDEN_DIR, exist_ok=True)

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


if __name__ == "__main__":
    main()
