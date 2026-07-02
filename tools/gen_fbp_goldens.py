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


if __name__ == "__main__":
    main()
