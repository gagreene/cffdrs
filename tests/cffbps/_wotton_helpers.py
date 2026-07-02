"""Shared helpers for the Wotton (2009) FBP regression fixtures.

Used by both the golden-snapshot generator (``tools/gen_fbp_goldens.py``) and the
pytest regression tests so the input-construction logic lives in exactly one place.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INPUT_CSV = os.path.join(DATA_DIR, "Inputs_for_Test_Cases_Wotton2009.csv")
RASTER_INPUT_DIR = os.path.join(DATA_DIR, "inputs")
GOLDEN_DIR = os.path.join(DATA_DIR, "golden")
SCALAR_GOLDEN = os.path.join(GOLDEN_DIR, "wotton2009_scalar_snapshot.json")
ARRAY_GOLDEN = os.path.join(GOLDEN_DIR, "raster_array_snapshot.json")

# initialize() defaults, applied when the CSV cell is blank/NA.
_DEFAULTS = {"pc": 50, "pdf": 35, "gfl": 0.35, "gcf": 80}

# Raster fixture: wx_date is scalar (no raster); Dj.tif == day-of-year of this date.
RASTER_WX_DATE = 20160516
RASTER_DJ = 137

# Maps FBP.initialize array arg -> input raster filename.
_RASTER_FILES = {
    "lat": "LAT.tif", "long": "LONG.tif", "elevation": "ELV.tif",
    "slope": "GS.tif", "aspect": "Aspect.tif", "ws": "WS.tif",
    "wd": "WD.tif", "ffmc": "FFMC.tif", "bui": "BUI.tif",
    "pc": "PC.tif", "pdf": "PDF.tif", "gfl": "GFL.tif", "gcf": "cc.tif",
}


def load_input_frame() -> pd.DataFrame:
    """Load the Wotton 2009 input CSV with NA handling."""
    return pd.read_csv(INPUT_CSV, na_values=["NA", ""])


def _cell(row: pd.Series, col: str, default=None):
    """Return a plain Python value for a CSV cell, or ``default`` if NA."""
    val = row[col]
    if pd.isna(val):
        return default
    return val


def row_to_kwargs(row: pd.Series, out_request: list[str]) -> dict:
    """Convert one CSV row into keyword args for ``FBP.initialize``.

    Blank optional fields fall back to ``initialize`` defaults; ``d0``/``dj`` stay
    ``None`` when absent so foliar-moisture logic derives them.
    """
    d0 = _cell(row, "d0")
    dj = _cell(row, "dj")
    return dict(
        fuel_type=int(row["fuel_type"]),
        wx_date=int(row["wx_date"]),
        lat=float(row["lat"]),
        long=float(row["long"]),
        elevation=float(row["elevation"]),
        slope=float(row["slope"]),
        aspect=float(row["aspect"]),
        ws=float(row["ws"]),
        wd=float(row["wd"]),
        ffmc=float(row["ffmc"]),
        bui=float(row["bui"]),
        pc=float(_cell(row, "pc", _DEFAULTS["pc"])),
        pdf=float(_cell(row, "pdf", _DEFAULTS["pdf"])),
        gfl=float(_cell(row, "gfl", _DEFAULTS["gfl"])),
        gcf=float(_cell(row, "gcf", _DEFAULTS["gcf"])),
        d0=None if d0 is None else int(d0),
        dj=None if dj is None else int(dj),
        out_request=list(out_request),
    )


def load_raster_inputs() -> dict:
    """Read the input raster fixtures into a kwargs dict for ``FBP.initialize``.

    All spatial fields load as float64 arrays (fuel_type as int8) to exercise the
    masked-array/array code path; ``wx_date``/``dj`` are the scalar fixture values.
    Requires ``rasterio`` (a test-only dependency).
    """
    import rasterio as rio

    def read(name, dtype):
        with rio.open(os.path.join(RASTER_INPUT_DIR, name)) as src:
            return src.read(1).astype(dtype)

    kwargs = {"fuel_type": read("FuelType.tif", np.int8)}
    for arg, fname in _RASTER_FILES.items():
        kwargs[arg] = read(fname, np.float64)
    kwargs["wx_date"] = RASTER_WX_DATE
    kwargs["dj"] = RASTER_DJ
    return kwargs


def arrayize(value) -> list:
    """Normalize an array-path ``getParams`` element to nested lists for JSON.

    Masked/NaN cells become ``None`` so the golden round-trips exactly.
    """
    arr = np.ma.filled(np.ma.asarray(value), np.nan)
    out = arr.astype(object)
    if np.issubdtype(arr.dtype, np.floating):
        out[~np.isfinite(arr)] = None
    return [[(None if v is None else (v.item() if hasattr(v, "item") else v))
             for v in row] for row in np.atleast_2d(out)]


def scalarize(value) -> object:
    """Normalize an ``FBP.getParams`` return element to a JSON-friendly scalar.

    Scalar-path outputs come back as 1-element (possibly masked) arrays; collapse
    them to a Python float/int, mapping masked/NaN cells to ``None``.
    """
    arr = np.ma.asarray(value).ravel()
    if arr.size == 0:
        return None
    item = arr[0]
    if item is np.ma.masked:
        return None
    scalar = item.item() if hasattr(item, "item") else item
    if isinstance(scalar, float) and not np.isfinite(scalar):
        return None
    return scalar
