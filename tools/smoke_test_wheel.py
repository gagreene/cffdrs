"""Smoke-test an *installed* cffdrs wheel.

Run this with the interpreter of a clean virtual environment into which only the
built wheel was installed, from a directory outside the repository checkout:

    python tools/smoke_test_wheel.py

It checks that the package imported from site-packages (not the source tree),
that its metadata declares the ``cp311-abi3`` tag, that the compiled
``cffdrs._rust`` extension loads, and that the compiled grid
pass agrees with the pure-Python FBP on a small synthetic grid, including a
percentile-growth run, a Fortran-ordered input, and rejection of an invalid date.
"""
from __future__ import annotations

import sys
from importlib.metadata import distribution

import numpy as np

import cffdrs
from cffdrs import _rust
from cffdrs.cffbps import FBP as AutoFBP
from cffdrs.cffbps.common.constants import valid_outputs
from cffdrs.cffbps.python import FBP as PythonFBP
from cffdrs.cffbps.rust import FBP as RustFBP

FIELDS = ["hros", "bros", "raz", "wsv", "hfi", "rso", "sros", "sfc", "fmc", "accel"]
SCALARS = dict(wx_date=20230615, ffmc=91.2, bui=76.4, pdf=42.0, gfl=0.41)


def assert_close(a, b, label):
    for name in FIELDS:
        if not np.allclose(a[name], b[name], rtol=1e-9, atol=1e-12, equal_nan=True):
            raise AssertionError(f"{label}: {name} differs between Rust and Python")


def check_public_facades():
    # One array cell checks all requested fields, including integer output parity.
    kwargs = dict(fuel_type=np.array([2], dtype=np.int8), wx_date=20230615,
                  lat=52., long=-115., elevation=800., slope=10., aspect=180.,
                  ws=20., wd=270., ffmc=90., bui=80., out_request=list(valid_outputs))
    reference = PythonFBP()
    reference.initialize(**kwargs)
    expected = reference.runFBP()
    for cls in (RustFBP, AutoFBP):
        model = cls()
        model.initialize(**kwargs)
        if isinstance(model, AutoFBP):
            assert model.backend_in_use == 'rust', model.backend_in_use
        for name, actual, wanted in zip(valid_outputs, model.runFBP(), expected, strict=True):
            if np.asarray(wanted).dtype.kind in 'iu':
                np.testing.assert_array_equal(actual, wanted, err_msg=name)
            else:
                np.testing.assert_allclose(actual, wanted, rtol=1e-9, atol=1e-12,
                                           equal_nan=True, err_msg=name)
    print('public Rust and default AutoFBP: all 54 outputs match Python; auto uses rust')


def main() -> None:
    wheel_metadata = distribution("cffdrs-fire").read_text("WHEEL")
    assert wheel_metadata is not None and any(
        line.startswith("Tag: cp311-abi3-") for line in wheel_metadata.splitlines()
    ), f"installed wheel does not contain a cp311-abi3 tag: {wheel_metadata}"
    location = cffdrs.__file__.replace("\\", "/")
    if "site-packages" not in location:
        raise SystemExit(f"cffdrs was imported from {location}, not an installed wheel")
    print(f"cffdrs {getattr(cffdrs, '__version__', '?')} from {location}")
    print(f"cffdrs._rust core {_rust.__core_version__} from {_rust.__file__}")

    check_public_facades()
    g = make_grid()
    for percentile in (50, 90, 10):
        assert_close(run_python(g, percentile), run_rust(g, percentile), f"percentile {percentile}")

    # memory layout must not change results
    fortran = {k: (np.asfortranarray(v) if v.ndim == 2 else v) for k, v in g.items()}
    assert_close(run_rust(g, 50), run_rust(fortran, 50), "fortran-ordered inputs")

    # an impossible calendar date is rejected, as in Python
    try:
        run_rust(g, 50, wx_date=20230230)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid wx_date was accepted")

    print("wheel smoke test passed")


def make_grid():
    rng = np.random.default_rng(7)
    shape = (3, 6)  # fuel types 1-18, one per cell
    g = {
        "fuel_type": np.arange(1, 19, dtype=np.int32).reshape(shape),
        "lat": rng.uniform(48, 60, shape),
        "long": rng.uniform(-120, -95, shape),
        "elevation": rng.uniform(0, 1500, shape),
        "slope": rng.uniform(0, 40, shape),
        "aspect": rng.uniform(0, 360, shape),
        "ws": rng.uniform(1, 40, shape),
        "wd": rng.uniform(0, 360, shape),
        "pc": rng.uniform(0, 100, shape),
        "gcf": rng.uniform(10, 100, shape),
    }
    return g


def run_python(g, percentile):
    fbp = PythonFBP()
    fbp.initialize(
        fuel_type=g["fuel_type"].astype(np.float64), wx_date=SCALARS["wx_date"],
        lat=g["lat"], long=g["long"], elevation=g["elevation"], slope=g["slope"], aspect=g["aspect"],
        ws=g["ws"], wd=g["wd"], ffmc=SCALARS["ffmc"], bui=SCALARS["bui"], pc=g["pc"],
        pdf=SCALARS["pdf"], gfl=SCALARS["gfl"], gcf=g["gcf"],
        out_request=FIELDS, percentile_growth=percentile,
    )
    return {n: np.ma.asarray(a).astype(np.float64).filled(np.nan) for n, a in zip(FIELDS, fbp.runFBP(), strict=True)}


def run_rust(g, percentile, wx_date=None):
    res = _rust.run_fbp_grid(
        g["fuel_type"], g["lat"], g["long"], g["elevation"], g["slope"], g["aspect"],
        g["pc"], g["gcf"], g["ws"], g["wd"],
        SCALARS["wx_date"] if wx_date is None else wx_date,
        SCALARS["ffmc"], SCALARS["bui"], SCALARS["pdf"], SCALARS["gfl"], float(percentile),
    )
    return {n: np.asarray(res[n]) for n in FIELDS}


if __name__ == "__main__":
    sys.exit(main())
