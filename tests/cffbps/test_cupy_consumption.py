"""The experimental CuPy backend follows the spread when it consumes fuel.

CuPy and a GPU are not needed: a NumPy-backed stand-in is installed as ``cupy`` while the
backend module is imported, which is enough to exercise ``calcTFC``.
"""
import importlib
import sys
import types

import numpy as np
import pytest


def _drop_cupy_backend_modules():
    for name in [m for m in sys.modules if m.startswith('cffdrs.cffbps.cupy')]:
        sys.modules.pop(name, None)


@pytest.fixture
def cupy_core(monkeypatch):
    stand_in = types.ModuleType('cupy')
    stand_in.__dict__.update({k: getattr(np, k) for k in dir(np) if not k.startswith('__')})
    stand_in.asnumpy = np.asarray
    stand_in.cuda = types.SimpleNamespace(Device=lambda *a, **k: None, runtime=types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'cupy', stand_in)
    _drop_cupy_backend_modules()
    yield importlib.import_module('cffdrs.cffbps.cupy.core')
    _drop_cupy_backend_modules()


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_calc_tfc_consumes_no_fuel_without_spread(cupy_core, dtype):
    fbp = object.__new__(cupy_core.FBP)
    fbp.hfros = np.array([0.0, np.nan, 2.0], dtype=dtype)
    fbp.ffc = np.array([1.0, 1.0, 1.0], dtype=dtype)
    fbp.wfc = np.array([np.nan, 1.0, 1.0], dtype=dtype)
    fbp.sfc = np.array([2.0, 2.0, 2.0], dtype=dtype)
    fbp.cfc = np.array([0.5, 0.5, 0.5], dtype=dtype)
    fbp.calcTFC()
    np.testing.assert_array_equal(fbp.ffc, [0, np.nan, 1])
    np.testing.assert_array_equal(fbp.wfc, [0, np.nan, 1])
    np.testing.assert_array_equal(fbp.sfc, [0, np.nan, 2])
    np.testing.assert_array_equal(fbp.cfc, [0, np.nan, 0.5])
    np.testing.assert_array_equal(fbp.tfc, [0, np.nan, 2.5])
    assert fbp.tfc.dtype == dtype
