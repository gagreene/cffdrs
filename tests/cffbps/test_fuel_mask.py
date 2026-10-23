"""The mask of the fuel type grid is applied to every dataset (decided 2026-10-16).

Contract: a cell whose fuel is masked or invalid (not 19/20, which are valid codes) behaves exactly like a cell with an
invalid fuel and every other input missing, as both backends export it; the only deliberate change is that
``fuel_type`` exports 0 there.
"""
import importlib

import numpy as np
import pytest

from cffdrs.cffbps.python import FBP as PythonFBP
from cffdrs.cffbps.python import valid_outputs

ALL = list(valid_outputs)
N = 3
INPUTS = ('lat', 'long', 'elevation', 'slope', 'aspect', 'ws', 'wd', 'ffmc', 'bui', 'pc', 'pdf', 'gfl', 'gcf')
BASE = dict(wx_date=20230615, lat=52., long=-115., elevation=800., slope=10., aspect=180., ws=20., wd=270., ffmc=90.,
            bui=80., pc=50., pdf=35., gfl=0.35, gcf=80.)


def model(backend, fuel, missing_cells=(), **changes):
    """A grid (or scalar) model; ``missing_cells`` have every input except the fuel set to NaN."""
    values = {k: np.full(N, v) for k, v in BASE.items() if k != 'wx_date'} | {'wx_date': BASE['wx_date']}
    for name in INPUTS:
        values[name] = values[name].copy()
        values[name][list(missing_cells)] = np.nan
    cls = importlib.import_module('cffdrs.cffbps.rust').FBP if backend == 'rust' else PythonFBP
    fbp = cls()
    fbp.initialize(fuel_type=fuel, out_request=ALL, **(values | changes))
    return fbp


def cell(results, i):
    flat = [np.asarray(r).ravel() for r in results]
    return {n: (r[i] if r.size > 1 else r[0]) for n, r in zip(ALL, flat, strict=True)}


def same(a, b, what=''):
    for name in ALL:
        x, y = a[name], b[name]
        assert (np.isnan(x) and np.isnan(y)) or x == y, (what, name, x, y)


BACKENDS = ['python', 'rust']


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fuel_code', [2, 6, 10, 14])
def test_a_fuel_masked_cell_behaves_like_invalid_fuel_with_every_other_input_missing(backend, fuel_code):
    fuel = np.ma.array([fuel_code, fuel_code, 6], mask=[True, False, False])
    masked = cell(model(backend, fuel).runFBP(), 0)
    all_missing = cell(model(backend, fuel, missing_cells=[0]).runFBP(), 0)
    same(masked, all_missing, 'contract')
    assert masked['fuel_type'] == 0


@pytest.mark.parametrize('backend', BACKENDS)
def test_other_cells_are_untouched(backend):
    fuel = np.ma.array([2, 2, 6], mask=[True, False, False])
    with_mask = model(backend, fuel).runFBP()
    without = model(backend, np.array([2, 2, 6])).runFBP()
    for i in (1, 2):
        same(cell(with_mask, i), cell(without, i), f'cell {i}')


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('code', [0, 21, 99, -1, 7.5])
def test_an_invalid_code_is_the_same_as_a_masked_fuel(backend, code):
    invalid = cell(model(backend, np.array([code, 2, 6], dtype=float)).runFBP(), 0)
    masked = cell(model(backend, np.ma.array([2, 2, 6], mask=[True, False, False])).runFBP(), 0)
    same(invalid, masked, 'invalid vs masked')
    assert invalid['fuel_type'] == 0


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('code', [19, 20])
def test_non_fuel_and_water_are_valid_codes_and_stay_unmasked(backend, code):
    out = cell(model(backend, np.array([code, 2, 6])).runFBP(), 0)
    assert np.isfinite(out['fmc']) and np.isfinite(out['ws']) and out['fuel_type'] == code


@pytest.mark.parametrize('backend', BACKENDS)
def test_fuel_independent_outputs_are_missing_in_a_masked_cell(backend):
    out = cell(model(backend, np.ma.array([2, 2, 6], mask=[True, False, False])).runFBP(), 0)
    for name in ('fmc', 'fme', 'latn', 'nd', 'ws', 'ffmc', 'bui', 'm', 'sf'):
        assert np.isnan(out[name]), name


@pytest.mark.parametrize('backend', BACKENDS)
def test_replacing_the_fuel_grid_masks_and_unmasks_the_right_cells(backend):
    fbp = model(backend, np.array([2, 2, 6]))
    fbp.setParams({'fuel_type': np.ma.array([2, 2, 6], mask=[False, True, False])})
    out = fbp.runFBP()
    reference = model(backend, np.ma.array([2, 2, 6], mask=[False, True, False])).runFBP()
    for i in range(N):
        same(cell(out, i), cell(reference, i), f'masked cell 1, cell {i}')
    fbp.setParams({'fuel_type': np.array([2, 2, 6])})
    out = fbp.runFBP()
    plain = model(backend, np.array([2, 2, 6])).runFBP()
    for i in range(N):
        same(cell(out, i), cell(plain, i), f'unmasked again, cell {i}')


@pytest.mark.parametrize('backend', BACKENDS)
def test_before_any_stage_runs_computed_fields_are_the_templates_even_in_a_masked_cell(backend):
    computed = ['fmc', 'fme', 'sfc', 'hros', 'a', 'b', 'c', 'isi', 'cbh', 'fire_type', 'fi_class', 'tfc', 'cfb']
    masked = model(backend, np.ma.array([2, 2, 6], mask=[True, False, False]))
    plain = model(backend, np.array([2, 2, 6]))
    for name in computed:
        x, y = np.ravel(masked.getParams([name])[0]), np.ravel(plain.getParams([name])[0])
        x, y = x[0], y[0]
        assert (np.isnan(x) and np.isnan(y)) or x == y, name


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fuel', [99, 0])
def test_a_scalar_with_an_invalid_or_masked_fuel_follows_the_contract(backend, fuel):
    scalar = {k: v for k, v in BASE.items()}
    cls = importlib.import_module('cffdrs.cffbps.rust').FBP if backend == 'rust' else PythonFBP

    def run(**changes):
        fbp = cls()
        fbp.initialize(fuel_type=fuel, out_request=ALL, **(scalar | changes))
        return dict(zip(ALL, fbp.runFBP(), strict=True))

    got = run()
    missing = run(**{k: np.nan for k in INPUTS})
    for name in ALL:
        assert (np.isnan(got[name]) and np.isnan(missing[name])) or got[name] == missing[name], name


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('route', ['setParams', 'set_override', 'calcFMC'])
def test_updates_after_initialize_keep_the_fuel_mask(backend, route):
    fuel = np.ma.array([2, 2, 6], mask=[True, False, False])
    fbp = model(backend, fuel)
    if route == 'setParams':
        fbp.setParams({'ws': np.full(N, 25.), 'lat': np.full(N, 50.), 'wd': np.full(N, 90.), 'dj': np.full(N, 170.)})
    elif route == 'set_override':
        fbp.set_override('d0', np.full(N, 150.))
        fbp.set_override('fmc', np.full(N, 95.))
    else:
        fbp.calcFMC(lat=np.full(N, 50.), d0=np.full(N, 150.))
    out = fbp.runFBP()
    reference = model(backend, fuel, missing_cells=[0])
    if route == 'setParams':
        reference.setParams({'ws': np.full(N, 25.), 'lat': np.full(N, 50.), 'wd': np.full(N, 90.),
                             'dj': np.full(N, 170.)})
        reference.setParams({'ws': np.r_[np.nan, np.full(2, 25.)], 'lat': np.r_[np.nan, np.full(2, 50.)],
                             'wd': np.r_[np.nan, np.full(2, 90.)], 'dj': np.r_[np.nan, np.full(2, 170.)]})
    elif route == 'set_override':
        reference.set_override('d0', np.r_[np.nan, np.full(2, 150.)])
        reference.set_override('fmc', np.r_[np.nan, np.full(2, 95.)])
    else:
        reference.calcFMC(lat=np.r_[np.nan, np.full(2, 50.)], d0=np.r_[np.nan, np.full(2, 150.)])
    same(cell(out, 0), cell(reference.runFBP(), 0), route)
    assert np.isnan(cell(out, 0)['ws']) or cell(out, 0)['ws'] == 0


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_a_nan_fuel_cell_is_a_masked_fuel_cell(backend, dtype):
    nan_fuel = cell(model(backend, np.array([2, np.nan, 6], dtype=dtype)).runFBP(), 1)
    masked = cell(model(backend, np.ma.array([2, 2, 6], mask=[False, True, False])).runFBP(), 1)
    same(nan_fuel, masked, 'NaN fuel')
    assert nan_fuel['a'] == 0 and nan_fuel['cbh'] == 0 and np.isnan(nan_fuel['fmc'])


def test_python_and_rust_agree_on_a_nan_fuel_cell():
    fuel = np.array([2., np.nan, 6.])
    py, rs = (cell(model(b, fuel).runFBP(), 1) for b in BACKENDS)
    same(py, rs, 'python vs rust')
