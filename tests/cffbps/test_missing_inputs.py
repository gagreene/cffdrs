"""Missing inputs stay missing through fuel consumption.

A NaN ``bui`` or ``ffmc`` cell arrives masked. The rate of spread then depends on a
missing value, so it is NaN, and consumption follows the spread: ``sfc``, ``ffc``,
``wfc``, ``cfc`` and ``tfc`` are NaN (never a finite number left behind by masked
arithmetic), even for fuels whose ``sfc`` does not itself use that input. ``rso`` is
not a consumption output: it is NaN only where the input feeds ``sfc`` and otherwise
keeps its finite value.
"""
from __future__ import annotations

import numpy as np
import pytest

from cffdrs.cffbps.python import FBP

SCENARIO = dict(
    lat=55.0, long=-110.0, elevation=500.0, slope=10.0, aspect=270.0, ws=20.0, wd=0.0,
    ffmc=91.0, bui=76.0, pc=50.0, pdf=35.0, gfl=0.35, gcf=80.0,
)
FIELDS = ['sfc', 'ffc', 'wfc', 'rso', 'tfc']
FUELS = range(1, 19)

# Which surface outputs each missing input feeds, per fuel code.
SFC_USES = {'bui': {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18}, 'ffmc': {1, 7}}
FFC_USES = {'bui': {16, 17, 18}, 'ffmc': {7}}
WFC_USES = {'bui': {7, 16, 17, 18}, 'ffmc': set()}
# Fuels that cannot crown: their tfc is sfc alone, so it is NaN exactly when sfc is.
NON_CROWNING = {8, 9, 14, 15, 16, 17, 18}


def _run(backend_fbp, fuel, missing=None):
    """One-cell run with every input passed as an array; ``missing`` is set to NaN."""
    kwargs = {k: np.array([[v]]) for k, v in SCENARIO.items()}
    if missing is not None:
        kwargs[missing] = np.array([[np.nan]])
    fbp = backend_fbp()
    fbp.initialize(fuel_type=np.array([[fuel]], dtype=np.float64), wx_date=20230615,
                   out_request=FIELDS, **kwargs)
    out = fbp.runFBP()
    return {name: float(np.ma.asarray(arr).astype(np.float64).filled(np.nan).ravel()[0])
            for name, arr in zip(FIELDS, out, strict=True)}


def _expected_nan(fuel, missing):
    # consumption follows the spread, which is NaN whenever bui or ffmc is missing
    nan = {'sfc', 'ffc', 'wfc', 'tfc'}
    if fuel in SFC_USES[missing]:
        nan.add('rso')
    return nan


@pytest.mark.parametrize('missing', ['bui', 'ffmc'])
@pytest.mark.parametrize('fuel', FUELS)
def test_missing_input_propagates_to_dependent_surface_outputs(backend_fbp, fuel, missing):
    base = _run(backend_fbp, fuel)
    got = _run(backend_fbp, fuel, missing)
    nan = _expected_nan(fuel, missing)
    for name in sorted(nan):
        assert np.isnan(got[name]), f'fuel {fuel}, NaN {missing}: {name} = {got[name]!r}, expected NaN'
    # rso does not follow the spread: it keeps its finite value unless the input feeds sfc.
    for name in sorted({'rso'} - nan):
        assert np.array_equal(got[name], base[name], equal_nan=True), (
            f'fuel {fuel}, NaN {missing}: {name} = {got[name]!r}, expected {base[name]!r}')


def test_consumption_is_missing_when_the_spread_is_missing(backend_fbp):
    # sfc does not use ffmc for C-2, bui for C-1 or bui for O-1a, but their spread does
    assert np.isnan(_run(backend_fbp, 2, 'ffmc')['sfc'])
    assert np.isnan(_run(backend_fbp, 1, 'bui')['sfc'])
    assert np.isnan(_run(backend_fbp, 14, 'bui')['sfc'])
    assert np.isfinite(_run(backend_fbp, 2, 'ffmc')['rso'])


def test_finite_baseline_is_unchanged(backend_fbp):
    assert _run(backend_fbp, 2)['sfc'] == pytest.approx(2.9136045, abs=1e-7)


# ── Python/Rust parity on the same scenario, one cell per modeled fuel ─────────
GRID_FIELDS = ['hros', 'bros', 'raz', 'wsv', 'hfi', 'rso', 'sros', 'sfc', 'fmc', 'accel']
CELL_INPUTS = ['lat', 'long', 'elevation', 'slope', 'aspect', 'pc', 'gcf', 'ws', 'wd']


def _parity(missing, percentile):
    """Run every modeled fuel once with ``missing`` NaN through both implementations."""
    from cffdrs import _rust as rust_backend

    fuel = np.arange(1, 19, dtype=np.int32).reshape(1, 18)
    values = dict(SCENARIO)
    if missing is not None:
        values[missing] = np.nan
    cells = {k: np.full(fuel.shape, values[k]) for k in CELL_INPUTS}
    fbp = FBP()
    fbp.initialize(fuel_type=fuel.astype(np.float64), wx_date=20230615,
                   ffmc=values['ffmc'], bui=values['bui'], pdf=values['pdf'], gfl=values['gfl'],
                   out_request=GRID_FIELDS, percentile_growth=percentile, **cells)
    py = {n: np.ma.asarray(a).astype(np.float64).filled(np.nan)
          for n, a in zip(GRID_FIELDS, fbp.runFBP(), strict=True)}
    rs = rust_backend.run_fbp_grid(
        fuel, cells['lat'], cells['long'], cells['elevation'], cells['slope'], cells['aspect'],
        cells['pc'], cells['gcf'], cells['ws'], cells['wd'], 20230615, values['ffmc'], values['bui'],
        values['pdf'], values['gfl'], float(percentile))
    for name in GRID_FIELDS:
        a, b = py[name].ravel(), np.asarray(rs[name]).ravel()
        same = (np.isnan(a) & np.isnan(b)) | np.isclose(b, a, rtol=1e-9, atol=1e-12)
        assert same.all(), (
            f'NaN {missing}, percentile {percentile}, {name}: fuels {(np.flatnonzero(~same) + 1).tolist()} '
            f'py={a[~same].tolist()} rs={b[~same].tolist()}')


@pytest.mark.parametrize('percentile', [50, 90])
@pytest.mark.parametrize('missing', ['bui', 'ffmc', 'lat'])
def test_missing_input_matches_rust(missing, percentile):
    _parity(missing, percentile)


@pytest.mark.parametrize('missing', ['gfl', 'pdf'])
def test_nan_scalar_default_matches_rust(missing):
    """A NaN scalar gfl/pdf takes Python's default (0.35 / 35) in both implementations."""
    _parity(missing, 50)
