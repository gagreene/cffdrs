"""Missing inputs stay missing through surface fuel consumption.

A NaN ``bui`` or ``ffmc`` cell arrives masked. Every surface output that depends
on it (``sfc``, the fine/woody split ``ffc``/``wfc`` where a fuel defines one,
and the ``rso``/``tfc`` derived from ``sfc``) must come out NaN, never a finite
number left behind by masked arithmetic. Outputs that do not depend on the
missing input keep their finite values.
"""
from __future__ import annotations

import numpy as np
import pytest

from cffdrs.cffbps import FBP

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


def _run(fuel, missing=None):
    """One-cell run with every input passed as an array; ``missing`` is set to NaN."""
    kwargs = {k: np.array([[v]]) for k, v in SCENARIO.items()}
    if missing is not None:
        kwargs[missing] = np.array([[np.nan]])
    fbp = FBP()
    fbp.initialize(fuel_type=np.array([[fuel]], dtype=np.float64), wx_date=20230615,
                   out_request=FIELDS, **kwargs)
    out = fbp.runFBP()
    return {name: float(np.ma.asarray(arr).astype(np.float64).filled(np.nan).ravel()[0])
            for name, arr in zip(FIELDS, out, strict=True)}


def _expected_nan(fuel, missing):
    nan = set()
    if fuel in SFC_USES[missing]:
        nan |= {'sfc', 'rso', 'tfc'}
    if fuel in FFC_USES[missing]:
        nan.add('ffc')
    if fuel in WFC_USES[missing]:
        nan.add('wfc')
    return nan


@pytest.mark.parametrize('missing', ['bui', 'ffmc'])
@pytest.mark.parametrize('fuel', FUELS)
def test_missing_input_propagates_to_dependent_surface_outputs(fuel, missing):
    base = _run(fuel)
    got = _run(fuel, missing)
    nan = _expected_nan(fuel, missing)
    for name in sorted(nan):
        assert np.isnan(got[name]), f'fuel {fuel}, NaN {missing}: {name} = {got[name]!r}, expected NaN'
    # Outputs that do not depend on the missing input keep their finite values. tfc also
    # carries crown consumption (NaN through the spread chain), so it is compared only
    # for fuels that cannot crown.
    kept = {'sfc', 'ffc', 'wfc', 'rso'} | ({'tfc'} if fuel in NON_CROWNING else set())
    for name in sorted(kept - nan):
        assert np.array_equal(got[name], base[name], equal_nan=True), (
            f'fuel {fuel}, NaN {missing}: {name} = {got[name]!r}, expected {base[name]!r}')


def test_independent_fuels_keep_their_finite_values():
    assert _run(2, 'ffmc')['sfc'] == pytest.approx(2.9136045, abs=1e-7)
    assert _run(1, 'bui')['sfc'] == pytest.approx(1.421, abs=1e-3)
    assert _run(14, 'bui')['sfc'] == pytest.approx(0.35, abs=1e-12)


def test_finite_baseline_is_unchanged():
    assert _run(2)['sfc'] == pytest.approx(2.9136045, abs=1e-7)
