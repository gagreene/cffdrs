"""Differential parity: the compiled grid pass vs the Python implementation.

The Python package is the spec. These tests run both implementations over the
same synthetic grids (all modeled fuel types, slopes, aspects, calm and windy
cells, non-fuel cells) and require agreement at rtol 1e-9 on every consumed
output. Complements the static Wotton goldens (which pin the scalar chain) by
exercising the array path end to end.

The extension is part of the main package and is built by ``uv sync``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cffdrs import _rust as rust_backend
from cffdrs.cffbps import FBP

RNG = np.random.default_rng(42)
SHAPE = (12, 18)  # every modeled fuel type appears more than once

# outputs consumed by fire-growth engines, in the order we request them
FIELDS = ['hros', 'bros', 'raz', 'wsv', 'hfi', 'rso', 'sros', 'sfc', 'fmc', 'accel']


@pytest.fixture(scope='module')
def grids():
    n = SHAPE[0] * SHAPE[1]
    fuel = np.tile(np.arange(1, 19, dtype=np.int32), n // 18 + 1)[:n].reshape(SHAPE)
    # sprinkle non-fuel and water cells
    fuel = fuel.copy()
    fuel[0, :3] = 19
    fuel[1, :2] = 20
    g = {
        'fuel_type': fuel,
        'lat': RNG.uniform(48, 60, SHAPE),
        'long': RNG.uniform(-120, -95, SHAPE),
        'elevation': RNG.uniform(0, 1500, SHAPE),
        'slope': RNG.uniform(0, 60, SHAPE),
        'aspect': RNG.uniform(0, 360, SHAPE),
        'ws': RNG.uniform(0, 45, SHAPE),
        'wd': RNG.uniform(0, 360, SHAPE),
        'pc': RNG.uniform(0, 100, SHAPE),
        'gcf': RNG.uniform(0, 100, SHAPE),
    }
    g['ws'][2, :4] = 0.0        # calm-wind cells (zero-WSV path)
    g['slope'][2, :4] = 0.0
    g['gcf'][3, 0] = 0.0        # gcf==0 -> 0.1 clamp
    # nodata cells: NaN inputs arrive masked in the Python package and must
    # surface as NaN behaviour, never be laundered into calm/flat/default
    g['ws'][4, 0] = np.nan
    g['slope'][4, 1] = np.nan
    g['aspect'][4, 2] = np.nan
    g['pc'][4, 3] = np.nan
    g['gcf'][4, 4] = np.nan
    return g


SCALARS = dict(wx_date=20230615, ffmc=91.2, bui=76.4, pdf=42.0, gfl=0.41)


def python_reference(g, percentile=50):
    fbp = FBP()
    fbp.initialize(
        fuel_type=g['fuel_type'].astype(np.float64),
        wx_date=SCALARS['wx_date'],
        lat=g['lat'], long=g['long'], elevation=g['elevation'],
        slope=g['slope'], aspect=g['aspect'],
        ws=g['ws'], wd=g['wd'],
        ffmc=SCALARS['ffmc'], bui=SCALARS['bui'],
        pc=g['pc'], pdf=SCALARS['pdf'], gfl=SCALARS['gfl'], gcf=g['gcf'],
        out_request=FIELDS,
        percentile_growth=percentile,
    )
    result = fbp.runFBP()
    out = {}
    for name, arr in zip(FIELDS, result):
        a = np.ma.asarray(arr).astype(np.float64)
        out[name] = a.filled(np.nan)
    return out


def rust_grid(g, percentile=50.0):
    res = rust_backend.run_fbp_grid(
        g['fuel_type'],
        g['lat'], g['long'], g['elevation'],
        g['slope'], g['aspect'],
        g['pc'], g['gcf'],
        g['ws'], g['wd'],
        SCALARS['wx_date'], SCALARS['ffmc'], SCALARS['bui'],
        SCALARS['pdf'], SCALARS['gfl'], percentile,
    )
    return {name: np.asarray(res[name]) for name in FIELDS}


def assert_grid_parity(grids, percentile):
    py = python_reference(grids, percentile)
    rs = rust_grid(grids, float(percentile))
    modeled = (grids['fuel_type'] >= 1) & (grids['fuel_type'] <= 18)
    for name in FIELDS:
        a, b = py[name], rs[name]
        assert a.shape == b.shape == SHAPE, name
        pa, pb = a[modeled], b[modeled]
        both_nan = np.isnan(pa) & np.isnan(pb)
        close = np.isclose(pb, pa, rtol=1e-9, atol=1e-12)
        bad = ~(both_nan | close)
        assert not bad.any(), (
            f'{name}: {int(bad.sum())} modeled cells differ; '
            f'first: py={pa[bad][0]!r} rs={pb[bad][0]!r}'
        )


def test_grid_parity_all_fields(grids):
    assert_grid_parity(grids, 50)


# 0 and 100 are capped to 0.001 / 99.999 by both implementations; -5 and 150
# exercise the cap from outside the 0-100 range.
@pytest.mark.parametrize('percentile', [5, 10, 25, 45, 75, 90, 95, 99, 0, 100, -5, 150])
def test_grid_parity_percentile_growth(grids, percentile):
    assert_grid_parity(grids, percentile)


def test_percentile_growth_changes_results(grids):
    """The percentile actually reaches the compiled pass: 90 differs from the median on adjusted fuels."""
    median = rust_grid(grids, 50.0)['hros']
    p90 = rust_grid(grids, 90.0)['hros']
    conifer = (grids['fuel_type'] >= 1) & (grids['fuel_type'] <= 7)
    assert (p90[conifer] > median[conifer]).any()


def test_nan_percentile_matches_python(grids):
    """NaN is not capped: both implementations propagate it identically."""
    assert_grid_parity(grids, float('nan'))


def test_non_fuel_cells_are_nan(grids):
    rs = rust_grid(grids)
    non_fuel = ~((grids['fuel_type'] >= 1) & (grids['fuel_type'] <= 18))
    assert non_fuel.sum() >= 5
    for name in FIELDS:
        assert np.isnan(rs[name][non_fuel]).all(), name
