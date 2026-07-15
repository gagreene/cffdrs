"""Differential parity: the cffdrs-rs compiled grid pass vs the Python package.

The Python package is the spec. These tests run both implementations over the
same synthetic grids (all modeled fuel types, slopes, aspects, calm and windy
cells, non-fuel cells) and require agreement at rtol 1e-9 on every consumed
output. Complements the static Wotton goldens (which pin the scalar chain) by
exercising the array path end to end.

Skipped unless the extension is built:
    PYO3_PYTHON=$(pwd)/.venv/bin/python \
      uv run maturin develop -m rust/crates/cffdrs-py/Cargo.toml
"""

from __future__ import annotations

import numpy as np
import pytest

cffdrs_rs = pytest.importorskip('cffdrs_rs')

from cffdrs.cffbps import FBP  # noqa: E402

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
    return g


SCALARS = dict(wx_date=20230615, ffmc=91.2, bui=76.4, pdf=42.0, gfl=0.41)


def python_reference(g):
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
    )
    result = fbp.runFBP()
    out = {}
    for name, arr in zip(FIELDS, result):
        a = np.ma.asarray(arr).astype(np.float64)
        out[name] = a.filled(np.nan)
    return out


def rust_grid(g):
    res = cffdrs_rs.run_fbp_grid(
        g['fuel_type'],
        g['lat'], g['long'], g['elevation'],
        g['slope'], g['aspect'],
        g['pc'], g['gcf'],
        g['ws'], g['wd'],
        SCALARS['wx_date'], SCALARS['ffmc'], SCALARS['bui'],
        SCALARS['pdf'], SCALARS['gfl'], 50.0,
    )
    return {name: np.asarray(res[name]) for name in FIELDS}


def test_grid_parity_all_fields(grids):
    py = python_reference(grids)
    rs = rust_grid(grids)
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


def test_non_fuel_cells_are_nan(grids):
    rs = rust_grid(grids)
    non_fuel = ~((grids['fuel_type'] >= 1) & (grids['fuel_type'] <= 18))
    assert non_fuel.sum() >= 5
    for name in FIELDS:
        assert np.isnan(rs[name][non_fuel]).all(), name


def test_percentile_growth_not_yet_supported():
    with pytest.raises(ValueError):
        cffdrs_rs.run_fbp_grid(
            np.full((2, 2), 2, dtype=np.int32),
            np.full((2, 2), 55.0), np.full((2, 2), -110.0), np.zeros((2, 2)),
            np.zeros((2, 2)), np.full((2, 2), 270.0),
            np.full((2, 2), 50.0), np.full((2, 2), 80.0),
            np.full((2, 2), 10.0), np.zeros((2, 2)),
            20230615, 90.0, 80.0, 35.0, 0.35, 90.0,  # percentile 90
        )
