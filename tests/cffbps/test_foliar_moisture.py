"""One function per foliar moisture quantity, and the orchestrator that chains them."""
import numpy as np
import pytest
from numpy import ma as mask

from cffdrs.cffbps.python.equations import foliar_moisture as fm


def _arr(*values):
    return mask.array(np.array(values, dtype=float))


def test_latn_uses_the_elevation_branch_and_follows_a_missing_longitude():
    long = _arr(-110.0, -110.0, -110.0)
    elevation = _arr(500.0, 0.0, -5.0)
    latn = fm.calc_latn(long=long, elevation=elevation)
    above = 43 + 33.7 * np.exp(-0.0351 * (150 - 110.0))
    below = 46 + 23.4 * np.exp(-0.036 * (150 - 110.0))
    np.testing.assert_allclose(latn, [above, below, below], rtol=1e-14)
    assert above == pytest.approx(51.277142808079475, rel=1e-14)
    missing = fm.calc_latn(long=mask.array([np.nan], mask=[True]), elevation=_arr(500.0))
    assert mask.is_masked(missing)


def test_d0_is_rounded_to_a_whole_day_and_uses_the_elevation_branch():
    lat, elevation = _arr(55.0, 55.0), _arr(500.0, 0.0)
    latn = fm.calc_latn(long=_arr(-110.0, -110.0), elevation=elevation)
    d0 = fm.calc_d0(lat=lat, latn=latn, elevation=elevation)
    np.testing.assert_array_equal(d0, np.round([142.1 * (55.0 / latn[0]) + 0.0172 * 500.0, 151 * (55.0 / latn[1])]))
    assert d0[0] == 161
    assert np.all(d0 == np.floor(d0))


def test_dj_is_the_day_of_year_and_zero_where_longitude_is_missing():
    latn = mask.array([51.0, np.nan], mask=[False, True])
    assert mask.getdata(fm.calc_dj(latn=latn, wx_date=20230615)).tolist() == [166, 0]
    assert mask.getdata(fm.calc_dj(latn=latn, wx_date=20240301)).tolist() == [61, 0]          # leap year
    assert mask.getdata(fm.calc_dj(latn=mask.array([51.0]), wx_date=20231231)).tolist() == [365]


def test_nd_is_the_distance_in_days_whatever_the_side():
    nd = fm.calc_nd(dj=_arr(100.0, 200.0, 150.0), d0=_arr(150.0, 150.0, 150.0))
    np.testing.assert_array_equal(nd, [50, 50, 0])


@pytest.mark.parametrize('nd,expected', [
    (0, 85.0), (10, 85.0 + 0.0189 * 100), (29.999, 85.0 + 0.0189 * 29.999 ** 2),
    (30, 32.9 + 3.17 * 30 - 0.0288 * 900), (49.999, 32.9 + 3.17 * 49.999 - 0.0288 * 49.999 ** 2),
    (50, 120.0), (300, 120.0),
])
def test_fmc_follows_the_three_branches_of_nd(nd, expected):
    assert fm.calc_fmc(nd=_arr(nd))[0] == pytest.approx(expected, rel=1e-14)


def test_fmc_of_a_missing_nd_is_missing():
    assert mask.is_masked(fm.calc_fmc(nd=mask.array([np.nan], mask=[True])))


def test_fme_follows_the_formula_and_a_missing_fmc():
    fme = fm.calc_fme(fmc=_arr(85.0, 100.0, 120.0))
    values = np.array([85.0, 100.0, 120.0])
    expected = 1000 * (1.5 - 0.00275 * values) ** 4 / (460 + 25.9 * values)
    np.testing.assert_allclose(fme, expected, rtol=1e-14)
    assert fme[2] == pytest.approx(0.5251926036995515, rel=1e-14)
    assert mask.is_masked(fm.calc_fme(fmc=mask.array([np.nan], mask=[True])))


def test_the_orchestrator_chains_the_functions_and_keeps_the_field_order():
    assert fm.FoliarMoistureResult._fields == ('latn', 'd0', 'dj', 'nd', 'fmc', 'fme')
    lat, long, elevation = _arr(55.0, 60.0), _arr(-110.0, -120.0), _arr(500.0, 0.0)
    result = fm.calc_foliar_moisture(lat=lat, long=long, elevation=elevation, wx_date=20230615)
    latn = fm.calc_latn(long=long, elevation=elevation)
    d0 = fm.calc_d0(lat=lat, latn=latn, elevation=elevation)
    dj = fm.calc_dj(latn=latn, wx_date=20230615)
    nd = fm.calc_nd(dj=dj, d0=d0)
    fmc = fm.calc_fmc(nd=nd)
    for got, want in zip(result, (latn, d0, dj, nd, fmc, fm.calc_fme(fmc=fmc)), strict=True):
        np.testing.assert_array_equal(got, want)
    assert result.fmc[0] == pytest.approx(85.4725, rel=1e-12)


def test_the_orchestrator_prefers_explicit_d0_and_dj():
    kwargs = dict(lat=_arr(55.0), long=_arr(-110.0), elevation=_arr(500.0), wx_date=20230615)
    result = fm.calc_foliar_moisture(**kwargs, d0_override=150, dj_override=170)
    assert [mask.getdata(a).item() for a in (result.d0, result.dj, result.nd)] == [150, 170, 20]
    assert mask.getdata(result.fmc).item() == pytest.approx(85.0 + 0.0189 * 400, rel=1e-14)
    masked = fm.calc_foliar_moisture(**kwargs, d0_override=np.nan)
    assert mask.is_masked(masked.d0) and mask.is_masked(masked.fmc)
