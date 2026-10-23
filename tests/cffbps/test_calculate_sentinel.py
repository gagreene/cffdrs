"""A negative d0/dj/fmc override cell (documented as -1) means "calculate this cell"; a scalar one means no override."""
import warnings

import numpy as np
import pytest
from numpy import ma as mask

from cffdrs.cffbps.common.fmc_inputs import CALCULATE, coerce_override
from cffdrs.cffbps.python import FBP
from cffdrs.cffbps.python.equations import foliar_moisture as fm

N = 4
BASE = dict(wx_date=20230615, lat=np.full(N, 55.), long=np.full(N, -110.), elevation=np.full(N, 500.),
            slope=np.full(N, 10.), aspect=np.full(N, 180.), ws=np.full(N, 20.), wd=np.full(N, 0.),
            ffmc=np.full(N, 90.), bui=np.full(N, 80.))
ALL = ['latn', 'd0', 'dj', 'nd', 'fmc', 'fme', 'hros', 'hfi', 'cros']

# ----------------------------------------------------------------------------- before calcFMC (array placeholder)

C6_STAGES = ['invertWindAspect', 'calcSF', 'calcISZ', 'calcISI_RSI_BE', 'calcROS', 'calcSFC', 'getCBH_CFL', 'calcCSFI',
             'calcRSO', 'calcC6BlendCFB', 'calcC6BlendCFC', 'calcC6CROS', 'calcC6HROS']


# ---- helpers ----

def _arr(*values):
    return mask.array(np.array(values, dtype=float))


def derived():
    fbp = grid()
    fbp.runFBP()
    return fbp


def grid(**changes):
    fbp = FBP()
    fbp.initialize(**(BASE | dict(fuel_type=np.full(N, 6), out_request=ALL) | changes))
    return fbp


def run_c6_without_calcfmc(fbp):
    for stage in C6_STAGES:
        getattr(fbp, stage)()


# ---- calcfmc tests ----
def test_before_calcfmc_given_cells_have_fmc_and_fme_and_calculate_cells_are_missing():
    fbp = grid(fmc=np.array([-1., 99., 101., -999.]))
    fmc, fme = fbp.getParams(['fmc', 'fme'])
    assert np.isnan(fmc[[0, 3]]).all() and np.isnan(fme[[0, 3]]).all()
    np.testing.assert_array_equal(fmc[[1, 2]], [99, 101])
    np.testing.assert_allclose(fme[[1, 2]], fm.calc_fme(fmc=mask.array([99., 101.])))


def test_calcfmc_replaces_the_placeholder_and_the_result_equals_the_full_run():
    given = np.array([-1., 99., 101., -1.])
    skipped = grid(fmc=given)
    run_c6_without_calcfmc(skipped)
    skipped.calcFMC()
    full = grid(fmc=given)
    full.runFBP()
    fresh = grid(fmc=given)
    out = fresh.runFBP()
    for name in ('latn', 'd0', 'dj', 'nd', 'fmc', 'fme'):
        np.testing.assert_array_equal(skipped.getParams([name])[0], full.getParams([name])[0])
    base = derived().getParams(['fmc'])[0]
    np.testing.assert_array_equal(out[ALL.index('fmc')], [base[0], 99, 101, base[3]])


def test_calcfmc_with_a_calculate_array_replaces_an_existing_override():
    fbp = grid(fmc=np.full(N, 99.))
    fbp.calcFMC(fmc=np.full(N, -1.))
    np.testing.assert_array_equal(fbp.getParams(['fmc'])[0], derived().getParams(['fmc'])[0])


# ---- calculate tests ----

def test_calculate_constant_is_minus_one():
    assert CALCULATE == -1



@pytest.mark.parametrize('route', ['initialize', 'set_override', 'setParams', 'attribute', 'calcFMC'])
@pytest.mark.parametrize('name', ['d0', 'dj', 'fmc'])
def test_calculate_cells_on_every_route_give_the_derived_value_and_keep_the_given_ones(name, route):
    given = np.array([-1., 140., -999., 130.])
    if name == 'fmc':
        given = np.array([-1., 99., -999., 101.])
    fbp = grid(**({name: given} if route == 'initialize' else {}))
    if route == 'set_override':
        fbp.set_override(name, given)
    elif route == 'setParams':
        fbp.setParams({name: given})
    elif route == 'attribute':
        setattr(fbp, name, given)
    elif route == 'calcFMC':
        fbp.calcFMC(**{name: given})
    out = fbp.runFBP()
    base = derived().runFBP()
    col = ALL.index(name)
    np.testing.assert_array_equal(out[col][[0, 2]], base[col][[0, 2]])
    np.testing.assert_array_equal(out[col][[1, 3]], given[[1, 3]])


def test_calculate_overrides_are_sticky_across_runs_and_reset_and_can_be_cleared():
    given = np.array([-1., 99., -1., 101.])
    fbp = grid(fmc=given)
    first = fbp.runFBP()
    fbp.reset()
    second = fbp.runFBP()
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    fbp.setParams({'fmc': None})
    cleared = fbp.runFBP()
    base = derived().runFBP()
    for a, b in zip(cleared, base, strict=True):
        np.testing.assert_array_equal(a, b)


def test_calculate_cells_in_an_integer_override_array_keep_working():
    fbp = grid(d0=np.array([-1, 150, -9, 140], dtype=np.int16))
    out = fbp.runFBP()
    base = derived().runFBP()
    d0 = out[ALL.index('d0')]
    assert d0[0] == base[ALL.index('d0')][0] and d0[1] == 150 and d0[3] == 140


# ---- dj tests ----
def test_dj_still_rejects_infinity_before_the_negative_rule():
    for value in (np.inf, -np.inf, np.array([-np.inf, 5.])):
        with pytest.raises(ValueError, match='infin'):
            coerce_override('dj', value)


def test_dj_calculate_cells_take_the_derived_value():
    base = fm.calc_foliar_moisture(**KW)
    got = fm.calc_foliar_moisture(**KW, dj_override=_arr(-1, 170, 100, -1))
    assert got.dj[0] == base.dj[0] == got.dj[3] and got.dj[1] == 170 and got.dj[2] == 100


def test_dj_calculate_cells_do_not_warn_or_count_and_zero_still_clips():
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        fbp = grid(dj=np.array([-1., -5., 0., 400.]))
        fbp.runFBP()
    assert [str(w.message) for w in seen] == ['dj override: 2 value(s) outside 1..365 (year 2023) were clipped']
    dj = fbp.getParams(['dj'])[0]
    base = derived().getParams(['dj'])[0]
    np.testing.assert_array_equal(dj, [base[0], base[1], 1, 365])


def test_no_dj_warning_when_every_cell_is_calculated():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        grid(dj=np.array([-1., -999., -5., -0.5])).runFBP()


# ---- scalar tests ----
@pytest.mark.parametrize('name', ['d0', 'dj', 'fmc'])
@pytest.mark.parametrize('value', [-1, -5, -0.5, -999.0])
def test_a_scalar_negative_is_no_override(name, value):
    assert coerce_override(name, value) is None


@pytest.mark.parametrize('name', ['d0', 'dj', 'fmc'])
@pytest.mark.parametrize('value', [-1, -5, -0.5])
def test_a_scalar_negative_runs_like_no_override(name, value):
    base = derived().runFBP()
    got = grid(**{name: value}).runFBP()
    for a, b in zip(got, base, strict=True):
        np.testing.assert_array_equal(a, b)
    fbp = grid(**{name: 120})
    fbp.calcFMC(**{name: value})
    for a, b in zip(fbp.runFBP(), base, strict=True):
        np.testing.assert_array_equal(a, b)


def test_a_scalar_negative_before_calcfmc_is_the_zero_template_like_no_override():
    for value in (-1, -5):
        fbp, plain = grid(fmc=value), grid()
        for a, b in zip(fbp.getParams(['fmc', 'fme']), plain.getParams(['fmc', 'fme']), strict=True):
            np.testing.assert_array_equal(a, b)
        run_c6_without_calcfmc(fbp)
        run_c6_without_calcfmc(plain)
        np.testing.assert_array_equal(fbp.getParams(['hros'])[0], plain.getParams(['hros'])[0])


# ---- other tests ----
@pytest.mark.parametrize('name', ['d0', 'dj', 'fmc'])
def test_every_negative_cell_becomes_minus_one_and_the_callers_array_is_untouched(name):
    given = np.array([-999., -5., -0.5, -1., 150., np.nan])
    keep = given.copy()
    got = coerce_override(name, given)
    np.testing.assert_array_equal(np.ma.getdata(got)[:5], [-1, -1, -1, -1, 150])
    assert bool(got.mask[5]) and np.isnan(np.ma.getdata(got)[5])
    np.testing.assert_array_equal(given, keep)


@pytest.mark.parametrize('name', ['d0', 'dj', 'fmc'])
def test_signed_integer_arrays_keep_their_dtype(name):
    got = coerce_override(name, np.array([-999, 5, -1], dtype=np.int16))
    assert got.dtype == np.int16
    np.testing.assert_array_equal(np.ma.getdata(got), [-1, 5, -1])


@pytest.mark.parametrize('name', ['d0', 'dj', 'fmc'])
def test_masked_cells_are_not_calculate_cells(name):
    given = np.ma.array([-999., 150., -7.], mask=[True, False, False])
    got = coerce_override(name, given)
    assert list(got.mask) == [True, False, False]
    np.testing.assert_array_equal(np.ma.getdata(got)[1:], [150, -1])


@pytest.mark.parametrize('name', ['d0', 'dj', 'fmc'])
def test_a_one_element_array_is_an_array_and_keeps_the_request(name):
    got = coerce_override(name, np.array([-3.]))
    np.testing.assert_array_equal(np.ma.getdata(got), [-1])


@pytest.mark.parametrize('name', ['d0', 'dj', 'fmc'])
def test_non_negative_values_and_nan_scalars_are_unchanged(name):
    assert np.ma.getdata(coerce_override(name, 150)).tolist() == [150]
    assert bool(coerce_override(name, np.nan).mask[0])


def test_d0_calculate_cells_take_the_derived_value_and_the_rest_the_given_one():
    base = fm.calc_foliar_moisture(**KW)
    got = fm.calc_foliar_moisture(**KW, d0_override=_arr(-1, 150, np.nan, 120))
    assert got.d0[0] == base.d0[0] and got.d0[1] == 150 and got.d0[3] == 120
    assert mask.is_masked(got.d0[2]) and mask.is_masked(got.nd[2]) and mask.is_masked(got.fmc[2])
    assert got.nd[0] == base.nd[0] and got.nd[1] == abs(base.dj[1] - 150)


def test_fmc_calculate_cells_use_the_calculated_chain_and_given_cells_the_given_value():
    base = fm.calc_foliar_moisture(**KW, d0_override=_arr(150, 150, 150, 150))
    got = fm.calc_foliar_moisture(**KW, d0_override=_arr(150, 150, 150, 150), fmc_override=_arr(-1, 99, np.nan, 101))
    assert got.fmc[0] == base.fmc[0] and got.fmc[1] == 99 and got.fmc[3] == 101
    assert mask.is_masked(got.fmc[2]) and mask.is_masked(got.fme[2])
    np.testing.assert_allclose(got.fme[[0, 1, 3]], fm.calc_fme(fmc=mask.array([base.fmc[0], 99, 101])))


def test_all_three_overrides_can_mix_calculate_given_and_missing_cells():
    got = fm.calc_foliar_moisture(**KW, d0_override=_arr(-1, 150, np.nan, 150), dj_override=_arr(170, -1, 170, 170),
                                  fmc_override=_arr(-1, -1, 90, 95))
    base = fm.calc_foliar_moisture(**KW)
    assert got.d0[0] == base.d0[0] and got.d0[1] == 150
    assert got.dj[0] == 170 and got.dj[1] == base.dj[1]
    assert not mask.is_masked(got.fmc[2]) and got.fmc[2] == 90      # a given fmc does not depend on the masked d0
    assert mask.is_masked(got.nd[2])
    assert got.fmc[3] == 95
    assert got.fmc[0] == fm.calc_fmc(nd=got.nd[0:1])[0]


def test_equation_level_a_fully_calculated_override_equals_no_override():
    base = fm.calc_foliar_moisture(**KW)
    got = fm.calc_foliar_moisture(**KW, d0_override=_arr(-1, -1, -1, -1), dj_override=_arr(-1, -1, -1, -1),
                                  fmc_override=_arr(-1, -1, -1, -1))
    for a, b in zip(got, base, strict=True):
        np.testing.assert_array_equal(mask.getdata(a), mask.getdata(b))


def test_c6_with_calcfmc_skipped_is_missing_only_at_the_calculate_cells():
    fbp = grid(fmc=np.array([-1., 99., 101., -1.]))
    run_c6_without_calcfmc(fbp)
    hros = fbp.getParams(['hros'])[0]
    assert np.isnan(hros[[0, 3]]).all()
    assert np.isfinite(hros[[1, 2]]).all() and (hros[[1, 2]] > 0).all()


def test_changing_an_input_after_the_override_does_nothing_until_calcfmc_runs():
    fbp = grid(fmc=np.array([-1., 99., 101., -1.]))
    before = fbp.getParams(['fmc', 'fme'])
    fbp.setParams({'wx_date': 20230716, 'lat': np.full(N, 50.)})
    after = fbp.getParams(['fmc', 'fme'])
    for a, b in zip(before, after, strict=True):
        np.testing.assert_array_equal(a, b)
    fbp.calcFMC()
    assert np.isfinite(fbp.getParams(['fmc'])[0]).all()


KW = dict(lat=_arr(55., 55., 55., 55.), long=_arr(-110., -110., -110., -110.),
          elevation=_arr(500., 500., 500., 500.), wx_date=20230615)
