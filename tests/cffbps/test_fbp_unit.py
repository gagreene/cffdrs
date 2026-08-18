"""Targeted unit tests for FBP paths NOT exercised by the golden snapshots.

The regression snapshots (test_fbp_regression.py) run with default settings, so
several branches never fire there: percentile growth, the multiprocessing driver,
alphanumeric fuel-type inputs, R grid-code conversion, C6 canopy overrides, and
the season grass-curing helper. These tests lock those paths.
"""
from __future__ import annotations

import numpy as np
import pytest

from cffdrs.cffbps import (
    FBP,
    constants,
    convert_grid_codes,
    fbpMultiprocessArray,
    getSeasonGrassCuring,
)

BASE_KWARGS = dict(
    wx_date=20230701, lat=55.0, long=-110.0, elevation=500,
    slope=10, aspect=180, ws=20, wd=0, ffmc=90, bui=80,
)


def _run(fuel_type, out_request, **overrides):
    kwargs = dict(BASE_KWARGS, fuel_type=fuel_type, out_request=list(out_request), **overrides)
    fbp = FBP()
    fbp.initialize(**kwargs)
    return [float(np.asarray(v).ravel()[0]) for v in fbp.runFBP()]


# ── Alphanumeric fuel-type input ───────────────────────────────────────────────
def test_alpha_fuel_type_equals_numeric():
    """fuel_type='C2' must produce identical outputs to fuel_type=2."""
    out_req = ['hros', 'hfi', 'fire_type', 'wsv', 'raz', 'sfc', 'cfb']
    assert _run('C2', out_req) == _run(2, out_req)


# ── R grid-code conversion ─────────────────────────────────────────────────────
def test_convert_grid_codes_mapping():
    """cffdrs-R codes 9-13 shift up by one; 19 (R water) becomes 20."""
    arr = np.ma.array([1, 8, 9, 10, 11, 12, 13, 19], dtype=np.int8)
    out = convert_grid_codes(arr)
    assert list(np.asarray(out)) == [1, 8, 10, 11, 12, 13, 19, 20]


def test_convert_fuel_type_codes_flag_matches_manual_conversion():
    """initialize(convert_fuel_type_codes=True) equals passing the converted code."""
    out_req = ['hros', 'hfi', 'fire_type']
    # R code 9 == this module's code 10 (M-1)
    assert _run(9, out_req, convert_fuel_type_codes=True) == _run(10, out_req)


# ── Percentile growth branch (Han & Braun 2014, "Dionysus") ────────────────────
def test_percentile_growth_locks_current_behavior():
    """Lock hros/bros under percentile_growth=90 (branch is a no-op at 50).

    Literals captured from the current, snapshot-validated code. If a refactor
    changes this branch, these values must be consciously regenerated.

    The C6 literals reflect the corrected pipeline: deterministic C6 HROS is
    blended before percentile growth, while HROS and BROS select their regimes
    from direction-specific CFB. The adjusted C6 HROS is no longer overwritten
    by a later blend, and C6 BROS no longer reuses head-derived SROS for its CFB.
    """
    expected = {
        2: (27.11959245018106, 1.6012591850964195),   # C-2: crown table entry
        6: (21.229110929146493, 0.074701148095697),  # C-6: adjust blended HROS; BROS uses its own CFB
        14: (17.667074274782212, 2.0109729485336385),  # O-1a: no crown entry
    }
    for ft, (hros_exp, bros_exp) in expected.items():
        hros, bros = _run(ft, ['hros', 'bros'], percentile_growth=90)
        assert hros == pytest.approx(hros_exp, rel=1e-12), f'ft={ft} hros'
        assert bros == pytest.approx(bros_exp, rel=1e-12), f'ft={ft} bros'


def test_percentile_growth_50_is_noop():
    """percentile_growth=50 (default) must not adjust ROS."""
    assert _run(2, ['hros', 'bros'], percentile_growth=50) == _run(2, ['hros', 'bros'])


def test_percentile_growth_surface_regime_uses_fuel_type_sigma():
    """Surface-regime (cfb < 0.1) percentile growth must scale by the fuel type's
    fitted sigma (Han & Braun 2014's per-fuel-type noise standard deviation), not
    a bare exp(tinv). Locks the fix for a bug (inherited from WISE) where the
    surface sigma was checked for eligibility (>= 0) but its magnitude was never
    actually multiplied into the adjustment."""
    from cffdrs.cffbps.equations.growth import _tinv, calc_ros_percentile_growth

    fuel_type = np.ma.array([8], dtype=np.int8, mask=False)  # D1: surface sigma 0.716, no crown entry
    cfb = np.ma.array([0.0], mask=False)  # surface regime
    wsv = np.ma.array([0.0], mask=False)
    hros_in = np.ma.array([1.0], mask=False)
    bros_in = np.ma.array([1.0], mask=False)

    hros90, _ = calc_ros_percentile_growth(
        percentile_growth=90, fuel_type=fuel_type, hros_cfb=cfb, bros_cfb=cfb, wsv=wsv,
        hros=hros_in.copy(), bros=bros_in.copy(),
    )
    expected = np.exp(_tinv(0.9) * np.float32(0.716))
    assert float(hros90[0]) == pytest.approx(expected, rel=1e-6)


def test_percentile_growth_bros_wind_decay():
    """Backing-fire growth-percentile noise must shrink toward the unadjusted
    value as wind speed increases (Han & Braun 2014's k(w)); head-fire noise must
    not depend on wind speed at all."""
    from cffdrs.cffbps.equations.growth import calc_ros_percentile_growth

    fuel_type = np.ma.array([6], dtype=np.int8, mask=False)  # C6: crown sigma 1.54
    cfb = np.ma.array([0.9], mask=False)  # crown regime, shared by both directions here
    hros_in = np.ma.array([10.0], mask=False)
    bros_in = np.ma.array([10.0], mask=False)

    hros_low, bros_low = calc_ros_percentile_growth(
        percentile_growth=90, fuel_type=fuel_type, hros_cfb=cfb, bros_cfb=cfb, wsv=np.ma.array([0.0], mask=False),
        hros=hros_in.copy(), bros=bros_in.copy(),
    )
    hros_high, bros_high = calc_ros_percentile_growth(
        percentile_growth=90, fuel_type=fuel_type, hros_cfb=cfb, bros_cfb=cfb, wsv=np.ma.array([80.0], mask=False),
        hros=hros_in.copy(), bros=bros_in.copy(),
    )

    # head-fire adjustment never depends on wind speed
    assert float(hros_low[0]) == pytest.approx(float(hros_high[0]), rel=1e-12)
    # backing-fire adjustment shrinks toward the unadjusted value as wind increases
    assert abs(float(bros_high[0]) - 10.0) < abs(float(bros_low[0]) - 10.0)
    # at wsv=0, k(w)=1: backing gets the *same* adjustment as head fire (identical inputs)
    assert float(bros_low[0]) == pytest.approx(float(hros_low[0]), rel=1e-12)


def test_percentile_growth_uses_direction_specific_cfb():
    """hros and bros must each pick their surface-vs-crown regime from their own
    CFB, not a single shared value — matches WISE's FBPFuel::ROS/BROS each
    computing CFB from their own direction's spread rate (FBPFuel.cpp:754-759,
    793-798). A case straddling the cfb=0.1 threshold (hros_cfb >= 0.1, i.e.
    head-fire crown regime; bros_cfb < 0.1, i.e. backing-fire surface regime)
    must apply different formulas to hros vs bros."""
    from cffdrs.cffbps.equations.growth import calc_ros_percentile_growth

    fuel_type = np.ma.array([2], dtype=np.int8, mask=False)  # C2: surface sigma 0.84, crown sigma 1.82
    hros_cfb = np.ma.array([0.9], mask=False)   # head fire: crown regime
    bros_cfb = np.ma.array([0.05], mask=False)  # backing fire: surface regime
    wsv = np.ma.array([0.0], mask=False)        # k(0) = 1, isolates the CFB effect from wind decay
    hros_in = np.ma.array([10.0], mask=False)
    bros_in = np.ma.array([10.0], mask=False)

    hros90, bros90 = calc_ros_percentile_growth(
        percentile_growth=90, fuel_type=fuel_type, hros_cfb=hros_cfb, bros_cfb=bros_cfb, wsv=wsv,
        hros=hros_in.copy(), bros=bros_in.copy(),
    )

    # Same raw RSI (10.0) and same tinv, but different regimes -> different formulas,
    # so the two adjusted values must differ even though hros_in == bros_in.
    assert float(hros90[0]) != pytest.approx(float(bros90[0]), rel=1e-9)


@pytest.mark.parametrize(('percentile', 'expected'), [(0, 0.0), (100, np.inf)])
def test_percentile_growth_boundary_behavior(percentile, expected):
    """Percentiles 0/100 retain scipy's zero/infinity boundary behavior."""
    from cffdrs.cffbps.equations.growth import calc_ros_percentile_growth

    one = np.ma.array([1.0], mask=False)
    zero = np.ma.array([0.0], mask=False)
    with np.errstate(invalid='ignore', over='ignore'):
        hros, bros = calc_ros_percentile_growth(
            percentile_growth=percentile,
            fuel_type=np.ma.array([8], dtype=np.int8, mask=False),
            hros_cfb=zero, bros_cfb=zero, wsv=zero,
            hros=one, bros=one,
        )

    if np.isinf(expected):
        assert np.isposinf(hros[0])
        assert np.isposinf(bros[0])
    else:
        assert float(hros[0]) == expected
        assert float(bros[0]) == expected


@pytest.mark.parametrize('percentile', [np.nan, -1, 101])
def test_percentile_growth_invalid_values_propagate_nan(percentile):
    """NaN and out-of-range percentiles preserve the existing NaN contract."""
    from cffdrs.cffbps.equations.growth import calc_ros_percentile_growth

    one = np.ma.array([1.0], mask=False)
    zero = np.ma.array([0.0], mask=False)
    with np.errstate(invalid='ignore'):
        hros, bros = calc_ros_percentile_growth(
            percentile_growth=percentile,
            fuel_type=np.ma.array([8], dtype=np.int8, mask=False),
            hros_cfb=zero, bros_cfb=zero, wsv=zero,
            hros=one, bros=one,
        )

    assert np.isnan(hros[0])
    assert np.isnan(bros[0])


def test_calc_cfb_backing_uses_bros_not_hros():
    """Directional CFB must differ when HROS and BROS differ.

    This applies to every crowning fuel type, including C6; otherwise the two
    directional facade calls would be silently redundant.
    """
    from cffdrs.cffbps.equations.crown import calc_cfb

    fuel_type = np.ma.array([2, 6], dtype=np.int8, mask=False)
    ftypes = [2, 6]
    non_crowning_fuels = constants.non_crowning_fuels
    rso = np.ma.array([2.0, 2.0], mask=False)

    hros_based = calc_cfb(fuel_type=fuel_type, ftypes=ftypes, non_crowning_fuels=non_crowning_fuels,
                          rso=rso, ros=np.ma.array([15.0, 15.0], mask=False))
    bros_based = calc_cfb(fuel_type=fuel_type, ftypes=ftypes, non_crowning_fuels=non_crowning_fuels,
                          rso=rso, ros=np.ma.array([3.0, 3.0], mask=False))

    assert np.all(np.asarray(hros_based) != np.asarray(bros_based))


# ── C6 percentile-growth pipeline ──────────────────────────────────────────────
def test_c6_cros_and_hros_are_separate_ros_equations():
    """C6 CROS and HROS must be independently callable from equations.ros."""
    from cffdrs.cffbps.equations.ros import calc_c6_cros, calc_c6_hros

    fuel_type = np.ma.array([6, 2], dtype=np.int8, mask=False)
    cfc = np.ma.array([0.5, 0.5], mask=False)
    isi = np.ma.array([10.0, 10.0], mask=False)
    fme = np.ma.array([0.8, 0.8], mask=False)
    cros_in = np.ma.array([0.0, 7.0], mask=False)
    cros = calc_c6_cros(fuel_type=fuel_type, cfc=cfc, isi=isi, fme=fme, cros=cros_in)

    expected_cros = 60 * (1 - np.exp(-0.0497 * 10.0)) * (0.8 / 0.778237)
    assert float(cros[0]) == pytest.approx(expected_cros)
    assert float(cros[1]) == 7.0

    sros = np.ma.array([4.0, 4.0], mask=False)
    hros_in = np.ma.array([4.0, 9.0], mask=False)
    blend_cfb = np.ma.array([0.25, 0.25], mask=False)
    hros = calc_c6_hros(
        fuel_type=fuel_type, sros=sros, cros=cros,
        c6_blend_cfb=blend_cfb, hros=hros_in,
    )

    assert float(hros[0]) == pytest.approx(4.0 + 0.25 * (expected_cros - 4.0))
    assert float(hros[1]) == 9.0


def test_c6_percentile_pipeline_uses_blended_hros_and_recalculates_cfb():
    """C6 percentile growth must adjust blended HROS and final CFB must use it."""
    fbp = FBP()
    fbp.initialize(fuel_type=6, percentile_growth=90, out_request=['hros'], **BASE_KWARGS)
    fbp.runFBP()

    deterministic_hros = (
        float(fbp.sros[0])
        + float(fbp.c6_blend_cfb[0]) * (float(fbp.cros[0]) - float(fbp.sros[0]))
    )
    expected_regime_cfb = 1 - np.exp(-0.23 * (deterministic_hros - float(fbp.rso[0])))
    expected_final_cfb = 1 - np.exp(-0.23 * (float(fbp.hros[0]) - float(fbp.rso[0])))
    expected_final_bros_cfb = 1 - np.exp(-0.23 * (float(fbp.bros[0]) - float(fbp.rso[0])))

    assert float(fbp.hros[0]) != pytest.approx(deterministic_hros)
    assert float(fbp.percentile_cfb[0]) == pytest.approx(np.clip(expected_regime_cfb, 0, 1))
    assert float(fbp.cfb[0]) == pytest.approx(np.clip(expected_final_cfb, 0, 1))
    assert float(fbp.bros_cfb[0]) == pytest.approx(np.clip(expected_final_bros_cfb, 0, 1))
    assert float(fbp.c6_blend_cfc[0]) == pytest.approx(float(fbp.c6_blend_cfb[0] * fbp.cfl[0]))
    assert float(fbp.cfc[0]) == pytest.approx(float(fbp.cfb[0] * fbp.cfl[0]))


def test_c6_percentile_50_keeps_blended_hros_but_uses_generic_final_cfb():
    """At percentile 50, C6 ROS is unchanged but final CFB is based on blended HROS."""
    fbp = FBP()
    fbp.initialize(fuel_type=6, percentile_growth=50, out_request=['hros'], **BASE_KWARGS)
    fbp.runFBP()

    deterministic_hros = (
        float(fbp.sros[0])
        + float(fbp.c6_blend_cfb[0]) * (float(fbp.cros[0]) - float(fbp.sros[0]))
    )
    expected_final_cfb = 1 - np.exp(-0.23 * (deterministic_hros - float(fbp.rso[0])))

    assert float(fbp.hros[0]) == pytest.approx(deterministic_hros)
    assert float(fbp.cfb[0]) == pytest.approx(np.clip(expected_final_cfb, 0, 1))
    assert float(fbp.cfb[0]) != pytest.approx(float(fbp.c6_blend_cfb[0]))


def test_non_c6_downstream_outputs_use_post_percentile_cfb():
    """Fire type, acceleration, and CFC must consume recalculated final CFB."""
    fbp = FBP()
    fbp.initialize(fuel_type=2, percentile_growth=90, out_request=['hros'], **BASE_KWARGS)
    fbp.runFBP()

    expected_cfb = np.clip(1 - np.exp(-0.23 * (float(fbp.hros[0]) - float(fbp.rso[0]))), 0, 1)
    expected_fire_type = 1 if expected_cfb <= 0.1 else 2 if expected_cfb < 0.9 else 3
    expected_accel = 0.115 - 18.8 * expected_cfb ** 2.5 * np.exp(-8 * expected_cfb)

    assert float(fbp.cfb[0]) == pytest.approx(expected_cfb)
    assert int(fbp.fire_type[0]) == expected_fire_type
    assert float(fbp.accel_param[0]) == pytest.approx(expected_accel)
    assert float(fbp.cfc[0]) == pytest.approx(expected_cfb * float(fbp.cfl[0]))


def test_percentile_regime_is_selected_once(monkeypatch):
    """Final CFB recalculation must not trigger a second percentile adjustment."""
    from cffdrs.cffbps import facade

    calls = []

    def adjust_once(**kwargs):
        calls.append((kwargs['hros_cfb'].copy(), kwargs['bros_cfb'].copy()))
        return kwargs['hros'] * 10, kwargs['bros'] * 10

    monkeypatch.setattr(facade.growth_eq, 'calc_ros_percentile_growth', adjust_once)
    fbp = FBP()
    fbp.initialize(fuel_type=2, percentile_growth=90, out_request=['hros'], **BASE_KWARGS)
    fbp.runFBP()

    assert len(calls) == 1
    assert float(fbp.cfb[0]) != pytest.approx(float(calls[0][0][0]))


# ── Multiprocessing driver ─────────────────────────────────────────────────────
def test_multiprocess_matches_direct_run():
    """fbpMultiprocessArray block results must equal a direct FBP run."""
    ft = np.array([[[1, 2, 3, 6, 14], [7, 8, 9, 10, 16]]], dtype=np.int8)
    shape = ft.shape

    def full(v):
        return np.full(shape, v, dtype=np.float64)

    arrays = dict(
        lat=full(55.0), long=full(-110.0), elevation=full(500), slope=full(10),
        aspect=full(180), ws=full(20), wd=full(0), ffmc=full(90), bui=full(80),
        pc=full(50), pdf=full(35), gfl=full(0.35), gcf=full(80),
    )
    out_req = ['hros', 'hfi', 'fire_type']

    mp_out = fbpMultiprocessArray(
        fuel_type=ft, wx_date=20230701, out_request=out_req,
        num_processors=2, block_size=2, **arrays,
    )

    fbp = FBP()
    fbp.initialize(fuel_type=ft, wx_date=20230701, out_request=out_req, **arrays)
    direct = fbp.runFBP()

    for name, mp_arr, direct_arr in zip(out_req, mp_out, direct, strict=True):
        np.testing.assert_allclose(
            np.asarray(mp_arr, dtype=np.float64),
            np.ma.filled(np.ma.asarray(direct_arr), np.nan).astype(np.float64),
            rtol=0, atol=0, err_msg=f'multiprocess {name} != direct',
        )


# ── C6 canopy overrides ────────────────────────────────────────────────────────
def test_cbh_cfl_override_applies_for_c6():
    """C6 cbh/cfl overrides must change CSFI/CFC-dependent outputs."""
    fbp = FBP()
    fbp.initialize(fuel_type=6, out_request=['csfi'], **BASE_KWARGS)
    fbp.invertWindAspect()
    fbp.calcSF()
    fbp.calcISZ()
    fbp.calcFMC()
    fbp.calcISI_RSI_BE()
    fbp.getCBH_CFL(ftype=6, cbh=3.0, cfl=1.0)
    assert float(np.asarray(fbp.cbh).ravel()[0]) == 3.0
    assert float(np.asarray(fbp.cfl).ravel()[0]) == 1.0


def test_cbh_cfl_override_rejected_for_non_c6():
    """Only C-6 may have canopy values adjusted; other types must raise."""
    fbp = FBP()
    fbp.initialize(fuel_type=2, out_request=['csfi'], **BASE_KWARGS)
    with pytest.raises(ValueError, match='C-6'):
        fbp.getCBH_CFL(ftype=2, cbh=3.0)


# ── Season grass curing helper ─────────────────────────────────────────────────
def test_get_season_grass_curing():
    assert getSeasonGrassCuring('spring', 'AB') == 75
    assert getSeasonGrassCuring('summer', 'BC') == 60
    assert getSeasonGrassCuring('summer', 'BC', subregion='southeast') == 90
    assert getSeasonGrassCuring('spring', 'XX') is None  # unknown province


# ── Table mutability contract ──────────────────────────────────────────────────
def test_module_constants_are_runtime_immutable():
    """Module-level LUTs must reject mutation (guards against global leaks)."""
    with pytest.raises(TypeError):
        constants.rosParams[1] = None  # type: ignore[index]
    with pytest.raises(TypeError):
        constants.fbpCBH_CFL_HT_LUT[6] = (0, 0, 0)  # type: ignore[index]
    assert isinstance(constants.open_fuel_types, tuple)
    assert isinstance(constants.valid_outputs, tuple)


def test_instance_tables_are_isolated_and_mutable():
    """Per-instance tables are calibratable copies: mutating one instance must not
    leak into other instances or the module constants (historical behavior)."""
    a, b = FBP(), FBP()
    assert a.rosParams is not b.rosParams
    original = a.rosParams[6]
    a.rosParams[6] = (30, 0.08, 3, 0.8, 62, 1.3)
    assert b.rosParams[6] == original
    assert constants.rosParams[6] == original
    a.open_fuel_types.append(99)
    assert 99 not in b.open_fuel_types
    assert 99 not in constants.open_fuel_types


# ── Multiprocessing driver edge cases ──────────────────────────────────────────
def test_multiprocess_default_out_request():
    """Omitting out_request must use the runFBP defaults, not crash."""
    ft = np.array([[[2, 3], [7, 8]]], dtype=np.int8)
    shape = ft.shape

    def full(v):
        return np.full(shape, v, dtype=np.float64)

    out = fbpMultiprocessArray(
        fuel_type=ft, wx_date=20230701,
        lat=full(55.0), long=full(-110.0), elevation=full(500), slope=full(10),
        aspect=full(180), ws=full(20), wd=full(0), ffmc=full(90), bui=full(80),
        num_processors=2, block_size=1,
    )
    assert len(out) == 3  # hros, hfi, fire_type
    assert np.isfinite(out[0]).all()


def test_block_size_estimator_never_returns_zero():
    """Small rasters must not crash the stride-0 path (3x3 previously did)."""
    from cffdrs.cffbps.parallel import _estimate_optimal_block_size
    assert _estimate_optimal_block_size((1, 3, 3), 2) >= 1
    assert _estimate_optimal_block_size((1, 1, 1), 2) >= 1
    assert _estimate_optimal_block_size((1, 6, 6), 2) >= 1


# ── Multiprocessing num_processors warning ─────────────────────────────────────
def test_low_num_processors_warns_not_prints(capsys):
    """num_processors < 2 must emit a UserWarning (visible to pytest.warns / logging
    capture), and the old print() fallback message must be gone from stdout — not
    just "a warning was added alongside the print that's still there" (the first
    draft's pytest.warns-only assertion would have passed even if the print stayed)."""
    ft = np.array([[[2, 3], [7, 8]]], dtype=np.int8)
    shape = ft.shape

    def full(v):
        return np.full(shape, v, dtype=np.float64)

    with pytest.warns(UserWarning, match='at least two cores'):
        fbpMultiprocessArray(
            fuel_type=ft, wx_date=20230701,
            lat=full(55.0), long=full(-110.0), elevation=full(500), slope=full(10),
            aspect=full(180), ws=full(20), wd=full(0), ffmc=full(90), bui=full(80),
            num_processors=1, block_size=1,
        )
    captured = capsys.readouterr()
    assert 'Defaulting num_processors to 2' not in captured.out


# ── Unknown fuel-type code on the explicit-ftype path ──────────────────────────
def test_cbh_cfl_unknown_ftype_raises_keyerror():
    """An ftype absent from the LUT must fail with a KeyError naming the code,
    not an opaque NoneType subscript error."""
    fbp = FBP()
    fbp.initialize(fuel_type=2, out_request=['csfi'], **BASE_KWARGS)
    with pytest.raises(KeyError):
        fbp.getCBH_CFL(ftype=25)


# ── Own-and-return convention (facade.py shared-template optimization) ────────
def _assert_untouched(arr, data_snapshot, mask_snapshot):
    np.testing.assert_array_equal(arr.data, data_snapshot)
    np.testing.assert_array_equal(np.ma.getmaskarray(arr), mask_snapshot)


def test_equations_do_not_mutate_input_arrays():
    """calc_cbh_cfl/calc_ros/calc_accel_param/calc_isi_rsi_be must never write into
    their inputs — the facade's shared zero-filled output template (initialize(),
    facade.py:406-413) depends on every calcX reassigning whole attributes rather
    than mutating them. Checks both .data and the mask: a function that leaves data
    alone but corrupts the mask is still an own-and-return violation."""
    from cffdrs.cffbps.equations.crown import calc_cbh_cfl
    from cffdrs.cffbps.equations.growth import calc_accel_param
    from cffdrs.cffbps.equations.ros import calc_ros
    from cffdrs.cffbps.equations.slope_wind import calc_isi_rsi_be

    fuel_type = np.ma.array([2, 6, 9], dtype=np.int8, mask=False)

    # calc_cbh_cfl: cbh/cfl passed in as templates
    cbh_in = np.ma.array([0.0, 0.0, 0.0], mask=False)
    cfl_in = np.ma.array([0.0, 0.0, 0.0], mask=False)
    cbh_in_data, cbh_in_mask = cbh_in.data.copy(), np.ma.getmaskarray(cbh_in).copy()
    cfl_in_data, cfl_in_mask = cfl_in.data.copy(), np.ma.getmaskarray(cfl_in).copy()
    cbh_out, _cfl_out = calc_cbh_cfl(
        fuel_type=fuel_type, cbh=cbh_in, cfl=cfl_in,
        cbh_cfl_ht_lut=constants.fbpCBH_CFL_HT_LUT,
    )
    _assert_untouched(cbh_in, cbh_in_data, cbh_in_mask)
    _assert_untouched(cfl_in, cfl_in_data, cfl_in_mask)
    assert not np.array_equal(cbh_out.data, cbh_in_data)  # sanity: it did compute something

    # calc_ros: sros passed in as template, only C6 cells written
    sros_in = np.ma.array([0.0, 0.0, 0.0], mask=False)
    sros_in_data, sros_in_mask = sros_in.data.copy(), np.ma.getmaskarray(sros_in).copy()
    rsi = np.ma.array([5.0, 5.0, 5.0], mask=False)
    brsi = np.ma.array([2.0, 2.0, 2.0], mask=False)
    be = np.ma.array([1.0, 1.0, 1.0], mask=False)
    bui = np.ma.array([80.0, 80.0, 80.0], mask=False)
    calc_ros(rsi=rsi, brsi=brsi, be=be, fuel_type=fuel_type, bui=bui, sros=sros_in)
    _assert_untouched(sros_in, sros_in_data, sros_in_mask)

    # calc_accel_param: accel_param passed in as template
    accel_in = np.ma.array([0.0, 0.0, 0.0], mask=False)
    accel_in_data, accel_in_mask = accel_in.data.copy(), np.ma.getmaskarray(accel_in).copy()
    cfb = np.ma.array([0.3, 0.3, 0.3], mask=False)
    calc_accel_param(
        fuel_type=fuel_type, ftypes=[2, 6, 9],
        open_fuel_types=constants.open_fuel_types, cfb=cfb, accel_param=accel_in,
    )
    _assert_untouched(accel_in, accel_in_data, accel_in_mask)

    # calc_isi_rsi_be: ref_array passed in as the shared template a/b/c/q/bui0/be_max
    # are copied from (slope_wind.py:189-194) — missed in the first draft of this plan.
    ref_array_in = np.ma.array([0.0, 0.0, 0.0], mask=False)
    ref_in_data, ref_in_mask = ref_array_in.data.copy(), np.ma.getmaskarray(ref_array_in).copy()
    ones = np.ma.array([1.0, 1.0, 1.0], mask=False)
    calc_isi_rsi_be(
        fuel_type=fuel_type, ros_params=dict(constants.rosParams),
        gcf=80.0 * ones, isz=10.0 * ones, sf=ones, pc=50.0 * ones, pdf=35.0 * ones,
        bui=80.0 * ones, fF=0.5 * ones, wd=np.ma.array([0.0, 0.0, 0.0], mask=False),
        aspect=180.0 * ones, ws=20.0 * ones, ref_array=ref_array_in,
    )
    _assert_untouched(ref_array_in, ref_in_data, ref_in_mask)


# ── NamedTuple contracts (each consumed differently — see plan Task 2) ─────────
def test_setattr_namedtuples_match_facade_attributes():
    """VerifiedInputs/ISIRSIBEResult field names are used as
    setattr(self, field, value) targets in facade.py (facade.py:236-237, :566-568).
    If a field is ever renamed without a matching facade attribute rename, setattr
    would silently create a new, never-read attribute instead of updating the
    intended one. This test catches that class of drift immediately."""
    from cffdrs.cffbps.equations.slope_wind import ISIRSIBEResult
    from cffdrs.cffbps.inputs import VerifiedInputs

    fbp = FBP()
    for named_tuple_cls in (VerifiedInputs, ISIRSIBEResult):
        missing = [f for f in named_tuple_cls._fields if not hasattr(fbp, f)]
        assert not missing, f'{named_tuple_cls.__name__} fields not found on FBP: {missing}'


def test_fmc_result_field_order_matches_positional_unpack():
    """FMCResult is unpacked positionally, not via setattr (facade.py:547):
        self.latn, self.d0, self.dj, self.nd, self.fmc, self.fme = fmc_eq.calc_fmc(...)
    A field reorder in FMCResult's class body would silently scramble which value
    lands on which attribute — hasattr can't catch that, only order can."""
    from cffdrs.cffbps.equations.fmc import FMCResult

    assert FMCResult._fields == ('latn', 'd0', 'dj', 'nd', 'fmc', 'fme')


def test_slope_wind_isi_fields_are_wired_into_isi_rsi_be_result():
    """SlopeWindISI is internal to calc_isi_rsi_be (never touches the facade
    directly); its fields are folded into ISIRSIBEResult one by one
    (slope_wind.py:322-323). If SlopeWindISI grows a field, this catches it not
    also being wired through to the result the facade actually consumes."""
    from cffdrs.cffbps.equations.slope_wind import ISIRSIBEResult, SlopeWindISI

    missing = set(SlopeWindISI._fields) - set(ISIRSIBEResult._fields)
    assert not missing, f'SlopeWindISI fields not present on ISIRSIBEResult: {missing}'


# ── out_request validation ──────────────────────────────────────────────────────
def test_unknown_out_request_raises():
    """A typo'd or unsupported out_request name must raise, not silently return NaN
    (previously indistinguishable from a real missing-data NaN)."""
    fbp = FBP()
    fbp.initialize(fuel_type=2, out_request=['hors'], **BASE_KWARGS)  # typo: 'hors' not 'hros'
    with pytest.raises(ValueError, match='hors'):
        fbp.runFBP()


def test_valid_out_request_still_works():
    """Sanity check: valid names must still run cleanly after the validation is added."""
    fbp = FBP()
    fbp.initialize(fuel_type=2, out_request=['hros', 'hfi', 'fire_type'], **BASE_KWARGS)
    result = fbp.runFBP()
    assert len(result) == 3


# ── pc/pdf/gfl/gcf NaN-to-default coercion (inputs.py:_coerce) ─────────────────
def test_nan_optional_fields_use_documented_defaults():
    """NaN for pc/pdf/gfl/gcf must be coerced to their documented defaults
    (50, 35, 0.35, 80) by inputs._coerce, not propagate as masked/missing — this is
    called out as deliberate, golden-locked behavior in inputs.py's _coerce
    docstring, but was not actually exercised by any existing test. Asserting the
    coerced fbp.pc/pdf/gfl/gcf attributes directly (rather than hros/hfi outputs
    that don't depend on them for a pure-conifer fuel type) is what actually
    exercises the coercion path."""
    nan = float('nan')
    fbp = FBP()
    fbp.initialize(fuel_type=2, pc=nan, pdf=nan, gfl=nan, gcf=nan, **BASE_KWARGS)
    assert float(np.asarray(fbp.pc).ravel()[0]) == 50
    assert float(np.asarray(fbp.pdf).ravel()[0]) == 35
    assert float(np.asarray(fbp.gfl).ravel()[0]) == 0.35
    assert float(np.asarray(fbp.gcf).ravel()[0]) == 80
    assert not np.ma.is_masked(fbp.pc)
    assert not np.ma.is_masked(fbp.pdf)
    assert not np.ma.is_masked(fbp.gfl)
    assert not np.ma.is_masked(fbp.gcf)
