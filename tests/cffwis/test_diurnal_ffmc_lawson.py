import sys
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cffdrs.cffwis import diurnalFFMC_lawson
from cffdrs.diurnal_ffmc_lawson import hourly_ffmc_lawson_vectorized


def test_hourly_ffmc_lawson_vectorized_supports_broadcasting():
    ffmc = 85.0
    rh = np.array([25.0, 50.0, 90.0], dtype=float)

    out = hourly_ffmc_lawson_vectorized(ffmc=ffmc, rh=rh, hour=10, minute=15)
    expected = np.array(
        [hourly_ffmc_lawson_vectorized(ffmc=ffmc, rh=float(r), hour=10, minute=15) for r in rh],
        dtype=float,
    )

    assert out.shape == rh.shape
    assert_allclose(out, expected, rtol=0.0, atol=1e-10)


def test_hourly_ffmc_lawson_vectorized_supports_masked_inputs():
    ffmc = np.ma.array([80.0, 85.0, 90.0], mask=[False, True, False])
    rh = np.ma.array([40.0, 50.0, 60.0], mask=[False, False, True])

    out = hourly_ffmc_lawson_vectorized(ffmc=ffmc, rh=rh, hour=10, minute=15)

    assert out.shape == (3,)
    assert np.isfinite(out[0])
    assert np.isnan(out[1])
    assert np.isnan(out[2])


def test_diurnal_ffmc_lawson_wrapper_handles_masked_and_broadcast_inputs():
    ffmc_1200 = np.ma.array([80.0, np.nan, 90.0], mask=[False, True, False])
    rh_1200 = 45.0

    out = diurnalFFMC_lawson(ffmc_1200=ffmc_1200, rh_1200=rh_1200, forecast_hour=14, forecast_minute=15)

    assert out.shape == (3,)
    assert np.isfinite(out[0])
    assert np.isnan(out[1])
    assert np.isfinite(out[2])


def test_hourly_ffmc_lawson_vectorized_rh_class_uses_half_hour_offset():
    # The RH-class threshold column shifts one column earlier than the
    # hour-row interpolation index for the first 30 minutes of an hour,
    # and matches it for the last 30. At 07:10/RH=70 that selects the M
    # (medium) table (threshold 77); at 07:40/RH=70 it crosses into H
    # (threshold 67). Expected values are hand-traced from the published
    # Lawson, Armitage & Hoskins (1996) M/H "700"/"800" table rows.
    early = hourly_ffmc_lawson_vectorized(ffmc=85.0, rh=70.0, hour=7, minute=10)
    late = hourly_ffmc_lawson_vectorized(ffmc=85.0, rh=70.0, hour=7, minute=40)

    assert_allclose(early, 72.25, rtol=0.0, atol=1e-9)
    assert_allclose(late, 69.4, rtol=0.0, atol=1e-9)


def test_diurnal_ffmc_lawson_wrapper_returns_nan_for_all_nan_input():
    ffmc_1200 = np.array([np.nan, np.nan], dtype=float)
    rh_1200 = np.array([45.0, 55.0], dtype=float)

    out = diurnalFFMC_lawson(ffmc_1200=ffmc_1200, rh_1200=rh_1200, forecast_hour=12, forecast_minute=0)

    assert out.shape == (2,)
    assert np.isnan(out).all()


