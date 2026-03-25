import numpy as np
from numpy.testing import assert_allclose
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cffwis import diurnalFFMC_lawson
from diurnal_ffmc_lawson import hourly_ffmc_lawson_vectorized


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


def test_diurnal_ffmc_lawson_wrapper_returns_nan_for_all_nan_input():
    ffmc_1200 = np.array([np.nan, np.nan], dtype=float)
    rh_1200 = np.array([45.0, 55.0], dtype=float)

    out = diurnalFFMC_lawson(ffmc_1200=ffmc_1200, rh_1200=rh_1200, forecast_hour=12, forecast_minute=0)

    assert out.shape == (2,)
    assert np.isnan(out).all()


