"""ROS percentile growth and point-ignition acceleration parameter."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy import ma as mask
from scipy.stats import t

MaskedArray = mask.MaskedArray


def _tinv(probability: float | int, freedom: int = 9999999):
    """Standard-normal quantile, computed via a Student's t at very high freedom.

    Han & Braun (2014) specify the standard normal quantile directly; a t
    distribution at freedom=9999999 is numerically indistinguishable from it and
    is what this coefficient table's fuel-type sigmas were fit against.
    """
    return t.ppf(probability, freedom)


_MAX_FTYPE = 20
# Per-fuel-type growth-percentile coefficients, indexed directly by CFFBPS
# fuel-type code. Both are fitted noise standard deviations (Han & Braun 2014,
# Section 3): the surface-fire value scales a log-normal shift, the crown-fire
# value scales a Box-Cox-transformed (delta=0.6) power-law adjustment. NaN marks
# a fuel type with no fitted coefficient for that regime (percentile growth is a
# no-op there).
_SURFACE_SIGMA = np.full(_MAX_FTYPE + 1, np.nan, dtype=np.float32)
_CROWN_SIGMA = np.full(_MAX_FTYPE + 1, np.nan, dtype=np.float32)
for _ftype, _surface, _crown in (
    (1, None, 0.95), (2, 0.84, 1.82), (3, 0.62, 1.78), (4, 0.74, 1.38),
    (5, 0.80, None), (6, 0.66, 1.54), (7, 1.22, 1.00), (8, 0.716, None),
    (12, 0.551, None),
):
    if _surface is not None:
        _SURFACE_SIGMA[_ftype] = _surface
    if _crown is not None:
        _CROWN_SIGMA[_ftype] = _crown
del _ftype, _surface, _crown


def _wind_decay(w: MaskedArray) -> MaskedArray:
    """Wind-speed decay factor for backing-fire growth-percentile noise.

    Han & Braun (2014), the k(w) accompanying their Eq. 3: k(0) = 1, decaying as
    wind speed increases — backing-spread variability shrinks in strong wind, the
    same way backing ROS itself does.
    """
    low = np.exp(-0.10078 * w)
    high = np.exp(-0.05039 * w) / (12.0 * (1.0 - np.exp(-0.0818 * (w - 28.0))))
    return mask.where(w < 40, low, high)


def calc_ros_percentile_growth(*,
                               percentile_growth: float | int | None,
                               fuel_type: MaskedArray,
                               hros_cfb: MaskedArray,
                               bros_cfb: MaskedArray,
                               wsv: MaskedArray,
                               hros: MaskedArray,
                               bros: MaskedArray) -> tuple[MaskedArray, MaskedArray]:
    """Adjust head/backing ROS by a growth-percentile factor.

    Implements the variance-stabilized ROS quantile model of Han, L. & Braun,
    W.J. (2014), "Dionysus: a stochastic fire growth scenario generator",
    Environmetrics 25(6):431-442. Below the crowning threshold (cfb < 0.1), ROS
    residuals are treated as log-normal and the ROS is scaled multiplicatively
    by exp(tinv * sigma_surface). At or above it, a closed-form Box-Cox
    power-law adjustment (delta=0.6, the paper's fitted crown-fire transform)
    applies unless its radicand would go negative (outside the transform's
    valid domain), in which case it falls back to the same log-normal-shift
    form using the crown-fire sigma. Fuel types with no fitted sigma for the
    applicable regime are left unchanged, as is percentile_growth of None or 50
    (the median, i.e. no adjustment).

    Head and backing ROS are adjusted using their own, direction-specific CFB
    (hros_cfb/bros_cfb) to decide the surface-vs-crown regime — matching WISE's
    FBPFuel::ROS/BROS each computing CFB from their own direction's spread rate,
    rather than sharing one CFB value between both directions.

    Backing ROS additionally has its noise term scaled by a wind-speed decay
    factor, k(wsv) (paper Eq. 3's k(w)): backing-spread variability shrinks as
    wind speed increases, the same way backing ROS itself does. Head fire's
    noise is not wind-scaled.
    """
    if percentile_growth is None or percentile_growth == 50:
        return hros, bros

    tinv_value = _tinv(probability=percentile_growth / 100, freedom=9999999)

    ftype_idx = np.ma.filled(fuel_type, 0).astype(np.intp)
    surface_sigma = _SURFACE_SIGMA[ftype_idx]
    crown_sigma = _CROWN_SIGMA[ftype_idx]
    has_surface = ~np.isnan(surface_sigma)
    has_crown = ~np.isnan(crown_sigma)

    wind_decay = _wind_decay(wsv)

    adjusted = []
    for rsi, noise_scale, regime_cfb in ((hros, 1.0, hros_cfb), (bros, wind_decay, bros_cfb)):
        surface_regime = mask.where(has_surface, rsi * np.exp(tinv_value * surface_sigma * noise_scale), rsi)

        radicand = mask.power(rsi, 0.6) + tinv_value * crown_sigma * noise_scale
        power_law = mask.power(mask.where(radicand >= 0, radicand, 0.0), 1.0 / 0.6)
        crown_fallback = rsi * np.exp(tinv_value * crown_sigma * noise_scale)
        crown_regime = mask.where(has_crown, mask.where(radicand >= 0, power_law, crown_fallback), rsi)

        adjusted.append(mask.where(regime_cfb < 0.1, surface_regime, crown_regime))

    return adjusted[0], adjusted[1]


def calc_accel_param(*,
                     fuel_type: MaskedArray,
                     ftypes: Sequence[int],
                     open_fuel_types: Sequence[int],
                     cfb: MaskedArray,
                     accel_param: MaskedArray) -> MaskedArray:
    """Calculate the acceleration parameter for a point-ignition fire.

    ``accel_param`` is passed in as the initialized template.
    """
    # Mask for open fuel types that use a fixed acceleration parameter (0.115)
    fixed_accel_mask = mask.where(np.isin(fuel_type, open_fuel_types), True, False)

    # Mask for closed fuel types that require computation
    variable_accel_mask = mask.where(np.isin(fuel_type, ftypes) & ~fixed_accel_mask, True, False)

    # Compute acceleration parameter for open fuel types
    accel_param = mask.where(fixed_accel_mask, 0.115, accel_param)

    # Compute acceleration parameter for closed fuel types (safe calculation)
    accel_param = mask.where(variable_accel_mask,
                             0.115 - 18.8 * np.power(cfb, 2.5) * np.exp(-8 * cfb),
                             accel_param)

    return accel_param
