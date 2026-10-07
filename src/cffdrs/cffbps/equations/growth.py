"""ROS percentile growth and point-ignition acceleration parameter."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy import ma as mask
from scipy.stats import t

MaskedArray = mask.MaskedArray


def _tinv(probability: float | int, freedom: int = 9999999):
    """Standard-normal quantile, computed via a Student's t at very high freedom.

    Han & Braun (2014) use the standard normal quantile; a t distribution at
    freedom=9999999 agrees with it to about 1e-7 relative.
    """
    return t.ppf(probability, freedom)


# Han & Braun (2014), Section 3: noise standard deviations estimated on pooled
# conifer data (surface fires on the log scale, crown fires on the Box-Cox
# delta=0.6 scale). The paper pools by category because per-fuel-type data are
# sparse, so these are single values, not per-fuel-type tables.
_SURFACE_SIGMA = 0.923
_CROWN_SIGMA = 1.637
_CROWN_DELTA = 0.6

# Project choice, not from the paper: percentiles are capped to this open-interval margin so the
# 0th/100th percentile (z = -inf/+inf) give finite ROS instead of 0/inf/masked values. The paper
# only demonstrates 10-90 and does not discuss extreme percentiles. NaN is not capped.
_MIN_PERCENTILE = 0.001
_MAX_PERCENTILE = 99.999

# Project choice, not from the paper: the pooled conifer fit is applied to the
# CFFBPS conifer fuel types C-1..C-7 (codes 1-7), per regime. C-1 is adjusted
# only for crown fires and C-5 only for surface fires. The paper reports only
# pooled conifer values and has no per-fuel-type coverage, so this pattern is a
# project decision, not something derived from it. Other fuel types are outside
# the conifer estimates the paper reports and are left unadjusted.
_SURFACE_FUEL_TYPES = (2, 3, 4, 5, 6, 7)
_CROWN_FUEL_TYPES = (1, 2, 3, 4, 6, 7)


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
    by exp(tinv * 0.923). At or above it, a closed-form Box-Cox power-law
    adjustment (delta=0.6, sigma=1.637) applies, giving ROS 0 where its
    radicand would go negative (outside the transform's range). The two
    sigmas are the paper's pooled conifer estimates. Fuel types other than
    C-1..C-7 are left unchanged, as are C-1 in the surface regime and C-5 in the
    crown regime, and percentile_growth of None or 50 (the median, i.e. no
    adjustment). Percentiles outside (0, 100) are capped to 0.001/99.999 rather than
    rejected; NaN propagates as NaN.

    Project choices, not from the paper: the cfb < 0.1 regime rule (the paper
    assumes the fire type is known), the negative-radicand zero guard, the percentile cap, and the
    C-1..C-7 fuel scope with its C-1 crown-only and C-5 surface-only coverage.

    Head and backing ROS are adjusted using their own, direction-specific CFB
    (hros_cfb/bros_cfb) to decide the surface-vs-crown regime — CFB is computed
    from their own direction's spread rate, rather than sharing one CFB value
    between both directions.

    Backing ROS additionally has its noise term scaled by a wind-speed decay
    factor, k(wsv) (paper Eq. 3's k(w)): backing-spread variability shrinks as
    wind speed increases, the same way backing ROS itself does. Head fire's
    noise is not wind-scaled.
    """
    if percentile_growth is None or percentile_growth == 50:
        return hros, bros

    capped_percentile = float(np.clip(percentile_growth, _MIN_PERCENTILE, _MAX_PERCENTILE))  # NaN stays NaN
    tinv_value = _tinv(probability=capped_percentile / 100, freedom=9999999)

    ftype = np.ma.filled(fuel_type, 0)
    has_surface = np.isin(ftype, _SURFACE_FUEL_TYPES)
    has_crown = np.isin(ftype, _CROWN_FUEL_TYPES)
    wind_decay = _wind_decay(wsv)

    adjusted = []
    for rsi, noise_scale, regime_cfb in ((hros, 1.0, hros_cfb), (bros, wind_decay, bros_cfb)):
        surface_regime = mask.where(has_surface, rsi * np.exp(tinv_value * _SURFACE_SIGMA * noise_scale), rsi)

        shift = tinv_value * _CROWN_SIGMA * noise_scale
        radicand = mask.power(rsi, _CROWN_DELTA) + shift
        # Guard (project choice, not in the paper): a negative radicand is outside the Box-Cox
        # transform's range, so no positive ROS exists at that percentile and the result is 0. This
        # keeps the output continuous and non-decreasing in ROS. NaN radicands (invalid percentile)
        # are restored as unmasked NaN, since mask.power would otherwise mask them.
        power_law = mask.power(mask.where(radicand < 0, 0.0, radicand), 1.0 / _CROWN_DELTA)
        power_law = mask.where(np.isnan(radicand), np.nan, power_law)
        crown_regime = mask.where(has_crown, power_law, rsi)

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
