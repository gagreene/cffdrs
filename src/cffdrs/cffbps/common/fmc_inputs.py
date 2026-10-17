"""Explicit ``d0`` / ``dj`` / ``fmc`` overrides for the foliar moisture calculation.

``d0`` (date of minimum foliar moisture) is a model parameter and is deliberately not
range-limited. ``dj`` is a calendar day: a supplied value is floored and then clipped to
1..365 (1..366 in a leap year), and clipping is announced with a ``UserWarning``.

A negative cell (documented as ``CALCULATE``, -1) asks for that cell to be calculated: every negative value is
normalised to -1 when the override is coerced. A negative scalar is no override at all (``None``), because a scalar
run has one cell. Nodata values (for example -999) are not recognised as nodata: mask them before they get here.
"""
import calendar
import warnings

import numpy as np
from numpy import ma as mask

from .fingerprint import fingerprint

#: The value that asks for a cell of a ``d0``, ``dj`` or ``fmc`` override to be calculated (any negative value does).
CALCULATE = -1


def coerce_override(name: str, value):
    """Return a supplied ``d0``/``dj``/``fmc`` as a private masked array (or ``None``).

    Scalars become 1-element arrays; NaN and masked cells are missing; the caller's array is copied.
    An unmasked infinity is rejected for ``dj`` (``ValueError``). A negative scalar means "calculate" and is
    ``None``; in an array every negative cell (masked and NaN cells excepted) is set to ``CALCULATE`` on the private
    copy, and nothing is rewritten when there is no negative or every negative already is ``CALCULATE``.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating, np.ndarray)):
        raise TypeError(f'{name} must be int, float, or numpy ndarray data types')
    data = np.array(np.ma.getdata(value)) if isinstance(value, np.ndarray) else np.array([value])
    if data.dtype.kind not in 'iuf':
        raise TypeError(f'{name} must be int, float, or numpy ndarray data types')
    missing = np.ma.getmaskarray(value).copy() if isinstance(value, np.ndarray) else np.zeros(data.shape, bool)
    if data.dtype.kind == 'f':
        missing |= np.isnan(data)
        if name == 'dj' and np.isinf(data[~missing]).any():
            raise ValueError('dj override must be finite (it contains infinity)')
    if data.dtype.kind in 'if':
        negative = (data < 0) & ~missing
        if negative.any():
            if not isinstance(value, np.ndarray):
                return None
            if (data[negative] != CALCULATE).any():
                data[negative] = CALCULATE
    return mask.array(data, mask=missing)


def resolve_dj_override(supplied, wx_date: int, warned: set | None = None):
    """Floor a supplied ``dj`` and clip it to the days of the year of ``wx_date``.

    :param supplied: masked array from :func:`coerce_override`; ``CALCULATE`` cells are left as they are (neither
        floored, clipped nor counted), the caller of the foliar moisture calculation replaces them by the derived value.
    :param wx_date: YYYYMMDD integer; its year decides 365 or 366 days.
    :param warned: set of (fingerprint, year) pairs already announced; a clipped value
        warns once per distinct pair. ``None`` always warns.
    :return: masked array, ``int64`` when no cell is missing, else ``float64`` with NaN data.
    """
    year = wx_date // 10000
    top = 366 if calendar.isleap(year) else 365
    data = np.ma.getdata(supplied).astype(np.float64)
    missing = np.ma.getmaskarray(supplied)
    calculate = (data == CALCULATE) & ~missing
    floored = np.floor(np.where(missing, 1.0, data))
    clipped_cells = (~missing) & ~calculate & ((floored < 1) | (floored > top))
    count = int(clipped_cells.sum())
    if count:
        key = (fingerprint(supplied), year)
        if warned is None or key not in warned:
            if warned is not None:
                warned.add(key)
            warnings.warn(f'dj override: {count} value(s) outside 1..{top} (year {year}) were clipped',
                          UserWarning, stacklevel=3)
    days = np.where(calculate, CALCULATE, np.clip(floored, 1, top))
    if missing.any():
        return mask.array(np.where(missing, np.nan, days), mask=missing)
    return mask.array(days.astype(np.int64), mask=False)
