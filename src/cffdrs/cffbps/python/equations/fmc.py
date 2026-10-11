"""Foliar moisture content (FMC) and foliar moisture effect (FME)."""
from __future__ import annotations

from datetime import datetime as dt
from typing import NamedTuple

import numpy as np
from numpy import ma as mask

MaskedArray = mask.MaskedArray


class FMCResult(NamedTuple):
    """Foliar moisture results; field names match the facade attributes."""
    latn: MaskedArray
    d0: MaskedArray
    dj: MaskedArray
    nd: MaskedArray
    fmc: MaskedArray
    fme: MaskedArray


def _as_masked(value) -> MaskedArray:
    """Coerce an explicit d0/dj override (scalar, array or masked array) to a masked array; NaN is masked."""
    data = np.ma.getdata(value)
    if data.dtype.kind == 'f':
        return mask.array(data, mask=np.ma.getmaskarray(value) | np.isnan(data))
    return mask.array(data, mask=np.ma.getmaskarray(value))


def calc_fmc(*,
             lat: MaskedArray,
             long: MaskedArray,
             elevation: MaskedArray,
             wx_date: int,
             d0: MaskedArray | None,
             dj: MaskedArray | None,
             d0_override: int | None = None,
             dj_override: int | None = None,
             ) -> FMCResult:
    """Compute foliar moisture content and effect.

    ``d0``/``dj`` are the already-validated instance values (may be ``None``);
    ``d0_override``/``dj_override`` are optional caller-supplied Julian dates. When
    both are ``None`` the value is derived from latitude/elevation and ``wx_date``.
    """
    # Calculate normalized latitude
    latn = mask.where((elevation is not None) & (elevation > 0),
                      43 + (33.7 * np.exp(-0.0351 * (150 - np.abs(long)))),
                      46 + (23.4 * (np.exp(-0.036 * (150 - np.abs(long))))))

    # D0 calculation (an explicit override takes priority over the derived value)
    if d0_override is not None:
        d0 = _as_masked(d0_override)
    elif d0 is None:
        # Calculate date of minimum foliar moisture content (D0)
        # This value is rounded to mimic the approach used in the cffdrs R package.
        d0 = mask.MaskedArray.round(mask.where((elevation is not None) & (elevation > 0),
                                               142.1 * (lat / latn) + (0.0172 * elevation),
                                               151 * (lat / latn)),
                                    0)

    # Julian Date (an explicit override takes priority over the derived value)
    if dj_override is not None:
        dj = _as_masked(dj_override)
    elif dj is None:
        # Calculate Julian date (Dj)
        dj = mask.where(np.isfinite(latn),
                        dt.strptime(str(wx_date), '%Y%m%d').timetuple().tm_yday,
                        0)

    # Number of days between Dj and D0 (ND)
    nd = np.absolute(dj - d0)

    # Calculate foliar moisture content (FMC)
    fmc = mask.where(nd < 30,
                     85 + (0.0189 * (nd ** 2)),
                     mask.where(nd < 50,
                                32.9 + (3.17 * nd) - (0.0288 * (nd ** 2)),
                                120))

    # Calculate foliar moisture effect (FME)
    fme = 1000 * np.power(1.5 - (0.00275 * fmc), 4) / (460 + (25.9 * fmc))

    return FMCResult(latn=latn, d0=d0, dj=dj, nd=nd, fmc=fmc, fme=fme)
