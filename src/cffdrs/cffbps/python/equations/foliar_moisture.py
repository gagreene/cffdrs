"""Foliar moisture content (FMC) and foliar moisture effect (FME), one function per quantity.

The quantities chain as latn -> d0, dj -> nd -> fmc -> fme; :func:`calc_foliar_moisture` runs the chain.
"""
from __future__ import annotations

from datetime import datetime as dt
from typing import NamedTuple

import numpy as np
from numpy import ma as mask

from ...common.fmc_inputs import CALCULATE

MaskedArray = mask.MaskedArray


class FoliarMoistureResult(NamedTuple):
    """Foliar moisture results; field names match the facade attributes (and the order is the unpack order)."""
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


def _resolve(override, derived: MaskedArray) -> MaskedArray:
    """An explicit override where it is given, the derived value in the cells that ask for a calculation.

    A cell holding ``CALCULATE`` (-1) takes the derived value; masked and NaN override cells stay missing.
    """
    given = _as_masked(override)
    calculate = (np.ma.getdata(given) == CALCULATE) & ~np.ma.getmaskarray(given)
    if not calculate.any():
        return given
    return mask.where(calculate, derived, given)


def calc_d0(*,
            lat: MaskedArray,
            latn: MaskedArray,
            elevation: MaskedArray,
            ) -> MaskedArray:
    """Calculate the date of minimum foliar moisture content (D0), a whole day number.

    The value is rounded to mimic the approach used in the cffdrs R package, and it is not limited to a
    calendar day (it can exceed 366 at extreme latitude and elevation).

    :param lat: latitude (decimal degrees)
    :param latn: normalized latitude from :func:`calc_latn`
    :param elevation: elevation (m)
    :return: D0 (day of the year)
    """
    return mask.MaskedArray.round(mask.where((elevation is not None) & (elevation > 0),
                                             142.1 * (lat / latn) + (0.0172 * elevation),
                                             151 * (lat / latn)),
                                  0)


def calc_dj(*,
            latn: MaskedArray,
            wx_date: int,
            ) -> MaskedArray:
    """Calculate the Julian date (day of the year) of the day being modelled (DJ).

    Where the normalized latitude is not finite (a missing longitude) the value is 0.

    :param latn: normalized latitude from :func:`calc_latn`
    :param wx_date: weather date as YYYYMMDD
    :return: DJ (day of the year)
    """
    return mask.where(np.isfinite(latn),
                      dt.strptime(str(wx_date), '%Y%m%d').timetuple().tm_yday,
                      0)


def calc_fmc(*,
             nd: MaskedArray,
             ) -> MaskedArray:
    """Calculate the foliar moisture content (FMC, %) from the number of days from D0.

    :param nd: number of days between DJ and D0 from :func:`calc_nd`
    :return: FMC (%)
    """
    return mask.where(nd < 30,
                      85 + (0.0189 * (nd ** 2)),
                      mask.where(nd < 50,
                                 32.9 + (3.17 * nd) - (0.0288 * (nd ** 2)),
                                 120))


def calc_fme(*,
             fmc: MaskedArray,
             ) -> MaskedArray:
    """Calculate the foliar moisture effect (FME) from the foliar moisture content; missing in, missing out.

    :param fmc: foliar moisture content (%)
    :return: FME
    """
    return 1000 * np.power(1.5 - (0.00275 * fmc), 4) / (460 + (25.9 * fmc))


def calc_foliar_moisture(*,
                         lat: MaskedArray,
                         long: MaskedArray,
                         elevation: MaskedArray,
                         wx_date: int,
                         d0_override: MaskedArray | int | None = None,
                         dj_override: MaskedArray | int | None = None,
                         fmc_override: MaskedArray | float | None = None,
                         ) -> FoliarMoistureResult:
    """Compute the foliar moisture quantities by chaining the functions of this module.

    An explicit ``d0_override``, ``dj_override`` or ``fmc_override`` replaces the derived D0, DJ or FMC; when it is
    ``None`` the value is derived (D0 and DJ from latitude, elevation and ``wx_date``, FMC from ND). In an override
    array a cell holding -1 (``CALCULATE``) is derived like the cells of no override. The FME always follows the FMC
    that is used. A NaN or masked cell in an override is a missing value.

    :param lat: latitude (decimal degrees)
    :param long: longitude (decimal degrees)
    :param elevation: elevation (m)
    :param wx_date: weather date as YYYYMMDD
    :param d0_override: explicit D0, or None to derive it
    :param dj_override: explicit DJ, or None to derive it
    :param fmc_override: explicit FMC (%), or None to derive it
    :return: latn, d0, dj, nd, fmc and fme
    """
    latn = calc_latn(long=long, elevation=elevation)
    d0 = calc_d0(lat=lat, latn=latn, elevation=elevation)
    if d0_override is not None:
        d0 = _resolve(d0_override, d0)
    dj = calc_dj(latn=latn, wx_date=wx_date)
    if dj_override is not None:
        dj = _resolve(dj_override, dj)
    nd = calc_nd(dj=dj, d0=d0)
    fmc = calc_fmc(nd=nd)
    if fmc_override is not None:
        fmc = _resolve(fmc_override, fmc)
    fme = calc_fme(fmc=fmc)
    return FoliarMoistureResult(latn=latn, d0=d0, dj=dj, nd=nd, fmc=fmc, fme=fme)


def calc_latn(*,
              long: MaskedArray,
              elevation: MaskedArray,
              ) -> MaskedArray:
    """Calculate the normalized latitude (LATN) from longitude and the elevation branch.

    :param long: longitude (decimal degrees)
    :param elevation: elevation (m)
    :return: LATN
    """
    return mask.where((elevation is not None) & (elevation > 0),
                      43 + (33.7 * np.exp(-0.0351 * (150 - np.abs(long)))),
                      46 + (23.4 * (np.exp(-0.036 * (150 - np.abs(long))))))


def calc_nd(*,
            dj: MaskedArray,
            d0: MaskedArray,
            ) -> MaskedArray:
    """Calculate the number of days between the modelled day and the date of minimum foliar moisture (ND).

    :param dj: Julian date of the modelled day
    :param d0: date of minimum foliar moisture content
    :return: ND (days, always non-negative)
    """
    return np.absolute(dj - d0)
