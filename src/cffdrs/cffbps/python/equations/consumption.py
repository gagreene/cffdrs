"""Total fuel consumption, head fire intensity, and fire intensity class."""
from __future__ import annotations

import numpy as np
from numpy import ma as mask

MaskedArray = mask.MaskedArray


def zero_without_spread(*, hros: MaskedArray, values: tuple[MaskedArray, ...]) -> tuple[MaskedArray, ...]:
    """Make fuel consumption follow whether the fire spreads.

    No fuel is consumed where the head rate of spread is exactly 0, and consumption is missing
    where the rate of spread is missing (NaN or masked); everywhere else the values are unchanged.

    :param hros: head fire rate of spread (m/min)
    :param values: consumption arrays (for example ffc, wfc, sfc, cfc) to adjust
    :return: the adjusted arrays, in the same order
    """
    data = np.asarray(mask.getdata(hros), dtype=np.float64)
    hmask = mask.getmaskarray(hros)
    zero = (data == 0) & ~hmask
    nan_ros = np.isnan(data) & ~hmask
    adjusted = []
    for value in values:
        arr = mask.asarray(value)
        base = np.asarray(mask.getdata(arr), dtype=np.float64)
        result = np.where(zero, 0.0, np.where(nan_ros, np.nan, base))
        adjusted.append(mask.array(result, mask=(mask.getmaskarray(arr) | hmask) & ~zero))
    return tuple(adjusted)


def calc_tfc(*, sfc: MaskedArray, cfc: MaskedArray) -> MaskedArray:
    """Calculate total fuel consumed (kg/m^2)."""
    return sfc + cfc


def calc_hfi(*, hros: MaskedArray, tfc: MaskedArray) -> MaskedArray:
    """Calculate head fire intensity (kW/m)."""
    return 300 * hros * tfc


def calc_fire_intensity_class(*, hfi: MaskedArray) -> MaskedArray:
    """Classify head fire intensity into CFFBPS fire intensity classes (1-6)."""
    return mask.where(
        (hfi > 0) & (hfi <= 10), 1,
        mask.where((hfi > 10) & (hfi <= 500), 2,
                   mask.where((hfi > 500) & (hfi <= 2000), 3,
                              mask.where((hfi > 2000) & (hfi <= 4000), 4,
                                         mask.where((hfi > 4000) & (hfi <= 10000), 5,
                                                    mask.where((hfi > 10000), 6,
                                                               -99)
                                                    )
                                         )
                              )
                   )
    )
