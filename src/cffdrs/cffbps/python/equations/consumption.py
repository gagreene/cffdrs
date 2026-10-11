"""Total fuel consumption, head fire intensity, and fire intensity class."""
from __future__ import annotations

from numpy import ma as mask

MaskedArray = mask.MaskedArray


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
