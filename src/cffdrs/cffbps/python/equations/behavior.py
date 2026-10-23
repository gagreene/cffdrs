"""Fire behavior calculated from the other results: fire type, head fire intensity and intensity class."""
from __future__ import annotations

from numpy import ma as mask

MaskedArray = mask.MaskedArray


def calc_fire_type(*, fuel_type: MaskedArray, cfb: MaskedArray) -> MaskedArray:
    """Calculate fire type (1: surface, 2: intermittent crown, 3: active crown)."""
    return mask.where((fuel_type < 19),
                      mask.where(cfb <= 0.1,
                                 # Surface fire
                                 1,
                                 mask.where((cfb > 0.1) & (cfb < 0.9),
                                            # Intermittent crown fire
                                            2,
                                            mask.where(cfb >= 0.9,
                                                       # Active crown fire
                                                       3,
                                                       # No fire type
                                                       0
                                                       )
                                            )
                                 ),
                      0
                      )


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
