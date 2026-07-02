"""Surface fuel consumption (FFC, WFC, SFC)."""
from __future__ import annotations

import numpy as np
from numpy import ma as mask

MaskedArray = mask.MaskedArray


def calc_sfc(*,
             fuel_type: MaskedArray,
             ffmc: MaskedArray,
             bui: MaskedArray,
             pc: MaskedArray,
             gfl: MaskedArray,
             ref_array: MaskedArray) -> tuple[MaskedArray, MaskedArray, MaskedArray]:
    """Calculate forest floor (FFC), woody (WFC), and total surface (SFC) consumption.

    :return: (ffc, wfc, sfc) as masked arrays.
    """
    with np.errstate(invalid='ignore', over='ignore'):
        # FFC, WFC, SFC default to nan
        ffc = np.full_like(ref_array, np.nan, dtype=np.float64)
        wfc = np.full_like(ref_array, np.nan, dtype=np.float64)
        sfc = np.full_like(ref_array, np.nan, dtype=np.float64)

        # ftype == 1
        mask1 = fuel_type == 1
        sfc1 = np.where(ffmc > 84,
                        0.75 + 0.75 * np.sqrt(1 - np.exp(-0.23 * (ffmc - 84))),
                        0.75 - 0.75 * np.sqrt(1 - np.exp(0.23 * (ffmc - 84))))
        sfc = np.where(mask1, sfc1, sfc)

        # ftype == 2
        mask2 = fuel_type == 2
        sfc2 = 5 * (1 - np.exp(-0.0115 * bui))
        sfc = np.where(mask2, sfc2, sfc)

        # ftype in [3, 4]
        mask34 = np.isin(fuel_type, [3, 4])
        sfc34 = 5 * np.power(1 - np.exp(-0.0164 * bui), 2.24)
        sfc = np.where(mask34, sfc34, sfc)

        # ftype in [5, 6]
        mask56 = np.isin(fuel_type, [5, 6])
        sfc56 = 5 * np.power(1 - np.exp(-0.0149 * bui), 2.48)
        sfc = np.where(mask56, sfc56, sfc)

        # ftype == 7
        mask7 = fuel_type == 7
        ffc7 = 2 * (1 - np.exp(-0.104 * (ffmc - 70)))
        ffc7 = np.where(ffc7 < 0, 0, ffc7)
        wfc7 = 1.5 * (1 - np.exp(-0.0201 * bui))
        sfc7 = ffc7 + wfc7
        ffc = np.where(mask7, ffc7, ffc)
        wfc = np.where(mask7, wfc7, wfc)
        sfc = np.where(mask7, sfc7, sfc)

        # ftype in [8, 9]
        mask89 = np.isin(fuel_type, [8, 9])
        sfc89 = 1.5 * (1 - np.exp(-0.0183 * bui))
        sfc = np.where(mask89, sfc89, sfc)

        # ftype in [10, 11]
        mask1011 = np.isin(fuel_type, [10, 11])
        c2_sfc = 5 * (1 - np.exp(-0.0115 * bui))
        d1_sfc = 1.5 * (1 - np.exp(-0.0183 * bui))
        sfc1011 = ((pc / 100) * c2_sfc) + (((100 - pc) / 100) * d1_sfc)
        sfc = np.where(mask1011, sfc1011, sfc)

        # ftype in [12, 13]
        mask1213 = np.isin(fuel_type, [12, 13])
        sfc1213 = 5 * (1 - np.exp(-0.0115 * bui))
        sfc = np.where(mask1213, sfc1213, sfc)

        # ftype in [14, 15]
        mask1415 = np.isin(fuel_type, [14, 15])
        sfc = np.where(mask1415, gfl, sfc)

        # ftype == 16
        mask16 = fuel_type == 16
        ffc16 = 4 * (1 - np.exp(-0.025 * bui))
        wfc16 = 4 * (1 - np.exp(-0.034 * bui))
        sfc16 = ffc16 + wfc16
        ffc = np.where(mask16, ffc16, ffc)
        wfc = np.where(mask16, wfc16, wfc)
        sfc = np.where(mask16, sfc16, sfc)

        # ftype == 17
        mask17 = fuel_type == 17
        ffc17 = 10 * (1 - np.exp(-0.013 * bui))
        wfc17 = 6 * (1 - np.exp(-0.06 * bui))
        sfc17 = ffc17 + wfc17
        ffc = np.where(mask17, ffc17, ffc)
        wfc = np.where(mask17, wfc17, wfc)
        sfc = np.where(mask17, sfc17, sfc)

        # ftype == 18
        mask18 = fuel_type == 18
        ffc18 = 12 * (1 - np.exp(-0.0166 * bui))
        wfc18 = 20 * (1 - np.exp(-0.021 * bui))
        sfc18 = ffc18 + wfc18
        ffc = np.where(mask18, ffc18, ffc)
        wfc = np.where(mask18, wfc18, wfc)
        sfc = np.where(mask18, sfc18, sfc)

        # ftype == 19 or 20 or unknown
        mask1920 = np.isin(fuel_type, [19, 20])
        ffc = np.where(mask1920, np.nan, ffc)
        wfc = np.where(mask1920, np.nan, wfc)
        sfc = np.where(mask1920, np.nan, sfc)

        # Assign FFC, WFC, SFC as masked arrays
        ffc = mask.array(ffc, mask=np.isnan(ffc))
        wfc = mask.array(wfc, mask=np.isnan(wfc))
        sfc = mask.array(sfc, mask=np.isnan(sfc))

    return ffc, wfc, sfc
