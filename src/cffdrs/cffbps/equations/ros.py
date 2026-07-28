"""Fire rate of spread (head, backing, and C6 surface rate of spread)."""
from __future__ import annotations

from numpy import ma as mask

MaskedArray = mask.MaskedArray


def calc_ros(*,
             rsi: MaskedArray,
             brsi: MaskedArray,
             be: MaskedArray,
             fuel_type: MaskedArray,
             bui: MaskedArray,
             sros: MaskedArray) -> tuple[MaskedArray, MaskedArray, MaskedArray]:
    """Model head (hros), backing (bros), and C6 surface (sros) rate of spread.

    For C6, hros/bros are the surface heading/backing rates; for all other fuel
    types they are the overall heading/backing rates. ``sros`` is passed in as the
    initialized template (only C6 cells are written).

    :return: (hros, bros, sros)
    """
    # Initialize hfros and bros
    hros = rsi * be
    bros = brsi * be

    # Special handling for C6 (fuel_type == 6)
    is_c6 = fuel_type == 6
    sros = mask.where(is_c6, rsi * be, sros)

    # D2 correction: zero out if BUI < 70, then scale by 0.2
    is_d2 = fuel_type == 9
    hros = mask.where(
        is_d2,
        mask.where(bui < 70, 0.0, hros * 0.2),
        hros
    )
    bros = mask.where(
        is_d2,
        mask.where(bui < 70, 0.0, bros * 0.2),
        bros
    )

    return hros, bros, sros
