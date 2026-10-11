"""Fire rate-of-spread equations."""
from __future__ import annotations

import numpy as np
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


def calc_c6_cros(*,
                 fuel_type: MaskedArray,
                 cfc: MaskedArray,
                 isi: MaskedArray,
                 fme: MaskedArray,
                 cros: MaskedArray) -> MaskedArray:
    """Calculate C6 crown-fire ROS.

    ``cfc`` is the temporary C6 crown fuel consumption derived from the
    SROS-based blend CFB. ``cros`` is the initialized/current array; only C6
    cells are replaced.
    """
    return mask.where(
        fuel_type == 6,
        mask.where(
            cfc == 0,
            0,
            60 * (1 - np.exp(-0.0497 * isi)) * (fme / 0.778237),
        ),
        cros,
    )


def calc_c6_hros(*,
                 fuel_type: MaskedArray,
                 sros: MaskedArray,
                 cros: MaskedArray,
                 c6_blend_cfb: MaskedArray,
                 hros: MaskedArray) -> MaskedArray:
    """Blend C6 surface and crown ROS into deterministic heading ROS.

    ``c6_blend_cfb`` is calculated from SROS solely for this blend. ``hros`` is
    the current array; only C6 cells are replaced.
    """
    return mask.where(
        fuel_type == 6,
        sros + c6_blend_cfb * (cros - sros),
        hros,
    )
