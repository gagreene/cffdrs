"""Crown fire equations: CBH/CFL, CSFI, RSO, CFB, fire type, and CFC."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy import ma as mask

MaskedArray = mask.MaskedArray


def calc_cbh_cfl(*,
                 fuel_type: MaskedArray,
                 cbh: MaskedArray,
                 cfl: MaskedArray,
                 cbh_cfl_ht_lut: Mapping[int, tuple],
                 ftype: int | None = None,
                 cbh_override: float | None = None,
                 cfl_override: float | None = None) -> tuple[MaskedArray, MaskedArray]:
    """Populate canopy base height (CBH) and canopy fuel load (CFL) per fuel type.

    ``cbh``/``cfl`` are the current template arrays; they are copied, populated, and
    the copies returned (arguments are never mutated). When ``ftype`` is given, only
    that fuel type is set (with optional C6-only overrides); otherwise every present
    fuel type is filled from the table.

    :return: (cbh, cfl) as new arrays
    """
    # Own-and-return: never write into the caller's arrays.
    cbh = cbh.copy()
    cfl = cfl.copy()

    if ftype is not None:
        ftype_mask = fuel_type == ftype
        if cbh_override is None:
            # Indexed lookup: an unknown fuel-type code fails with a KeyError naming it
            # (nothing upstream validates an explicitly passed ftype).
            cbh_val = cbh_cfl_ht_lut[ftype][0]
        else:
            if not isinstance(cbh_override, float):
                raise ValueError('The "cbh" parameter must be a float data type.')
            if ftype != 6:
                raise ValueError('Only the C-6 fuel type can have the cbh value adjusted.')
            cbh_val = cbh_override
        cbh[ftype_mask] = cbh_val

        # Get canopy fuel load (CFL) for fuel type
        if cfl_override is None:
            cfl_val = cbh_cfl_ht_lut[ftype][1]
        else:
            if not isinstance(cfl_override, float):
                raise ValueError('The "cfl" parameter must be a float data type.')
            if ftype != 6:
                raise ValueError('Only the C-6 fuel type can have the cfl value adjusted.')
            cfl_val = cfl_override
        cfl[ftype_mask] = cfl_val
    else:
        for ft in mask.unique(fuel_type[~fuel_type.mask]):
            ftype_mask = fuel_type == ft
            cbh[ftype_mask], cfl[ftype_mask] = cbh_cfl_ht_lut[ft][:2]

    return cbh, cfl


def calc_csfi(*, fuel_type: MaskedArray, cbh: MaskedArray, fmc: MaskedArray) -> MaskedArray:
    """Calculate the critical surface fire intensity (CSFI)."""
    return mask.where(fuel_type < 14,
                      np.power(0.01 * cbh * (460 + (25.9 * fmc)), 1.5),
                      0)


def calc_rso(*, sfc: MaskedArray, csfi: MaskedArray) -> MaskedArray:
    """Calculate the critical surface fire rate of spread (RSO)."""
    return mask.where(sfc > 0,
                      csfi / (300.0 * sfc),
                      0)


def calc_cfb(*,
             fuel_type: MaskedArray,
             ftypes: Sequence[int],
             non_crowning_fuels: Sequence[int],
             rso: MaskedArray,
             ros: MaskedArray) -> MaskedArray:
    """Calculate directional CFB from completed ROS.

    The same equation applies to every crowning fuel type, including C6. The
    C6-specific SROS-derived value needed by the deterministic blend is
    calculated separately by :func:`calc_c6_blend_cfb`.
    """
    crowning = np.isin(fuel_type, ftypes) & ~np.isin(fuel_type, non_crowning_fuels)
    cfb = mask.where(crowning, _calc_cfb_from_ros(ros=ros, rso=rso), 0)
    return _sanitize_cfb(cfb)


def calc_c6_blend_cfb(*,
                      fuel_type: MaskedArray,
                      sros: MaskedArray,
                      rso: MaskedArray) -> MaskedArray:
    """Calculate the temporary SROS-derived CFB used only by the C6 ROS blend."""
    cfb = mask.where(fuel_type == 6, _calc_cfb_from_ros(ros=sros, rso=rso), 0)
    return _sanitize_cfb(cfb)


def _calc_cfb_from_ros(*, ros: MaskedArray, rso: MaskedArray) -> MaskedArray:
    """Apply the CFB equation to one directional ROS array."""
    delta_ros = ros - rso
    with np.errstate(over='ignore'):
        return mask.where(delta_ros < -3086, 0, 1 - np.exp(-0.23 * delta_ros))


def _sanitize_cfb(cfb: MaskedArray) -> MaskedArray:
    """Replace non-finite CFB values and constrain the result to [0, 1]."""
    cfb = mask.where(np.isfinite(cfb), cfb, 0)
    return mask.clip(cfb, 0, 1)


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


def calc_cfc(*,
             fuel_type: MaskedArray,
             cfb: MaskedArray,
             cfl: MaskedArray,
             pc: MaskedArray,
             pdf: MaskedArray) -> MaskedArray:
    """Calculate crown fuel consumed (kg/m^2)."""
    return mask.where((fuel_type == 10) | (fuel_type == 11),
                      cfb * cfl * pc / 100,
                      mask.where((fuel_type == 12) | (fuel_type == 13),
                                 cfb * cfl * pdf / 100,
                                 cfb * cfl))
