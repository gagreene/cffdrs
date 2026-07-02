"""Crown fire equations: CBH/CFL, CSFI, RSO, CFB, fire type, CFC, C6 head ROS."""
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
             sros: MaskedArray,
             rso: MaskedArray,
             hros: MaskedArray) -> MaskedArray:
    """Calculate crown fraction burned (Forestry Canada Fire Danger Group 1992)."""
    # Initialize CFB array
    cfb = np.full_like(fuel_type, 0, dtype=np.float64)

    with np.errstate(over='ignore'):
        # Create masks for C-6 and other fuel types
        is_c6 = mask.where(fuel_type == 6, True, False)
        non_crowning = mask.where(np.isin(fuel_type, non_crowning_fuels), True, False)
        is_other = mask.where(np.isin(fuel_type, ftypes) & ~is_c6 & ~non_crowning, True, False)

        # Precompute rate of spread differences
        delta_sros_c6 = sros - rso
        delta_hros_other = hros - rso

        # Compute CFB for C-6 and other fuel types
        cfb_c6 = mask.where(delta_sros_c6 < -3086, 0, 1 - np.exp(-0.23 * delta_sros_c6))
        cfb_other = mask.where(delta_hros_other < -3086, 0, 1 - np.exp(-0.23 * delta_hros_other))

        # Apply the calculations
        cfb = mask.where(is_c6, cfb_c6, cfb)
        cfb = mask.where(is_other, cfb_other, cfb)

        # Ensure cfb is finite and ranges between 0 and 1
        is_finite = mask.where(np.isfinite(cfb), True, False)
        cfb = mask.where(is_finite, cfb, 0)  # Replace NaNs/Infs with 0
        cfb = mask.clip(cfb, 0, 1)  # Prevent extremely high values causing overflow

    return cfb


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


def calc_c6hros(*,
                fuel_type: MaskedArray,
                cfc: MaskedArray,
                isi: MaskedArray,
                fme: MaskedArray,
                cros: MaskedArray,
                sros: MaskedArray,
                cfb: MaskedArray,
                hros: MaskedArray) -> tuple[MaskedArray, MaskedArray]:
    """Calculate crown and total head fire rate of spread for the C6 fuel type.

    ``cros``/``hros`` are passed in as current values (only C6 cells are updated).

    :return: (cros, hros)
    """
    cros = mask.where(fuel_type == 6,
                      mask.where(cfc == 0,
                                 0,
                                 60 * np.power(1 - np.exp(-0.0497 * isi), 1) * (fme / 0.778237)),
                      cros)

    hros = mask.where(fuel_type == 6,
                      sros + (cfb * (cros - sros)),
                      hros)

    return cros, hros
