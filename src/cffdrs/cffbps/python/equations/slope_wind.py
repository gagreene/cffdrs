"""Slope- and wind-related CFFBPS equations.

Wind/aspect inversion, slope factor, zero-wind/zero-slope ISI, the wind/slope
adjustment of ISI, and the fuel-type-specific RSI and BUI buildup effect (BE).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
from numpy import ma as mask

MaskedArray = mask.MaskedArray


class SlopeWindISI(NamedTuple):
    """Slope-equivalent wind, net wind vectors, and wind-adjusted ISI."""
    wse1: MaskedArray
    wse2: MaskedArray
    wse: MaskedArray
    wsx: MaskedArray
    wsy: MaskedArray
    wsv: MaskedArray
    raz: MaskedArray
    fW: MaskedArray
    bfW: MaskedArray
    isi: MaskedArray
    bisi: MaskedArray


class ISIRSIBEResult(NamedTuple):
    """Everything derived by :func:`calc_isi_rsi_be`.

    Field names match the ``FBP`` facade attributes they populate.
    """
    a: MaskedArray
    b: MaskedArray
    c: MaskedArray
    q: MaskedArray
    bui0: MaskedArray
    be_max: MaskedArray
    rsz: MaskedArray
    rsf: MaskedArray
    isf: MaskedArray
    rsi: MaskedArray
    brsi: MaskedArray
    be: MaskedArray
    wse1: MaskedArray
    wse2: MaskedArray
    wse: MaskedArray
    wsx: MaskedArray
    wsy: MaskedArray
    wsv: MaskedArray
    raz: MaskedArray
    fW: MaskedArray
    bfW: MaskedArray
    isi: MaskedArray
    bisi: MaskedArray


def invert_wind_aspect(wd: MaskedArray, aspect: MaskedArray) -> tuple[MaskedArray, MaskedArray]:
    """Invert/flip wind direction and aspect by 180 degrees.

    :return: (wd, aspect) inverted
    """
    wd = mask.where(wd > 180, wd - 180, wd + 180)
    aspect = mask.where(aspect > 180, aspect - 180, aspect + 180)
    return wd, aspect


def calc_sf(slope: MaskedArray) -> MaskedArray:
    """Calculate the slope factor (SF)."""
    return mask.where(slope < 70,
                      np.exp(3.533 * np.power((slope / 100), 1.2)),
                      10)


def calc_isz(ffmc: MaskedArray) -> tuple[MaskedArray, MaskedArray, MaskedArray]:
    """Calculate the no-wind/no-slope Initial Spread Index.

    :return: (m, fF, isz) — fine fuel moisture content (%), the FFMC function in the
        ISI equation, and the zero-wind/zero-slope ISI.
    """
    with np.errstate(invalid='ignore'):
        # Fine fuel moisture content in percent (default CFFBPS equation)
        m = (250 * (59.5 / 101) * (101 - ffmc)) / (59.5 + ffmc)

        # FFMC function from the ISI equation (fF)
        fF = (91.9 * np.exp(-0.1386 * m)) * (1 + (np.power(m, 5.31) / (4.93 * np.power(10, 7))))

        # No slope/no wind Initial Spread Index
        isz = 0.208 * fF

    return m, fF, isz


def calc_slope_wind_isi(*,
                        isf: MaskedArray,
                        fF: MaskedArray,
                        wd: MaskedArray,
                        aspect: MaskedArray,
                        ws: MaskedArray) -> SlopeWindISI:
    """Compute slope-equivalent wind, net wind vectors, RAZ, and wind-adjusted ISI."""
    with np.errstate(invalid='ignore', divide='ignore'):
        # Calculate slope-equivalent wind speeds using two formulas
        wse1 = (1 / 0.05039) * mask.log(isf / (0.208 * fF))
        wse2 = mask.where(
            isf < 0.999 * 2.496 * fF,
            28 - (1 / 0.0818) * np.log(1 - (isf / (2.496 * fF))),
            112.45  # cap maximum WSE
        )

        # Assign slope equivalent wind speed
        wse = mask.where(wse1 <= 40, wse1, wse2)

        # Compute directional components for wind and slope
        sin_wd = mask.sin(np.radians(wd))
        cos_wd = mask.cos(np.radians(wd))
        sin_asp = mask.sin(np.radians(aspect))
        cos_asp = mask.cos(np.radians(aspect))

        # Net wind vectors
        wsx = ws * sin_wd + wse * sin_asp
        wsy = ws * cos_wd + wse * cos_asp
        wsv = mask.sqrt(wsx ** 2 + wsy ** 2)

        # Wind azimuth calculation (RAZ)
        acos_val = mask.clip(wsy / wsv, -1, 1)
        angle_rad = mask.arccos(acos_val)
        raz = mask.where(
            wsx < 0,
            360 - np.degrees(angle_rad),
            np.degrees(angle_rad)
        )

        # When wsv == 0 (calm wind on flat ground, or exact wind/slope cancellation) the
        # azimuth is undefined. Spread is circular in this state (lb_ratio == 1),
        # so the direction is immaterial — substitute 0° to keep raz finite for downstream consumers.
        raz = mask.where(wsv > 0, raz, 0.0)

        # Compute head fire and backfire wind function
        fW = mask.where(
            wsv > 40,
            12 * (1 - np.exp(-0.0818 * (wsv - 28))),
            np.exp(0.05039 * wsv)
        )
        bfW = mask.exp(-0.05039 * wsv)

        # Final head fire and backfire ISI
        isi = 0.208 * fF * fW
        bisi = 0.208 * fF * bfW

    return SlopeWindISI(wse1=wse1, wse2=wse2, wse=wse, wsx=wsx, wsy=wsy, wsv=wsv,
                        raz=raz, fW=fW, bfW=bfW, isi=isi, bisi=bisi)


def calc_isi_rsi_be(*,
                    fuel_type: MaskedArray,
                    ros_params: Mapping[int, tuple],
                    gcf: MaskedArray,
                    isz: MaskedArray,
                    sf: MaskedArray,
                    pc: MaskedArray,
                    pdf: MaskedArray,
                    bui: MaskedArray,
                    fF: MaskedArray,
                    wd: MaskedArray,
                    aspect: MaskedArray,
                    ws: MaskedArray,
                    ref_array: MaskedArray) -> ISIRSIBEResult:
    """Compute slope-/wind-adjusted ISI, rate of spread (RSI), and BUI effect (BE).

    Allocates the per-cell fuel-type coefficient arrays (a, b, c, q, bui0, be_max)
    from ``ref_array`` (the zero-filled masked template), then derives RSZ/RSF/ISF,
    the wind-adjusted ISI (via :func:`calc_slope_wind_isi`), RSI/BRSI, and BE.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        # Generate mixed-wood and grass masks
        m12_mask = (fuel_type == 10) | (fuel_type == 11)
        m34_mask = (fuel_type == 12) | (fuel_type == 13)
        o1_mask = (fuel_type == 14) | (fuel_type == 15)

        # Precompute C2 and D1 parameters
        c2 = ros_params[2]
        d1 = ros_params[8]

        # Allocate and assign fuel-type-specific parameters
        a = ref_array.copy()
        b = ref_array.copy()
        c = ref_array.copy()
        q = ref_array.copy()
        bui0 = ref_array.copy()
        be_max = ref_array.copy()
        for ftype in mask.unique(fuel_type[~fuel_type.mask]):
            a_val, b_val, c_val, q_val, bui0_val, be_max_val = ros_params.get(ftype, (0, 0, 0, 0, 1, 1))
            ft_mask = (fuel_type == ftype)
            a[ft_mask] = a_val
            b[ft_mask] = b_val
            c[ft_mask] = c_val
            q[ft_mask] = q_val
            bui0[ft_mask] = bui0_val
            be_max[ft_mask] = be_max_val

        # Handle O1a/b (ftype 14 and 15) curing factor logic
        cf = mask.where(
            gcf < 58.8,
            0.005 * (np.exp(0.061 * gcf) - 1),
            0.176 + 0.02 * (gcf - 58.8)
        )

        # Compute RSZ
        rsz_core = a * np.power(1 - np.exp(-b * isz), c)
        # M1/2
        rsz_c2 = c2[0] * np.power(1 - np.exp(-c2[1] * isz), c2[2])
        rsz_d1 = d1[0] * np.power(1 - np.exp(-d1[1] * isz), d1[2])
        rsz_m1 = (pc / 100) * rsz_c2 + (1 - pc / 100) * rsz_d1
        rsz_m2 = (pc / 100) * rsz_c2 + 0.2 * (1 - pc / 100) * rsz_d1
        # O1a/b
        rsz_o1 = rsz_core * cf
        # Final calculation
        rsz = mask.where(fuel_type == 10, rsz_m1, rsz_core)
        rsz = mask.where(fuel_type == 11, rsz_m2, rsz)
        rsz = mask.where(o1_mask, rsz_o1, rsz)

        # Compute RSF
        rsf_c2 = rsz_c2 * sf
        rsf_d1 = rsz_d1 * sf
        rsf = rsz * sf

        # Compute ISF for M1/2 & M3/4 blending logic
        isf_c2_numer = 1 - np.power(rsf_c2 / c2[0], 1 / c2[2])
        isf_d1_numer = 1 - np.power(rsf_d1 / d1[0], 1 / d1[2])
        isf_m34_numer = 1 - np.power(rsf / a, 1 / c)
        isf_c2_core = mask.where(isf_c2_numer >= 0.01, np.log(isf_c2_numer) / -c2[1], np.log(0.01) / -c2[1])
        isf_d1_core = mask.where(isf_d1_numer >= 0.01, np.log(isf_d1_numer) / -d1[1], np.log(0.01) / -d1[1])
        isf_m34_core = mask.where(isf_m34_numer >= 0.01, np.log(isf_m34_numer) / -b, np.log(0.01) / -b)
        isf_blended_m12 = (pc / 100) * isf_c2_core + (1 - pc / 100) * isf_d1_core
        isf_blended_m34 = (pdf / 100) * isf_m34_core + (1 - pdf / 100) * isf_d1_core

        # Compute ISF
        isf_numer = mask.where(o1_mask,
                               1 - np.power(rsf / (a * cf), 1 / c),
                               1 - np.power(rsf / a, 1 / c))
        isf_final = mask.where(isf_numer >= 0.01, np.log(isf_numer) / -b, np.log(0.01) / -b)
        isf = mask.where(m12_mask, isf_blended_m12, isf_final)
        isf = mask.where(m34_mask, isf_blended_m34, isf)

        # Wind and slope adjusted ISI
        sw = calc_slope_wind_isi(isf=isf, fF=fF, wd=wd, aspect=aspect, ws=ws)
        isi = sw.isi
        bisi = sw.bisi

        # Final RSI and BRSI
        rsi_c2 = c2[0] * np.power(1 - np.exp(-c2[1] * isi), c2[2])
        rsi_d1 = d1[0] * np.power(1 - np.exp(-d1[1] * isi), d1[2])
        rsi = mask.where(
            (fuel_type == 12),  # M3
            (pdf / 100) * a * np.power(1 - np.exp(-b * isi), c) +
            (1 - pdf / 100) * rsi_d1,
            mask.where(
                (fuel_type == 13),  # M4
                (pdf / 100) * a * np.power(1 - np.exp(-b * isi), c) +
                0.2 * (1 - pdf / 100) * rsi_d1,
                mask.where(
                    (fuel_type == 10),  # M1
                    (pc / 100) * rsi_c2 + (1 - pc / 100) * rsi_d1,
                    mask.where(
                        (fuel_type == 11),  # M2
                        (pc / 100) * rsi_c2 + 0.2 * (1 - pc / 100) * rsi_d1,
                        mask.where(
                            o1_mask,
                            a * np.power(1 - np.exp(-b * isi), c) * cf,
                            a * np.power(1 - np.exp(-b * isi), c)
                        )
                    )
                )
            )
        )

        brsi_c2 = c2[0] * np.power(1 - np.exp(-c2[1] * bisi), c2[2])
        brsi_d1 = d1[0] * np.power(1 - np.exp(-d1[1] * bisi), d1[2])
        brsi = mask.where(
            (fuel_type == 12),
            (pdf / 100) * a * np.power(1 - np.exp(-b * bisi), c) +
            (1 - pdf / 100) * brsi_d1,
            mask.where(
                (fuel_type == 13),
                (pdf / 100) * a * np.power(1 - np.exp(-b * bisi), c) +
                0.2 * (1 - pdf / 100) * brsi_d1,
                mask.where(
                    (fuel_type == 11),
                    (pc / 100) * brsi_c2 + 0.2 * (1 - pc / 100) * brsi_d1,
                    mask.where(
                        (fuel_type == 10),
                        (pc / 100) * brsi_c2 + (1 - pc / 100) * brsi_d1,
                        mask.where(
                            o1_mask,
                            a * np.power(1 - np.exp(-b * bisi), c) * cf,
                            a * np.power(1 - np.exp(-b * bisi), c)
                        )
                    )
                )
            )
        )

        # Compute BE and clip
        raw_be = mask.where(
            (bui == 0) | ~np.isfinite(bui),
            0.0,
            mask.where(
                (bui0 == 0) | ~np.isfinite(bui0),
                1,
                np.exp(50 * np.log(q) * ((1 / bui) - (1 / bui0)))
            )
        )
        be = mask.clip(raw_be, 0, be_max)

    return ISIRSIBEResult(
        a=a, b=b, c=c, q=q, bui0=bui0, be_max=be_max,
        rsz=rsz, rsf=rsf, isf=isf, rsi=rsi, brsi=brsi, be=be,
        wse1=sw.wse1, wse2=sw.wse2, wse=sw.wse, wsx=sw.wsx, wsy=sw.wsy, wsv=sw.wsv,
        raz=sw.raz, fW=sw.fW, bfW=sw.bfW, isi=sw.isi, bisi=sw.bisi,
    )
