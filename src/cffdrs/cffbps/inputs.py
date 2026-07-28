"""Input validation and coercion for CFFBPS.

Converts the raw ``FBP.initialize`` arguments (scalars or ndarrays) into validated
masked arrays and the reference arrays that shape every downstream output. Pure
functions: they read only their arguments and the read-only tables in
``cffbps.constants``, returning new values (no mutation of inputs).
"""
from __future__ import annotations

from datetime import datetime as dt
from operator import itemgetter
from typing import NamedTuple

import numpy as np
from numpy import ma as mask

from . import constants
from .constants import fbpFTCode_AlphaToNum_LUT

MaskedArray = mask.MaskedArray


class VerifiedInputs(NamedTuple):
    """Validated, masked-array-coerced FBP inputs.

    Field names match the ``FBP`` facade attributes they populate.
    """
    fuel_type: MaskedArray
    ftypes: list
    lat: MaskedArray
    long: MaskedArray
    elevation: MaskedArray
    slope: MaskedArray
    aspect: MaskedArray
    ws: MaskedArray
    wd: MaskedArray
    ffmc: MaskedArray
    bui: MaskedArray
    pc: MaskedArray
    pdf: MaskedArray
    gfl: MaskedArray
    gcf: MaskedArray
    d0: MaskedArray | None
    dj: MaskedArray | None


def convert_grid_codes(fuel_type_array: np.ndarray) -> np.ndarray:
    """
    Function to convert grid code values from the cffdrs_r R package fuel type codes
    to the codes used in this module.

    :param fuel_type_array: CFFBPS fuel type array, containing the CFS cffdrs R version of grid codes
    :return: modified fuel_type_array
    """
    fuel_type_array = mask.where(
        fuel_type_array == 19, 20,
        mask.where(
            fuel_type_array == 13, 19,
            mask.where(
                fuel_type_array == 12, 13,
                mask.where(
                    fuel_type_array == 11, 12,
                    mask.where(
                        fuel_type_array == 10, 11,
                        mask.where(fuel_type_array == 9, 10, fuel_type_array)
                    )
                )
            )
        )
    )
    return fuel_type_array


def check_array(input_list: list) -> tuple[bool, mask.MaskedArray, mask.MaskedArray]:
    """Detect array inputs and build the float/int reference arrays.

    :param input_list: raw inputs in fixed order, with fuel_type first
        [fuel_type, lat, long, elevation, slope, aspect, ws, wd, ffmc, bui, pc, pdf, gfl, gcf]
    :return: (return_array, ref_array, ref_int_array)
    """
    if any(isinstance(data, np.ndarray) for data in input_list):
        return_array = True

        # Get indices of input parameters that are arrays
        array_indices = [i for i in range(len(input_list)) if isinstance(input_list[i], np.ndarray)]

        # If more than one array, verify they are all the same shape
        if len(array_indices) > 1:
            # Verify all arrays have the same shape
            arrays = itemgetter(*array_indices)(input_list)
            # Ensure the result is a list
            if isinstance(arrays, np.ndarray):  # Single array case
                arrays = [arrays]
            shapes = {arr.shape for arr in arrays}
            if len(shapes) > 1:
                raise ValueError(f'All arrays must have the same dimensions. '
                                 f'The following range of dimensions exists: {shapes}')

        # Get first input array as a masked array
        first_array = input_list[array_indices[0]]
        if (array_indices[0] == 0) and ('<U' in str(first_array.dtype)):
            # Convert the string representations to numeric codes using the lookup table
            convert_to_numeric = np.vectorize(fbpFTCode_AlphaToNum_LUT.get)
            converted_fuel_type = convert_to_numeric(input_list[0])
            if None in converted_fuel_type:
                raise ValueError('Unknown fuel type code found, conversion failed.')
            first_array = converted_fuel_type.astype(np.int8)

        ref_array = mask.array(
            np.full(first_array.shape, 0, dtype=np.float64),
            mask=np.isnan([first_array]),
            fill_value=np.nan
        )

        ref_int_array = mask.array(
            np.full(first_array.shape, 0, dtype=np.int8),
            mask=-99
        )
    else:
        return_array = False
        # Get first input parameter array as a masked array
        ref_array = mask.array(
            np.array([0.0], dtype=np.float64),
            mask=False,
            fill_value=np.nan
        )
        ref_int_array = mask.array([0], mask=-99).astype(np.int8)

    return return_array, ref_array, ref_int_array


def _coerce(name: str, value, default: float | None = None) -> MaskedArray:
    """Coerce one numeric input (scalar or ndarray) to a masked array.

    Arrays mask their NaN cells. Scalars are wrapped as 1-element arrays with a NaN
    mask — except when ``default`` is given: then a NaN scalar is replaced by the
    default and wrapped WITHOUT a NaN mask (the historical behavior of the optional
    pc/pdf/gfl/gcf fields, preserved deliberately — the golden snapshots lock it).
    """
    if not isinstance(value, (int, float, np.ndarray)):
        raise TypeError(f'{name} must be either int, float, or numpy ndarray data types')
    if isinstance(value, np.ndarray):
        return mask.array(value, mask=np.isnan(value))
    if default is not None:
        if np.isnan(value):
            value = default
        return mask.array([value])
    return mask.array([value], mask=np.isnan([value]))


def verify_inputs(*,
                  fuel_type: int | str | np.ndarray,
                  wx_date: int,
                  lat, long, elevation, slope, aspect, ws, wd, ffmc, bui,
                  pc, pdf, gfl, gcf, d0, dj, out_request,
                  convert_fuel_type_codes: bool) -> VerifiedInputs:
    """Validate all inputs and coerce them to masked numpy arrays.

    :return: :class:`VerifiedInputs` with the coerced fields plus ``ftypes``
        (unique valid fuel-type codes present).
    """
    # ### VERIFY ALL INPUTS AND CONVERT TO MASKED NUMPY ARRAYS
    # Verify fuel_type
    if not isinstance(fuel_type, (int, str, np.ndarray)):
        raise TypeError('fuel_type must be either int, string, or numpy ndarray data types')
    elif isinstance(fuel_type, np.ndarray):
        if '<U' in str(fuel_type.dtype):
            invalid_value = np.int8(-128)  # Define an explicit invalid value
            # Convert using np.vectorize and replace unknown fuel types with invalid_value
            convert_to_numeric = np.vectorize(lambda x: fbpFTCode_AlphaToNum_LUT.get(x, invalid_value))
            fuel_type = convert_to_numeric(fuel_type).astype(np.int8)
        if fuel_type.dtype != np.int8:
            fuel_type = np.asarray(fuel_type, dtype=np.int8)
        fuel_type = mask.array(fuel_type, mask=np.isnan(fuel_type))
    elif isinstance(fuel_type, str):
        fuel_type = mask.array([fbpFTCode_AlphaToNum_LUT.get(fuel_type)],
                               mask=np.isnan([fbpFTCode_AlphaToNum_LUT.get(fuel_type)]))
    else:
        fuel_type = mask.array([fuel_type], mask=np.isnan([fuel_type]))

    # Convert from cffdrs R fuel type grid codes to the grid codes used in this module
    if convert_fuel_type_codes:
        fuel_type = convert_grid_codes(fuel_type)

    # Apply an additional mask to exclude invalid fuel types (valid range: 1-20)
    valid_fuel_types = np.arange(1, 21, dtype=np.int8)
    current_mask = np.ma.getmaskarray(fuel_type)
    invalid_mask = ~np.isin(fuel_type.data, valid_fuel_types)
    fuel_type = mask.array(fuel_type.data, mask=(current_mask | invalid_mask))

    # Get unique fuel types present in the dataset
    ftypes = [ftype for ftype in np.unique(fuel_type) if ftype in list(constants.rosParams.keys())]

    # Verify wx_date
    if not isinstance(wx_date, int):
        raise TypeError('wx_date must be int data type')
    try:
        date_string = str(wx_date)
        dt.fromisoformat(f'{date_string[:4]}-{date_string[4:6]}-{date_string[6:]}')
    except ValueError:
        raise ValueError('wx_date must be formatted as: YYYYMMDD') from None

    # Coerce every numeric input to a masked array (one shared helper; see _coerce)
    lat = _coerce('lat', lat)
    long = np.absolute(_coerce('long', long))  # Get absolute longitude values
    elevation = _coerce('elevation', elevation)
    slope = mask.clip(_coerce('slope', slope), 0, None)  # Limit the lower slope value to 0
    aspect = _coerce('aspect', aspect)
    # Set negative aspect values to 270 degrees (assuming they represent "flat" terrain)
    aspect = mask.where(aspect < 0, 270, aspect)
    ws = _coerce('ws', ws)
    ws = mask.where(ws < 0, 0, ws)  # Set to 0 if negative
    wd = _coerce('wd', wd)
    ffmc = _coerce('ffmc', ffmc)
    ffmc = mask.where(ffmc < 0, 0, ffmc)  # Set to 0 if negative
    bui = _coerce('bui', bui)
    bui = mask.where(bui < 0, 0, bui)  # Set to 0 if negative
    pc = _coerce('pc', pc, default=50)  # Default to 50% if NaN
    pc = mask.where(pc < 0, 0, pc)  # Set to 0 if negative
    pdf = _coerce('pdf', pdf, default=35)  # Default to 35% if NaN
    pdf = mask.where(pdf < 0, 0, pdf)  # Set to 0 if negative
    gfl = _coerce('gfl', gfl, default=0.35)  # Default to 0.35 kg/m2 if NaN
    gfl = mask.where(gfl < 0, 0, gfl)  # Set to 0 if negative
    gcf = _coerce('gcf', gcf, default=80)  # Default to 80% if NaN
    gcf = mask.where(gcf == 0, 0.1, gcf)  # Set to 0.1% if 0%

    # Verify d0
    if d0 is not None:
        if not isinstance(d0, int):
            raise TypeError('d0 must be int data type')
        else:
            d0 = mask.array([d0], mask=np.isnan([d0]))

    # Verify dj
    if dj is not None:
        if not isinstance(dj, int):
            raise TypeError('dj must be int data type')
        else:
            dj = mask.array([dj], mask=np.isnan([dj]))

    # Verify out_request
    if not isinstance(out_request, (list, tuple, type(None))):
        raise TypeError('out_request must be a list, tuple, or None')

    return VerifiedInputs(
        fuel_type=fuel_type, ftypes=ftypes,
        lat=lat, long=long, elevation=elevation, slope=slope, aspect=aspect,
        ws=ws, wd=wd, ffmc=ffmc, bui=bui,
        pc=pc, pdf=pdf, gfl=gfl, gcf=gcf, d0=d0, dj=dj,
    )
