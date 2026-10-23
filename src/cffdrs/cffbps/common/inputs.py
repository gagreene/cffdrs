"""Input validation and coercion for CFFBPS.

Converts the raw ``FBP.initialize`` arguments (scalars or ndarrays) into validated
masked arrays and the reference arrays that shape every downstream output. Pure
functions: they read only their arguments and the read-only tables in
``cffdrs.cffbps.common.constants``, returning new values (no mutation of inputs).
"""
from __future__ import annotations

from datetime import datetime as dt
from operator import itemgetter
from typing import NamedTuple

import numpy as np
from numpy import ma as mask

from . import constants
from .constants import fbpFTCode_AlphaToNum_LUT
from .fmc_inputs import coerce_override

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
    fmc: MaskedArray | None


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


def prepare_fuel_type(fuel_type, convert_fuel_type_codes: bool):
    """Validate a fuel-type input and return ``(masked int8 codes, unique valid modelled codes)``."""
    if not isinstance(fuel_type, (int, str, np.ndarray)):
        raise TypeError('fuel_type must be either int, string, or numpy ndarray data types')
    elif isinstance(fuel_type, np.ndarray):
        if '<U' in str(fuel_type.dtype):
            invalid_value = np.int8(-128)  # Define an explicit invalid value
            # Convert using np.vectorize and replace unknown fuel types with invalid_value
            convert_to_numeric = np.vectorize(lambda x: fbpFTCode_AlphaToNum_LUT.get(x, invalid_value))
            fuel_type = convert_to_numeric(fuel_type).astype(np.int8)
        arr = mask.asarray(fuel_type)
        data, was_masked = mask.getdata(arr), mask.getmaskarray(arr)
        valid = (data >= 1) & (data <= 20)
        if data.dtype.kind == 'f':
            valid &= data == np.floor(data)
        fuel_type = mask.array(np.where(valid & ~was_masked, data, 0).astype(np.int8),
                               mask=was_masked | ~valid)
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
    return fuel_type, ftypes


def validate_wx_date(wx_date) -> int:
    """Require ``wx_date`` to be an int formatted YYYYMMDD and return it."""
    if not isinstance(wx_date, int):
        raise TypeError('wx_date must be int data type')
    try:
        date_string = str(wx_date)
        dt.fromisoformat(f'{date_string[:4]}-{date_string[4:6]}-{date_string[6:]}')
    except ValueError:
        raise ValueError('wx_date must be formatted as: YYYYMMDD') from None
    return wx_date


# Inputs that are normalised when they are set: name -> default substituted for a NaN scalar (or None)
NORMALISED_INPUTS = {'slope': None, 'aspect': None, 'ws': None, 'ffmc': None, 'bui': None,
                     'pc': 50, 'pdf': 35, 'gfl': 0.35, 'gcf': 80}


def normalize_input(name: str, value) -> MaskedArray:
    """Coerce one of the nine normalised inputs to a masked array and apply its rule.

    Used by ``verify_inputs`` and by every route that changes an input after initialisation, so an
    updated model equals one initialised with the same value. Rules: ``slope`` is limited below at 0;
    a negative ``aspect`` becomes 270 (flat terrain); negative ``ws``, ``ffmc``, ``bui``, ``pc``, ``pdf``
    and ``gfl`` become 0; ``gcf`` of 0 becomes 0.1. A NaN scalar ``pc``/``pdf``/``gfl``/``gcf`` takes its default.
    """
    if name not in NORMALISED_INPUTS:
        raise ValueError(f'{name} is not a normalised input')
    value = _coerce(name, value, default=NORMALISED_INPUTS[name])
    if name == 'slope':
        return mask.clip(value, 0, None)  # Limit the lower slope value to 0
    if name == 'aspect':
        return mask.where(value < 0, 270, value)  # Negative aspect represents "flat" terrain
    if name == 'gcf':
        return mask.where(value == 0, 0.1, value)  # Set to 0.1% if 0%
    return mask.where(value < 0, 0, value)  # Set to 0 if negative


def coerce_input(name: str, value, default: float | None = None) -> MaskedArray:
    """Public form of the shared numeric-input coercion (see ``_coerce``)."""
    return _coerce(name, value, default)


def verify_inputs(*,
                  fuel_type: int | str | np.ndarray,
                  wx_date: int,
                  lat, long, elevation, slope, aspect, ws, wd, ffmc, bui,
                  pc, pdf, gfl, gcf, d0, dj, out_request,
                  convert_fuel_type_codes: bool, fmc=None) -> VerifiedInputs:
    """Validate all inputs and coerce them to masked numpy arrays.

    :return: :class:`VerifiedInputs` with the coerced fields plus ``ftypes``
        (unique valid fuel-type codes present).
    """
    # ### VERIFY ALL INPUTS AND CONVERT TO MASKED NUMPY ARRAYS
    # Verify fuel_type
    fuel_type, ftypes = prepare_fuel_type(fuel_type, convert_fuel_type_codes)

    # Verify wx_date
    validate_wx_date(wx_date)

    # Coerce every numeric input to a masked array (one shared helper; see _coerce) and apply the
    # normalisation rules (one shared helper; see normalize_input)
    lat = _coerce('lat', lat)
    long = np.absolute(_coerce('long', long))  # Get absolute longitude values
    elevation = _coerce('elevation', elevation)
    slope = normalize_input('slope', slope)
    aspect = normalize_input('aspect', aspect)
    ws = normalize_input('ws', ws)
    wd = _coerce('wd', wd)
    ffmc = normalize_input('ffmc', ffmc)
    bui = normalize_input('bui', bui)
    pc = normalize_input('pc', pc)
    pdf = normalize_input('pdf', pdf)
    gfl = normalize_input('gfl', gfl)
    gcf = normalize_input('gcf', gcf)

    # Verify d0 and dj (scalars, arrays or masked arrays; NaN and masked cells are missing; no range limit on
    # d0, dj is floored and clipped to the year at calculation time, see fmc_inputs)
    d0 = coerce_override('d0', d0)
    dj = coerce_override('dj', dj)
    fmc = coerce_override('fmc', fmc)

    # Verify out_request
    if not isinstance(out_request, (list, tuple, type(None))):
        raise TypeError('out_request must be a list, tuple, or None')

    return VerifiedInputs(
        fuel_type=fuel_type, ftypes=ftypes,
        lat=lat, long=long, elevation=elevation, slope=slope, aspect=aspect,
        ws=ws, wd=wd, ffmc=ffmc, bui=bui,
        pc=pc, pdf=pdf, gfl=gfl, gcf=gcf, d0=d0, dj=dj, fmc=fmc,
    )


#: The inputs and overrides that carry the fuel mask: a cell whose fuel is masked or invalid is missing in all of them.
#: ``pc``, ``pdf``, ``gfl`` and ``gcf`` only feed fuel-dependent results, which the fuel mask already masks, so they are
#: left alone (masking them would only add four full-size masks and slow every masked-array operation).
FUEL_MASKED_NAMES = ('lat', 'long', 'elevation', 'slope', 'aspect', 'ws', 'wd', 'ffmc', 'bui', 'd0', 'dj', 'fmc')


def fuel_mask_of(fuel_type) -> np.ndarray | None:
    """The mask of a prepared fuel grid, or ``None`` when no cell is masked (the common case costs nothing)."""
    if fuel_type is None:
        return None
    fuel_mask = np.ma.getmaskarray(fuel_type)
    return fuel_mask if fuel_mask.any() else None


def apply_fuel_mask(value, fuel_mask):
    """``value`` with the cells of ``fuel_mask`` masked as well; the data are shared, only the mask is new.

    A value of one cell (a scalar input) is broadcast to the grid of the fuel mask. With no fuel mask, or for ``None``,
    ``value`` is returned as it is.
    """
    if value is None or fuel_mask is None:
        return value
    data, missing = np.ma.getdata(value), np.ma.getmaskarray(value)
    shape = np.broadcast_shapes(data.shape, fuel_mask.shape)
    if data.shape != shape:
        data, missing = np.broadcast_to(data, shape), np.broadcast_to(missing, shape)
    return mask.array(data, mask=missing | fuel_mask, copy=False)


class PreparedInputs(NamedTuple):
    """Everything ``initialize`` needs from the raw arguments (shared by every backend).

    ``verified`` already carries the fuel mask; ``raw`` holds the same inputs as the user gave them (before the fuel
    mask), so a later change of the fuel grid can mask and unmask the right cells.
    """
    verified: VerifiedInputs
    return_array: bool
    ref_array: MaskedArray
    ref_int_array: MaskedArray
    raw: dict


REQUIRED_INPUTS = ('fuel_type', 'wx_date', 'lat', 'long', 'elevation', 'slope', 'aspect', 'ws', 'wd', 'ffmc', 'bui')


def prepare_inputs(*, fuel_type, wx_date, lat, long, elevation, slope, aspect, ws, wd,
                   ffmc, bui, pc, pdf, gfl, gcf, d0, dj, out_request,
                   convert_fuel_type_codes, fmc=None) -> PreparedInputs:
    """Check the required parameters, detect array input and validate/coerce every input.

    :raises ValueError: listing every missing required parameter
    """
    given = dict(fuel_type=fuel_type, wx_date=wx_date, lat=lat, long=long, elevation=elevation, slope=slope,
                 aspect=aspect, ws=ws, wd=wd, ffmc=ffmc, bui=bui)
    missing = [name for name in REQUIRED_INPUTS if given[name] is None]
    if missing:
        raise ValueError(f'Missing required parameters: {missing}')
    return_array, ref_array, ref_int_array = check_array(
        [fuel_type, lat, long, elevation, slope, aspect, ws, wd, ffmc, bui, pc, pdf, gfl, gcf])
    verified = verify_inputs(
        fuel_type=fuel_type, wx_date=wx_date, lat=lat, long=long, elevation=elevation, slope=slope,
        aspect=aspect, ws=ws, wd=wd, ffmc=ffmc, bui=bui, pc=pc, pdf=pdf, gfl=gfl, gcf=gcf,
        d0=d0, dj=dj, fmc=fmc, out_request=out_request, convert_fuel_type_codes=convert_fuel_type_codes)
    raw = {name: getattr(verified, name) for name in FUEL_MASKED_NAMES}
    fuel_mask = fuel_mask_of(verified.fuel_type)
    if fuel_mask is not None:
        verified = verified._replace(**{name: apply_fuel_mask(value, fuel_mask) for name, value in raw.items()})
    return PreparedInputs(verified, return_array, ref_array, ref_int_array, raw)
