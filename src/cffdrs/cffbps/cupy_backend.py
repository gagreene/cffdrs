# -*- coding: utf-8 -*-
"""Experimental CuPy (GPU) backend for CFFBPS.

This is a near-duplicate of the CFFBPS ``FBP`` model rewritten against CuPy for
GPU acceleration. It is **experimental and unmaintained-for-parity**: it is
frozen at the pre-refactor monolith shape, predates the reference package's
masked-array NaN-propagation fixes, and defaults to ``float32``. It therefore
does **not** reproduce the reference golden snapshots, and there is no parity
test guarding it. The reference implementation is :mod:`cffdrs.cffbps` (the
``FBP`` facade over the pure ``equations`` modules); use this only for
deliberate GPU experimentation.

Requires the optional ``gpu`` extra (CuPy + a CUDA-capable environment):
``pip install cffdrs[gpu]`` (or install the CUDA-matched wheel yourself, e.g.
``cupy-cuda12x``). Importing this module without CuPy raises a clear
:class:`ImportError`.

Created on Thur Dec 26 20:45:00 2024

@author: Gregory A. Greene
"""
__author__ = ['Gregory A. Greene']

import os
from typing import Union, Optional, Literal
from operator import itemgetter
import numpy as np
import rasterio as rio
from scipy.stats import t
from datetime import datetime as dt

try:
    import cupy as cp
except ImportError as exc:  # pragma: no cover - depends on optional GPU stack
    raise ImportError(
        "The CuPy GPU backend requires the optional 'gpu' extra and a CUDA-capable "
        "environment. Install with: pip install cffdrs[gpu] (or install the "
        "CUDA-matched wheel directly, e.g. cupy-cuda12x). The reference CPU "
        "implementation is available as `from cffdrs.cffbps import FBP`."
    ) from exc

# Define lookup tables as in the original code
fbpFTCode_NumToAlpha_LUT = {
    1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6',
    7: 'C7', 8: 'D1', 9: 'D2', 10: 'M1', 11: 'M2', 12: 'M3',
    13: 'M4', 14: 'O1a', 15: 'O1b', 16: 'S1', 17: 'S2', 18: 'S3',
    19: 'NF', 20: 'WA'
}

fbpFTCode_AlphaToNum_LUT = {
    'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6,
    'C7': 7, 'D1': 8, 'D2': 9, 'M1': 10, 'M2': 11, 'M3': 12,
    'M4': 13, 'O1a': 14, 'O1b': 15, 'S1': 16, 'S2': 17, 'S3': 18,
    'NF': 19, 'WA': 20
}

# List of valid CFFBPS output parameters
valid_outputs = [
    'fire_type', 'hros', 'hfi', 'fuel_type', 'ws', 'wd', 'm', 'fF', 'fW', 'ffmc', 'bui', 'isi',
    'a', 'b', 'c', 'rsz', 'sf', 'rsf', 'isf', 'rsi', 'wse1', 'wse2', 'wse', 'wsx', 'wsy', 'wsv', 'raz',
    'q', 'bui0', 'be', 'be_max', 'ffc', 'wfc', 'sfc', 'latn', 'dj', 'd0', 'nd', 'fmc', 'fme',
    'csfi', 'rso', 'bfw', 'bisi', 'bros', 'sros', 'cros', 'cbh', 'cfb', 'cfl', 'cfc', 'tfc', 'accel', 'fi_class'
]


def convert_grid_codes(fuel_type_array: cp.ndarray) -> cp.ndarray:
    """
    Convert grid code values from the cffdrs R package fuel type codes
    to the codes used in this module.

    :param fuel_type_array: CFFBPS fuel type array (CuPy ndarray)
    :return: Converted CuPy ndarray with remapped fuel type codes
    """
    return cp.where(fuel_type_array == 19, 20,
                    cp.where(fuel_type_array == 13, 19,
                             cp.where(fuel_type_array == 12, 13,
                                      cp.where(fuel_type_array == 11, 12,
                                               cp.where(fuel_type_array == 10, 11,
                                                        cp.where(fuel_type_array == 9, 10,
                                                                 fuel_type_array)))))).astype(cp.int8)


def getSeasonGrassCuring(season: str,
                         province: str,
                         subregion: str = None) -> int:
    """
    Function returns a default grass curing code based on season, province, and subregion
    :param season: annual season ("spring", "summer", "fall", "winter")
    :param province: province being assessed ("AB", "BC")
    :param subregion: British Columbia subregions ("southeast", "other")
    :return: grass curing percent (%); "None" is returned if season or province are invalid
    """
    if province == 'AB':
        # Default seasonal grass curing values for Alberta
        # These curing rates are recommended by Neal McLoughlin to align with Alberta Wildfire knowledge and practices
        gc_dict = {
            'spring': 75,
            'summer': 40,
            'fall': 60,
            'winter': 100
        }
    elif province == 'BC':
        if subregion == 'southeast':
            # Default seasonal grass curing values for southeastern British Columbia
            gc_dict = {
                'spring': 100,
                'summer': 90,
                'fall': 90,
                'winter': 100
            }
        else:
            # Default seasonal grass curing values for British Columbia
            gc_dict = {
                'spring': 100,
                'summer': 60,
                'fall': 85,
                'winter': 100
            }
    else:
        gc_dict = {}

    return gc_dict.get(season.lower(), None)


# Replace NumPy arrays with CuPy arrays within the FBP class and associated methods
class FBP:
    """
    Class to model fire behavior with the Canadian Forest Fire Behavior Prediction System.
    Modified to use CuPy for GPU acceleration.
    """

    def __init__(self, cupy_float_type: cp.dtype = cp.float32):
        # Initialize CFFBPS input parameters
        self.cupy_float_type = cupy_float_type
        self.fuel_type = cp.array([0], dtype=self.cupy_float_type)
        self.wx_date = cp.array([0], dtype=self.cupy_float_type)  # For FMC calculations
        self.long = cp.array([0], dtype=self.cupy_float_type)
        self.elevation = cp.array([0], dtype=self.cupy_float_type)
        self.slope = cp.array([0], dtype=self.cupy_float_type)
        self.aspect = cp.array([0], dtype=self.cupy_float_type)
        self.ws = cp.array([0], dtype=self.cupy_float_type)
        self.wd = cp.array([0], dtype=self.cupy_float_type)
        self.ffmc = cp.array([0], dtype=self.cupy_float_type)
        self.bui = cp.array([0], dtype=self.cupy_float_type)
        self.pc = cp.array([0], dtype=self.cupy_float_type)
        self.pdf = cp.array([0], dtype=self.cupy_float_type)
        self.gfl = cp.array([0], dtype=self.cupy_float_type)
        self.gcf = cp.array([0], dtype=self.cupy_float_type)
        self.d0 = cp.array([0], dtype=self.cupy_float_type)  # For FMC calculations
        self.dj = cp.array([0], dtype=self.cupy_float_type)  # For FMC calculations
        self.out_request = cp.array([0], dtype=self.cupy_float_type)
        self.convert_fuel_type_codes = cp.array([0], dtype=self.cupy_float_type)
        self.percentile_growth = 50,
        self.return_array_as = cp.array([0], dtype=self.cupy_float_type)

        # Internal tracking
        self.return_array = cp.array([0], dtype=self.cupy_float_type)
        self.ref_array = cp.array([0], dtype=self.cupy_float_type)
        self.initialized = False

        # Initialize multiprocessing block variable
        self.block = cp.array([0], dtype=self.cupy_float_type)

        # Initialize unique fuel types list
        self.ftypes = cp.array([0], dtype=self.cupy_float_type)

        # Initialize weather parameters
        self.isi = cp.array([0], dtype=self.cupy_float_type)
        self.m = cp.array([0], dtype=self.cupy_float_type)
        self.fF = cp.array([0], dtype=self.cupy_float_type)
        self.fW = cp.array([0], dtype=self.cupy_float_type)

        # Initialize slope effect parameters
        self.a = cp.array([0], dtype=self.cupy_float_type)
        self.b = cp.array([0], dtype=self.cupy_float_type)
        self.c = cp.array([0], dtype=self.cupy_float_type)
        self.rsz = cp.array([0], dtype=self.cupy_float_type)
        self.isz = cp.array([0], dtype=self.cupy_float_type)
        self.sf = cp.array([0], dtype=self.cupy_float_type)
        self.rsf = cp.array([0], dtype=self.cupy_float_type)
        self.isf = cp.array([0], dtype=self.cupy_float_type)
        self.rsi = cp.array([0], dtype=self.cupy_float_type)
        self.wse1 = cp.array([0], dtype=self.cupy_float_type)
        self.wse2 = cp.array([0], dtype=self.cupy_float_type)
        self.wse = cp.array([0], dtype=self.cupy_float_type)
        self.wsx = cp.array([0], dtype=self.cupy_float_type)
        self.wsy = cp.array([0], dtype=self.cupy_float_type)
        self.wsv = cp.array([0], dtype=self.cupy_float_type)
        self.raz = cp.array([0], dtype=self.cupy_float_type)

        # Initialize BUI effect parameters
        self.q = cp.array([0], dtype=self.cupy_float_type)
        self.bui0 = cp.array([0], dtype=self.cupy_float_type)
        self.be = cp.array([0], dtype=self.cupy_float_type)
        self.be_max = cp.array([0], dtype=self.cupy_float_type)

        # Initialize surface parameters
        self.cf = cp.array([0], dtype=self.cupy_float_type)
        self.ffc = cp.array([0], dtype=self.cupy_float_type)
        self.wfc = cp.array([0], dtype=self.cupy_float_type)
        self.sfc = cp.array([0], dtype=self.cupy_float_type)
        self.rss = cp.array([0], dtype=self.cupy_float_type)

        # Initialize foliar moisture content (FMC) parameters
        self.latn = cp.array([0], dtype=self.cupy_float_type)
        self.nd = cp.array([0], dtype=self.cupy_float_type)
        self.fmc = cp.array([0], dtype=self.cupy_float_type)
        self.fme = cp.array([0], dtype=self.cupy_float_type)

        # Initialize crown and total fuel consumed parameters
        self.cbh = cp.array([0], dtype=self.cupy_float_type)
        self.csfi = cp.array([0], dtype=self.cupy_float_type)
        self.rso = cp.array([0], dtype=self.cupy_float_type)
        self.rsc = cp.array([0], dtype=self.cupy_float_type)
        self.cfb = cp.array([0], dtype=self.cupy_float_type)
        self.bros_cfb = cp.array([0], dtype=self.cupy_float_type)
        self.cfl = cp.array([0], dtype=self.cupy_float_type)
        self.cfc = cp.array([0], dtype=self.cupy_float_type)
        self.tfc = cp.array([0], dtype=self.cupy_float_type)

        # Initialize the back fire rate of spread parameters
        self.bfW = cp.array([0], dtype=self.cupy_float_type)
        self.brsi = cp.array([0], dtype=self.cupy_float_type)
        self.bisi = cp.array([0], dtype=self.cupy_float_type)
        self.bros = cp.array([0], dtype=self.cupy_float_type)

        # Initialize default CFFBPS output parameters
        self.fire_type = cp.array([0], dtype=cp.int8)
        self.hfros = cp.array([0], dtype=self.cupy_float_type)
        self.hfi = cp.array([0], dtype=self.cupy_float_type)

        # Initialize C-6 rate of spread parameters
        self.sfros = cp.array([0], dtype=self.cupy_float_type)
        self.cfros = cp.array([0], dtype=self.cupy_float_type)

        # Initialize point ignition acceleration parameter
        self.accel_param = cp.array([0], dtype=self.cupy_float_type)

        # Initialize fire intensity class parameter
        self.fi_class = cp.array([0], dtype=cp.int8)

        # ### Lists for CFFBPS Crown Fire Metric variables
        self.csfiVarList = ['cbh', 'fmc']
        self.rsoVarList = ['csfi', 'sfc']
        self.cfbVarList = ['cfros', 'rso']
        self.cfcVarList = ['cfb', 'cfl']
        self.cfiVarList = ['cfros', 'cfc']

        # Array of open fuel type codes
        self.open_fuel_types = cp.array([1, 7, 9, 14, 15, 16, 17, 18])

        # Array of non-crowning fuel type codes
        self.non_crowning_fuels = cp.array([8, 9, 14, 15, 16, 17, 18])

        # CFFBPS Canopy Base Height & Canopy Fuel Load Lookup Table (cbh, cfl, ht)
        self.fbpCBH_CFL_HT_LUT = {
            1: (2, 0.75, 10),
            2: (3, 0.8, 7),
            3: (8, 1.15, 18),
            4: (4, 1.2, 10),
            5: (18, 1.2, 25),
            6: (7, 1.8, 14),
            7: (10, 0.5, 20),
            8: (0, 0, 0),
            9: (0, 0, 0),
            10: (6, 0.8, 13),
            11: (6, 0.8, 13),
            12: (6, 0.8, 8),
            13: (6, 0.8, 8),
            14: (0, 0, 0),
            15: (0, 0, 0),
            16: (0, 0, 0),
            17: (0, 0, 0),
            18: (0, 0, 0)
        }

        # CFFBPS Surface Fire Rate of Spread Parameters (a, b, c, q, BUI0, be_max)
        self.rosParams = {
            1: (90, 0.0649, 4.5, 0.9, 72, 1.076),
            2: (110, 0.0282, 1.5, 0.7, 64, 1.321),
            3: (110, 0.0444, 3, 0.75, 62, 1.261),
            4: (110, 0.0293, 1.5, 0.8, 66, 1.184),
            5: (30, 0.0697, 4, 0.8, 56, 1.220),
            6: (30, 0.08, 3, 0.8, 62, 1.197),
            7: (45, 0.0305, 2, 0.85, 106, 1.134),
            8: (30, 0.0232, 1.6, 0.9, 32, 1.179),
            9: (30, 0.0232, 1.6, 0.9, 32, 1.179),
            10: (0, 0, 0, 0.8, 50, 1.250),
            11: (0, 0, 0, 0.8, 50, 1.250),
            12: (120, 0.0572, 1.4, 0.8, 50, 1.250),
            13: (100, 0.0404, 1.48, 0.8, 50, 1.250),
            14: (190, 0.0310, 1.4, 1, 0, 1),
            15: (250, 0.0350, 1.7, 1, 0, 1),
            16: (75, 0.0297, 1.3, 0.75, 38, 1.460),
            17: (40, 0.0438, 1.7, 0.75, 63, 1.256),
            18: (55, 0.0829, 3.2, 0.75, 31, 1.590)
        }

        return

    def _checkArray(self) -> None:
        """
        Check if any of the core input parameters are CuPy arrays and standardize reference shapes.

        :return: None
        """
        input_list = [
            self.fuel_type, self.lat, self.long,
            self.elevation, self.slope, self.aspect,
            self.ws, self.wd, self.ffmc, self.bui,
            self.pc, self.pdf, self.gfl, self.gcf
        ]

        if any(isinstance(data, cp.ndarray) for data in input_list):
            self.return_array = True
            array_indices = [i for i, data in enumerate(input_list) if isinstance(data, cp.ndarray)]
            arrays = itemgetter(*array_indices)(input_list)
            if isinstance(arrays, cp.ndarray):
                arrays = [arrays]
            shapes = {arr.shape for arr in arrays}
            if len(shapes) > 1:
                raise ValueError(f'Array dimensions mismatch: {shapes}')
            first_array = arrays[0]
            mask = cp.isnan(first_array)
            self.ref_array = cp.where(mask,
                                      cp.full_like(first_array, cp.nan, dtype=self.cupy_float_type),
                                      cp.zeros_like(first_array, dtype=self.cupy_float_type))
        else:
            self.return_array = False
            val = self.fuel_type
            if isinstance(val, str):
                val = fbpFTCode_AlphaToNum_LUT.get(val)
            elif isinstance(val, np.ndarray) and val.dtype.kind in ('U', 'S'):
                val = np.vectorize(fbpFTCode_AlphaToNum_LUT.get)(val).astype(np.uint16)
            mask = cp.isnan(cp.array([val], dtype=self.cupy_float_type))
            self.ref_array = cp.where(mask,
                                      cp.full_like(mask, cp.nan, dtype=self.cupy_float_type),
                                      cp.array([val], dtype=self.cupy_float_type))
        return

    def _verifyInputs(self) -> None:
        """
        Validate and convert all input parameters into CuPy arrays where needed.

        :return: None
        """
        if isinstance(self.fuel_type, str):
            self.fuel_type = cp.asarray([fbpFTCode_AlphaToNum_LUT.get(self.fuel_type)], dtype=cp.int8)
        elif isinstance(self.fuel_type, np.ndarray) and self.fuel_type.dtype.kind in ('U', 'S'):
            convert = np.vectorize(fbpFTCode_AlphaToNum_LUT.get)
            self.fuel_type = cp.asarray(convert(self.fuel_type), dtype=cp.int8)
        elif isinstance(self.fuel_type, cp.ndarray) and self.fuel_type.dtype.kind in ('U', 'S'):
            convert = cp.vectorize(fbpFTCode_AlphaToNum_LUT.get)
            self.fuel_type = convert(self.fuel_type).astype(cp.int8)
        elif not isinstance(self.fuel_type, cp.ndarray):
            self.fuel_type = cp.asarray([self.fuel_type], dtype=cp.int8)

        if self.convert_fuel_type_codes:
            self.fuel_type = convert_grid_codes(self.fuel_type)

        self.ftypes = cp.array([int(ft) for ft in cp.asnumpy(cp.unique(self.fuel_type)) if ft in self.rosParams])

        try:
            dt.strptime(str(self.wx_date), '%Y%m%d')
        except Exception:
            raise ValueError('wx_date must be in YYYYMMDD format')

        for attr in ['lat', 'long', 'elevation', 'slope', 'aspect', 'ws', 'wd', 'ffmc', 'bui']:
            val = getattr(self, attr)
            setattr(self, attr, cp.asarray(val, dtype=self.cupy_float_type))

        for attr, default in [('pc', 50), ('pdf', 35), ('gfl', 0.35), ('gcf', 80)]:
            val = getattr(self, attr)
            if val is None:
                val = default
            setattr(self, attr, cp.asarray(val, dtype=self.cupy_float_type))

        if self.return_array:
            if self.return_array_as not in ['cupy', 'numpy']:
                raise ValueError('The "return_array_as" parameter must be set to either "cupy" or "numpy".')

        return

    def _init_array(self, fill_value: Union[int, float] = 0, dtype: Optional[cp.dtype] = None) -> cp.ndarray:
        if dtype is None:
            dtype = self.cupy_float_type
        return cp.full(self.ref_array.shape, fill_value, dtype=dtype)

    def initialize(self,
                   fuel_type: Union[int, str, cp.ndarray] = None,
                   wx_date: int = None,
                   lat: Union[float, int, cp.ndarray] = None,
                   long: Union[float, int, cp.ndarray] = None,
                   elevation: Union[float, int, cp.ndarray] = None,
                   slope: Union[float, int, cp.ndarray] = None,
                   aspect: Union[float, int, cp.ndarray] = None,
                   ws: Union[float, int, cp.ndarray] = None,
                   wd: Union[float, int, cp.ndarray] = None,
                   ffmc: Union[float, int, cp.ndarray] = None,
                   bui: Union[float, int, cp.ndarray] = None,
                   pc: Optional[Union[float, int, cp.ndarray]] = 50,
                   pdf: Optional[Union[float, int, cp.ndarray]] = 35,
                   gfl: Optional[Union[float, int, cp.ndarray]] = 0.35,
                   gcf: Optional[Union[float, int, cp.ndarray]] = 80,
                   d0: Optional[int] = None,
                   dj: Optional[int] = None,
                   out_request: Optional[Union[list, tuple]] = None,
                   convert_fuel_type_codes: Optional[bool] = False,
                   percentile_growth: Optional[Union[float, int]] = 50,
                   return_array_as: Literal['numpy', 'cupy'] = 'numpy') -> None:
        """
        Initialize the FBP object with the provided parameters.

        :param fuel_type: CFFBPS fuel type (numeric code: 1-20)
            Model 1: C-1 fuel type ROS model
            Model 2: C-2 fuel type ROS model
            Model 3: C-3 fuel type ROS model
            Model 4: C-4 fuel type ROS model
            Model 5: C-5 fuel type ROS model
            Model 6: C-6 fuel type ROS model
            Model 7: C-7 fuel type ROS model
            Model 8: D-1 fuel type ROS model
            Model 9: D-2 fuel type ROS model
            Model 10: M-1 fuel type ROS model
            Model 11: M-2 fuel type ROS model
            Model 12: M-3 fuel type ROS model
            Model 13: M-4 fuel type ROS model
            Model 14: O-1a fuel type ROS model
            Model 15: O-1b fuel type ROS model
            Model 16: S-1 fuel type ROS model
            Model 17: S-2 fuel type ROS model
            Model 18: S-3 fuel type ROS model
            Model 19: Non-fuel (NF)
            Model 20: Water (WA)
        :param wx_date: Date of weather observation (used for fmc calculation) (YYYYMMDD)
        :param lat: Latitude of area being modelled (Decimal Degrees, floating point)
        :param long: Longitude of area being modelled (Decimal Degrees, floating point)
        :param elevation: Elevation of area being modelled (m)
        :param slope: Ground slope angle/tilt of area being modelled (%)
        :param aspect: Ground slope aspect/azimuth of area being modelled (degrees)
        :param ws: Wind speed (km/h @ 10m height)
        :param wd: Wind direction (degrees, direction wind is coming from)
        :param ffmc: CFFWIS Fine Fuel Moisture Code
        :param bui: CFFWIS Buildup Index
        :param pc: Percent conifer (%, value from 0-100)
        :param pdf: Percent dead fir (%, value from 0-100)
        :param gfl: Grass fuel load (kg/m^2)
        :param gcf: Grass curing factor (%, value from 0-100)
        :param d0: Julian date of minimum foliar moisture content (if None, calculated from latitude)
        :param dj: Julian date of modelled fire (if None, calculated from wx_date)
        :param out_request: Tuple or list of CFFBPS output variables
            # Default output variables
            fire_type = Type of fire predicted to occur (surface, intermittent crown, active crown)
            hfros = Head fire rate of spread (m/min)
            hfi = head fire intensity (kW/m)

            # Weather variables
            ws = Observed wind speed (km/h)
            wd = Wind azimuth/direction (degrees)
            m = Moisture content equivalent of the FFMC (%, value from 0-100+)
            fF = Fine fuel moisture function in the ISI
            fW = Wind function in the ISI
            ffmc = Fine Fuel Moisture Code
            bui = Buildup Index
            isi = Final calculated ISI, accounting for wind and slope

            # Slope + wind effect variables
            a = Rate of spread equation coefficient
            b = Rate of spread equation coefficient
            c = Rate of spread equation coefficient
            RSZ = Surface spread rate with zero wind on level terrain
            SF = Slope factor
            RSF = spread rate with zero wind, upslope
            ISF = ISI, with zero wind upslope
            RSI = Initial spread rate without BUI effect
            WSE1 = Original slope equivalent wind speed value
            WSE2 = New slope equivalent wind speed value for cases where WSE1 > 40 (capped at max of 112.45)
            WSE = Slope equivalent wind speed
            WSX = Net vectorized wind speed in the x-direction
            WSY = Net vectorized wind speed in the y-direction
            WSV = (aka: slope-adjusted wind speed) Net vectorized wind speed (km/h)
            RAZ = (aka: slope-adjusted wind direction) Net vectorized wind direction (degrees)

            # BUI effect variables
            q = Proportion of maximum rate of spread at BUI equal to 50
            bui0 = Average BUI for each fuel type
            BE = Buildup effect on spread rate
            be_max = Maximum allowable BE value

            # Surface fuel variables
            ffc = Estimated forest floor consumption
            wfc = Estimated woody fuel consumption
            sfc = Estimated total surface fuel consumption

            # Foliar moisture content variables
            latn = Normalized latitude
            d0 = Julian date of minimum foliar moisture content
            nd = number of days between modelled fire date and d0
            fmc = foliar moisture content
            fme = foliar moisture effect

            # Critical crown fire threshold variables
            csfi = critical intensity (kW/m)
            rso = critical rate of spread (m/min)

            # Crown fuel variables
            cbh = Height to live crown base (m)
            cfb = Crown fraction burned (proportion, value ranging from 0-1)
            cfl = Crown fuel load (kg/m^2)
            cfc = Crown fuel consumed (kg/m^2)

            # Final fuel parameters
            tfc = Total fuel consumed

            # Acceleration parameter
            accel = Acceleration parameter for point source ignition

            # Fire Intensity Class parameter
            fi_class = Fire intensity class (1-6)

        :param convert_fuel_type_codes: Convert from CFS cffdrs R fuel type grid codes
            to the grid codes used in this module.
        :param percentile_growth: Percentile growth to use for the ROS growth function.
        :param return_array_as: If the results are arrays, the type of array to return as. Options: 'numpy', 'cupy'.
        """
        self.fuel_type = fuel_type
        self.wx_date = wx_date
        self.lat = lat
        self.long = long
        self.elevation = elevation
        self.slope = slope
        self.aspect = aspect
        self.ws = ws
        self.wd = wd
        self.ffmc = ffmc
        self.bui = bui
        self.pc = pc
        self.pdf = pdf
        self.gfl = gfl
        self.gcf = gcf
        self.d0 = d0
        self.dj = dj
        self.out_request = out_request
        self.convert_fuel_type_codes = convert_fuel_type_codes
        self.percentile_growth = percentile_growth
        self.return_array_as = return_array_as

        self._checkArray()
        self._verifyInputs()

        # Initialize all model parameter arrays
        self.isi = self._init_array()
        self.m = self._init_array()
        self.fF = self._init_array()
        self.fW = self._init_array()
        self.a = self._init_array()
        self.b = self._init_array()
        self.c = self._init_array()
        self.rsz = self._init_array()
        self.isz = self._init_array()
        self.sf = self._init_array()
        self.rsf = self._init_array()
        self.isf = self._init_array()
        self.rsi = self._init_array()
        self.wse1 = self._init_array()
        self.wse2 = self._init_array()
        self.wse = self._init_array()
        self.wsx = self._init_array()
        self.wsy = self._init_array()
        self.wsv = self._init_array()
        self.raz = self._init_array()
        self.q = self._init_array()
        self.bui0 = self._init_array()
        self.be = self._init_array()
        self.be_max = self._init_array()
        self.cf = self._init_array()
        self.ffc = self._init_array(cp.nan)
        self.wfc = self._init_array(cp.nan)
        self.sfc = self._init_array(cp.nan)
        self.rss = self._init_array()
        self.latn = self._init_array()
        self.nd = self._init_array()
        self.fmc = self._init_array()
        self.fme = self._init_array()
        self.cbh = self._init_array()
        self.csfi = self._init_array()
        self.rso = self._init_array()
        self.rsc = self._init_array()
        self.cfb = self._init_array()
        self.bros_cfb = self._init_array()
        self.cfl = self._init_array()
        self.cfc = self._init_array()
        self.tfc = self._init_array()
        self.bfW = self._init_array()
        self.brsi = self._init_array()
        self.bisi = self._init_array()
        self.bros = self._init_array()
        self.fire_type = self._init_array(0, dtype=cp.int8)
        self.hfros = self._init_array()
        self.hfi = self._init_array()
        self.sfros = self._init_array()
        self.cfros = self._init_array()
        self.accel_param = self._init_array()
        self.fi_class = self._init_array(0, dtype=cp.int8)

        # List of required parameters
        required_params = [
            'fuel_type', 'wx_date', 'lat', 'long', 'elevation', 'slope', 'aspect', 'ws', 'wd', 'ffmc', 'bui'
        ]

        # Check for missing required parameters
        missing_params = [param for param in required_params if getattr(self, param) is None]
        if missing_params:
            raise ValueError(f'Missing required parameters: {missing_params}')

        # Set initialized to True
        self.initialized = True
        return

    def invertWindAspect(self):
        """
        Function to invert/flip wind direction and aspect by 180 degrees using CuPy
        :return: None
        """
        # Invert wind direction by 180 degrees (i.e., direction wind is heading to)
        self.wd = cp.where(self.wd > 180, self.wd - 180, self.wd + 180)

        # Invert aspect by 180 degrees (i.e., up slope direction)
        self.aspect = cp.where(self.aspect > 180, self.aspect - 180, self.aspect + 180)

        return

    def calcSF(self) -> None:
        """
        Function to calculate the slope factor using CuPy
        :return: None
        """
        # Calculate the slope factor with CuPy
        self.sf = cp.where(
            self.slope < 70,
            cp.exp(3.533 * cp.power((self.slope / 100), 1.2)),
            cp.full_like(self.slope, 10, dtype=cp.int8)
        )
        return

    def calcISZ(self) -> None:
        """
        Function to calculate the initial spread index with no wind/no slope effects.
        """
        self.m = (250 * (59.5 / 101) * (101 - self.ffmc)) / (59.5 + self.ffmc)
        self.fF = (91.9 * cp.exp(-0.1386 * self.m)) * (1 + (cp.power(self.m, 5.31) / (4.93 * cp.power(10, 7))))
        self.isz = 0.208 * self.fF

    def calcFMC(self,
                d0: Optional[int] = None,
                dj: Optional[int] = None,
                lat: Optional[float] = None,
                long: Optional[float] = None,
                elevation: Optional[float] = None,
                wx_date: Optional[int] = None) -> None:
        """
        Function to calculate foliar moisture content (FMC) and foliar moisture effect (FME) using CuPy.

        :return: None
        """
        if lat is not None:
            self.lat = cp.asarray(lat, dtype=self.cupy_float_type)
        if long is not None:
            self.long = cp.asarray(long, dtype=self.cupy_float_type)
        if elevation is not None:
            self.elevation = cp.asarray(elevation, dtype=self.cupy_float_type)
        if wx_date is not None:
            self.wx_date = wx_date

        # Normalize latitude
        abs_long = cp.abs(self.long)
        use_elev = (self.elevation > 0)
        self.latn = cp.where(
            use_elev,
            43 + (33.7 * cp.exp(-0.0351 * (150 - abs_long))),
            46 + (23.4 * cp.exp(-0.036 * (150 - abs_long)))
        )

        # Julian date
        if self.dj is None:
            if dj is None:
                dj_value = dt.strptime(str(self.wx_date), '%Y%m%d').timetuple().tm_yday
                self.dj = cp.full_like(self.latn, dj_value, dtype=cp.int16)
            else:
                self.dj = cp.full_like(self.latn, dj, dtype=cp.int16)

        # D0 calculation
        if self.d0 is None:
            if d0 is None:
                self.d0 = cp.round(cp.where(
                    use_elev,
                    142.1 * (self.lat / self.latn) + (0.0172 * self.elevation),
                    151 * (self.lat / self.latn)
                ), 0).astype(cp.int16)
            else:
                self.d0 = cp.full_like(self.latn, d0, dtype=cp.int16)

        # Number of days between Dj and D0 (ND)
        self.nd = cp.abs(self.dj - self.d0)

        # Foliar moisture content (FMC) calculation
        nd_lt_30 = self.nd < 30
        nd_lt_50 = (self.nd >= 30) & (self.nd < 50)

        self.fmc = cp.where(
            nd_lt_30,
            85 + (0.0189 * self.nd ** 2),
            cp.where(
                nd_lt_50,
                32.9 + (3.17 * self.nd) - (0.0288 * self.nd ** 2),
                cp.full_like(self.nd, 120, dtype=self.cupy_float_type)
            )
        )

        # FME calculation
        self.fme = 1000 * cp.power(1.5 - (0.00275 * self.fmc), 4) / (460 + (25.9 * self.fmc))
        return

    def _calcISI_slopeWind_vectorized(self) -> None:
        """
        Vectorized slope and wind adjustment function for ISI and RSI.
        Uses CuPy to compute slope-equivalent wind and directional vectors across all cells.
        """
        # Calculate slope-equivalent wind speeds using two formulas
        # with cp.errstate(divide='ignore', invalid='ignore'):
        self.wse1 = (1 / 0.05039) * cp.log(self.isf / (0.208 * self.fF))
        self.wse2 = cp.where(
            self.isf < 0.999 * 2.496 * self.fF,
            28 - (1 / 0.0818) * cp.log(1 - (self.isf / (2.496 * self.fF))),
            112.45  # cap maximum WSE
        )

        # Assign slope equivalent wind speed
        self.wse = cp.where(self.wse1 <= 40, self.wse1, self.wse2)

        # Compute directional components for wind and slope
        sin_wd = cp.sin(cp.radians(self.wd))
        cos_wd = cp.cos(cp.radians(self.wd))
        sin_asp = cp.sin(cp.radians(self.aspect))
        cos_asp = cp.cos(cp.radians(self.aspect))

        # Net wind vectors
        self.wsx = self.ws * sin_wd + self.wse * sin_asp
        self.wsy = self.ws * cos_wd + self.wse * cos_asp
        self.wsv = cp.sqrt(self.wsx ** 2 + self.wsy ** 2)

        # Wind azimuth calculation (RAZ)
        acos_val = cp.clip(self.wsy / self.wsv, -1, 1)
        angle_rad = cp.arccos(acos_val)
        self.raz = cp.where(
            self.wsx < 0,
            360 - cp.degrees(angle_rad),
            cp.degrees(angle_rad)
        )

        # Compute head fire and backfire wind function
        self.fW = cp.where(
            self.wsv > 40,
            12 * (1 - cp.exp(-0.0818 * (self.wsv - 28))),
            cp.exp(0.05039 * self.wsv)
        )
        self.bfW = cp.exp(-0.05039 * self.wsv)

        # Final head fire and backfire ISI
        self.isi = 0.208 * self.fF * self.fW
        self.bisi = 0.208 * self.fF * self.bfW

        return

    def calcISI_RSI_BE(self) -> None:
        """
        Function to calculate the slope-/wind-adjusted Initial Spread Index (ISI),
        rate of spread (RSI), and the BUI buildup effect (BE) using CuPy.

        :return: None
        """
        ft = self.fuel_type

        # Initialize vectors
        shape = self.fuel_type.shape
        a = cp.zeros(shape, dtype=self.cupy_float_type)
        b = cp.zeros(shape, dtype=self.cupy_float_type)
        c = cp.zeros(shape, dtype=self.cupy_float_type)
        q = cp.zeros(shape, dtype=self.cupy_float_type)
        bui0 = cp.ones(shape, dtype=self.cupy_float_type)
        be_max = cp.ones(shape, dtype=self.cupy_float_type)

        # Generate masks
        m12_mask = (ft == 10) | (ft == 11)
        m34_mask = (ft == 12) | (ft == 13)
        o1_mask = (ft == 14) | (ft == 15)

        # Precompute C2, D1 and D2 parameters
        c2 = self.rosParams[2]
        d1 = self.rosParams[8]
        d2 = self.rosParams[9]

        for ftype in range(1, 21):
            a_val, b_val, c_val, q_val, bui0_val, be_max_val = self.rosParams.get(ftype, (0, 0, 0, 0, 1, 1))

            mask = ft == ftype
            a = cp.where(mask, a_val, a)
            b = cp.where(mask, b_val, b)
            c = cp.where(mask, c_val, c)
            q = cp.where(mask, q_val, q)
            bui0 = cp.where(mask, bui0_val, bui0)
            be_max = cp.where(mask, be_max_val, be_max)
            del mask

        # Handle O1a/b (ftype 14 and 15) curing factor logic
        cf = cp.where(
            self.gcf < 58.8,
            0.005 * (cp.exp(0.061 * self.gcf) - 1),
            0.176 + 0.02 * (self.gcf - 58.8)
        )

        # Compute RSZ
        rsz_core = a * cp.power(1 - cp.exp(-b * self.isz), c)
        # M1/2
        rsz_c2 = c2[0] * cp.power(1 - cp.exp(-c2[1] * self.isz), c2[2])
        rsz_d1 = d1[0] * cp.power(1 - cp.exp(-d1[1] * self.isz), d1[2])
        rsz_m1 = (self.pc / 100) * rsz_c2 + (1 - self.pc / 100) * rsz_d1
        rsz_m2 = (self.pc / 100) * rsz_c2 + 0.2 * (1 - self.pc / 100) * rsz_d1
        # O1a/b
        rsz_o1 = rsz_core * cf
        # Final calculation
        self.rsz = cp.where(ft == 10, rsz_m1, rsz_core)
        self.rsz = cp.where(ft == 11, rsz_m2, self.rsz)
        self.rsz = cp.where(o1_mask, rsz_o1, self.rsz)

        # Compute RSF
        rsf_c2 = rsz_c2 * self.sf
        rsf_d1 = rsz_d1 * self.sf
        self.rsf = self.rsz * self.sf

        # Compute ISF for M1/2 & M3/4 blending logic
        isf_c2_numer = 1 - cp.power(rsf_c2 / c2[0], 1 / c2[2])
        isf_d1_numer = 1 - cp.power(rsf_d1 / d1[0], 1 / d1[2])
        isf_m34_numer = 1 - cp.power(self.rsf / a, 1 / c)
        isf_c2_core = cp.where(isf_c2_numer >= 0.01, cp.log(isf_c2_numer) / -c2[1], cp.log(0.01) / -c2[1])
        isf_d1_core = cp.where(isf_d1_numer >= 0.01, cp.log(isf_d1_numer) / -d1[1], cp.log(0.01) / -d1[1])
        isf_m34_core = cp.where(isf_m34_numer >= 0.01, cp.log(isf_m34_numer) / -b, cp.log(0.01) / -b)
        isf_blended_m12 = (self.pc / 100) * isf_c2_core + (1 - self.pc / 100) * isf_d1_core
        isf_blended_m34 = (self.pdf / 100) * isf_m34_core + (1 - self.pdf / 100) * isf_d1_core
        del rsz_core, rsz_m1, rsz_m2, rsz_o1
        del rsf_c2, rsf_d1
        del isf_c2_numer, isf_d1_numer, isf_c2_core, isf_d1_core

        # Compute ISF
        isf_numer = cp.where(o1_mask, 1 - cp.power(self.rsf / (a * cf), 1 / c), 1 - cp.power(self.rsf / a, 1 / c))
        isf_final = cp.where(isf_numer >= 0.01, cp.log(isf_numer) / -b, cp.log(0.01) / -b)
        self.isf = cp.where(m12_mask, isf_blended_m12, isf_final)
        self.isf = cp.where(m34_mask, isf_blended_m34, self.isf)
        del isf_numer, isf_final

        # Wind and slope adjusted ISI
        self._calcISI_slopeWind_vectorized()

        # Final RSI and BRSI
        rsi_c2 = c2[0] * cp.power(1 - cp.exp(-c2[1] * self.isi), c2[2])
        rsi_d1 = d1[0] * cp.power(1 - cp.exp(-d1[1] * self.isi), d1[2])
        self.rsi = cp.where(
            (ft == 12),  # M3
            (self.pdf / 100) * a * cp.power(1 - cp.exp(-b * self.isi), c) +
            (1 - self.pdf / 100) * rsi_d1,
            cp.where(
                (ft == 13),  # M4
                (self.pdf / 100) * a * cp.power(1 - cp.exp(-b * self.isi), c) +
                0.2 * (1 - self.pdf / 100) * rsi_d1,
                cp.where(
                    (ft == 10),  # M1
                    (self.pc / 100) * rsi_c2 + (1 - self.pc / 100) * rsi_d1,
                    cp.where(
                        (ft == 11),  # M2
                        (self.pc / 100) * rsi_c2 + 0.2 * (1 - self.pc / 100) * rsi_d1,
                        cp.where(
                            o1_mask,
                            a * cp.power(1 - cp.exp(-b * self.isi), c) * cf,
                            a * cp.power(1 - cp.exp(-b * self.isi), c)
                        )
                    )
                )
            )
        )

        brsi_c2 = c2[0] * cp.power(1 - cp.exp(-c2[1] * self.bisi), c2[2])
        brsi_d1 = d1[0] * cp.power(1 - cp.exp(-d1[1] * self.bisi), d1[2])
        self.brsi = cp.where(
            (ft == 12),
            (self.pdf / 100) * self.a * np.power(1 - np.exp(-self.b * self.bisi), self.c) +
            (1 - self.pdf / 100) * brsi_d1,
            cp.where(
                (ft == 13),
                (self.pdf / 100) * self.a * np.power(1 - np.exp(-self.b * self.bisi), self.c) +
                0.2 * (1 - self.pdf / 100) * brsi_d1,
                cp.where(
                    (ft == 11),
                    (self.pc / 100) * brsi_c2 + 0.2 * (1 - self.pc / 100) * brsi_d1,
                    cp.where(
                        (ft == 10),
                        (self.pc / 100) * brsi_c2 + (1 - self.pc / 100) * brsi_d1,
                        cp.where(
                            o1_mask,
                            a * cp.power(1 - cp.exp(-b * self.bisi), c) * cf,
                            a * cp.power(1 - cp.exp(-b * self.bisi), c)
                        )
                    )
                )
            )
        )

        # Compute BE and clip
        raw_be = cp.where(self.bui == 0,
                          0.0,
                          cp.where(bui0 == 0,
                                   1,
                                   cp.exp(50 * cp.log(q) * ((1 / self.bui) - (1 / bui0)))
                                   )
                          )
        self.be = cp.clip(raw_be, 0, be_max)

        del raw_be, ft, c2, d1, d2, a, b, c, q, bui0, be_max, cf, m12_mask, o1_mask

        return

    def calcROS(self) -> None:
        """
        Function to model the fire rate of spread (m/min) using CuPy.
        For C6, this is the surface fire heading and backing rate of spread.
        For all other fuel types, this is the overall heading and backing fire rate of spread.

        :return: None
        """
        ft = self.fuel_type

        # Initialize hfros and bros
        self.hfros = self.rsi * self.be
        self.bros = self.brsi * self.be

        # Special handling for C6 (fuel_type == 6)
        is_c6 = ft == 6
        self.sfros = cp.where(is_c6, self.rsi * self.be, self.sfros)

        # D2 correction: zero out if BUI < 70, then scale by 0.2
        is_d2 = ft == 9
        self.hfros = cp.where(
            is_d2,
            cp.where(self.bui < 70, 0.0, self.hfros * 0.2),
            self.hfros
        )
        self.bros = cp.where(
            is_d2,
            cp.where(self.bui < 70, 0.0, self.bros * 0.2),
            self.bros
        )
        del is_c6, is_d2, ft

        return

    def calcSFC(self) -> None:
        """
        Function to calculate forest floor consumption (FFC), woody fuel consumption (WFC),
        and total surface fuel consumption (SFC) using CuPy.

        :return: None
        """
        ft = self.fuel_type

        # Initialize arrays
        ffc = cp.full_like(ft, cp.nan, dtype=self.cupy_float_type)
        wfc = cp.full_like(ft, cp.nan, dtype=self.cupy_float_type)
        sfc = cp.full_like(ft, cp.nan, dtype=self.cupy_float_type)

        # Define masks
        m1 = ft == 1
        m2 = ft == 2
        m3_4 = (ft == 3) | (ft == 4)
        m5_6 = (ft == 5) | (ft == 6)
        m7 = ft == 7
        m8_9 = (ft == 8) | (ft == 9)
        m10_11 = (ft == 10) | (ft == 11)
        m12_13 = (ft == 12) | (ft == 13)
        m14_15 = (ft == 14) | (ft == 15)
        m16 = ft == 16
        m17 = ft == 17
        m18 = ft == 18

        # Assign values based on fuel type
        sfc = cp.where(m1,
                       cp.where(self.ffmc > 84,
                                0.75 + 0.75 * cp.sqrt(1 - cp.exp(-0.23 * (self.ffmc - 84))),
                                0.75 - 0.75 * cp.sqrt(1 - cp.exp(0.23 * (self.ffmc - 84)))),
                       sfc)

        sfc = cp.where(m2, 5 * (1 - cp.exp(-0.0115 * self.bui)), sfc)
        sfc = cp.where(m3_4, 5 * cp.power(1 - cp.exp(-0.0164 * self.bui), 2.24), sfc)
        sfc = cp.where(m5_6, 5 * cp.power(1 - cp.exp(-0.0149 * self.bui), 2.48), sfc)

        ffc = cp.where(m7, 2 * (1 - cp.exp(-0.104 * (self.ffmc - 70))), ffc)
        ffc = cp.where(m7 & (ffc < 0), 0.0, ffc)
        wfc = cp.where(m7, 1.5 * (1 - cp.exp(-0.0201 * self.bui)), wfc)
        sfc = cp.where(m7, ffc + wfc, sfc)

        sfc = cp.where(m8_9, 1.5 * (1 - cp.exp(-0.0183 * self.bui)), sfc)

        c2_sfc = 5 * (1 - cp.exp(-0.0115 * self.bui))
        d1_sfc = 1.5 * (1 - cp.exp(-0.0183 * self.bui))
        sfc = cp.where(m10_11, (self.pc / 100) * c2_sfc + ((100 - self.pc) / 100) * d1_sfc, sfc)

        sfc = cp.where(m12_13, 5 * (1 - cp.exp(-0.0115 * self.bui)), sfc)
        sfc = cp.where(m14_15, self.gfl, sfc)

        ffc = cp.where(m16, 4 * (1 - cp.exp(-0.025 * self.bui)), ffc)
        wfc = cp.where(m16, 4 * (1 - cp.exp(-0.034 * self.bui)), wfc)
        sfc = cp.where(m16, ffc + wfc, sfc)

        ffc = cp.where(m17, 10 * (1 - cp.exp(-0.013 * self.bui)), ffc)
        wfc = cp.where(m17, 6 * (1 - cp.exp(-0.06 * self.bui)), wfc)
        sfc = cp.where(m17, ffc + wfc, sfc)

        ffc = cp.where(m18, 12 * (1 - cp.exp(-0.0166 * self.bui)), ffc)
        wfc = cp.where(m18, 20 * (1 - cp.exp(-0.021 * self.bui)), wfc)
        sfc = cp.where(m18, ffc + wfc, sfc)

        # Assign final arrays
        self.ffc = cp.where(~cp.isnan(ffc), ffc, self.ffc)
        self.wfc = cp.where(~cp.isnan(wfc), wfc, self.wfc)
        self.sfc = cp.where(~cp.isnan(sfc), sfc, self.sfc)

        return

    def getCBH_CFL(self,
                   cbh_override: Optional[Union[float, cp.ndarray, np.ndarray]] = None,
                   cfl_override: Optional[Union[float, cp.ndarray, np.ndarray]] = None) -> None:
        """
        Function to get the default CFFBPS canopy base height (CBH) and canopy fuel load (CFL)
        values for a specified fuel type using CuPy. User can override the default values for
        the C6 fuel type if needed.

        :param cbh_override: A specific cbh value to use instead of the default (only for C6 fuel types)
        :param cfl_override: A specific cfl value to use instead of the default (only for C6 fuel types)
        :return: None
        """
        # Verify optional inputs
        if cbh_override is not None:
            if not isinstance(cbh_override, (float, cp.ndarray, np.ndarray)):
                raise ValueError('The "cbh_override" parameter must be a float data type.')
            if isinstance(cbh_override, np.ndarray):
                cbh_override = cp.array(cbh_override, dtype=self.cupy_float_type)
        if cfl_override is not None:
            if not isinstance(cfl_override, (float, cp.ndarray, np.ndarray)):
                raise ValueError('The "cfl_override" parameter must be a float data type.')
            if isinstance(cfl_override, np.ndarray):
                cfl_override = cp.array(cfl_override, dtype=self.cupy_float_type)

        # Get the fuel type array
        ft = self.fuel_type

        # Create lookup arrays for CBH and CFL indexed by fuel type (1–20)
        cbh_list = [0]
        cfl_list = [0]
        for i in range(1, 21):
            cbh_list.append(self.fbpCBH_CFL_HT_LUT.get(i, [0, 0, 0])[0])
            cfl_list.append(self.fbpCBH_CFL_HT_LUT.get(i, [0, 0, 0])[1])
        cbh_vals = cp.array(cbh_list + [0, 0], dtype=self.cupy_float_type)
        cfl_vals = cp.array(cfl_list + [0, 0], dtype=self.cupy_float_type)

        # Set a mask for C6 fuel types
        c6_mask = ft == 6

        # ## PROCESS CBH
        # Assign canopy base height (CBH) for all fuel types 1–20
        self.cbh = cp.where((ft >= 1) & (ft <= 20), cbh_vals[ft], self.cbh)
        # Override CBH for C6 if specified
        if cbh_override is not None:
            self.cbh = cp.where(c6_mask, cbh_override, self.cbh)

        # ## PROCESS CFL
        # Assign canopy fuel load (CFL) for all fuel types 1–20
        self.cfl = cp.where((ft >= 1) & (ft <= 20), cfl_vals[ft], self.cfl)
        # Override CFL for C6 if specified
        if cfl_override is not None:
            self.cfl = cp.where(c6_mask, cfl_override, self.cfl)

        del ft, cbh_list, cfl_list, cbh_vals, cfl_vals, c6_mask

        return

    def calcCSFI(self) -> None:
        """
        Function to calculate the critical surface fire intensity (CSFI) using CuPy.

        :return: None
        """
        ft = self.fuel_type

        # Calculate CSFI where fuel type is < 14
        valid_mask = ft < 14
        csfi = cp.power(0.01 * self.cbh * (460 + (25.9 * self.fmc)), 1.5)
        self.csfi = cp.where(valid_mask, csfi, 0.0)

        del ft, valid_mask, csfi

        return

    def calcRSO(self) -> None:
        """
        Function to calculate the critical surface fire rate of spread (RSO) using CuPy.

        :return: None
        """
        # Calculate critical surface fire rate of spread (RSO)
        self.rso = cp.where(
            self.sfc > 0,
            self.csfi / (300.0 * self.sfc),
            0.0
        )
        return

    def calcCFB(self) -> None:
        """
        Function to calculate crown fraction burned using CuPy.
        Equation per Forestry Canada Fire Danger Group (1992).

        Also computes a backing-fire-specific CFB (self.bros_cfb), using bros in
        place of hfros, for calcRosPercentileGrowth's backing-fire regime decision
        (matches WISE FBPFuel::BROS computing its own CFB from brss, distinct
        from FBPFuel::ROS's head-fire CFB from rss). For C6, sfros (the
        C6-specific surface ROS used in place of hfros for CFB) is head-fire-
        derived only -- no backing-fire equivalent is computed elsewhere in
        this pipeline, so the backing CFB for C6 reuses the same sfros as a
        documented simplification.

        :return: None
        """
        # Masks
        is_c6 = self.fuel_type == 6
        non_crowning = cp.isin(self.fuel_type, self.non_crowning_fuels)
        is_other = cp.isin(self.fuel_type, self.ftypes) & ~is_c6 & ~non_crowning

        # C6 always uses sfros (head-fire-derived; no backing equivalent available)
        delta_sfros_c6 = self.sfros - self.rso
        cfb_c6 = cp.where(delta_sfros_c6 < -3086, 0.0, 1 - cp.exp(-0.23 * delta_sfros_c6))

        def _cfb_for_other(ros_other: cp.ndarray) -> cp.ndarray:
            delta_other = ros_other - self.rso
            cfb_other = cp.where(delta_other < -3086, 0.0, 1 - cp.exp(-0.23 * delta_other))
            cfb = cp.zeros_like(self.fuel_type, dtype=self.cupy_float_type)
            cfb = cp.where(is_c6, cfb_c6, cfb)
            cfb = cp.where(is_other, cfb_other, cfb)
            return cp.clip(cp.nan_to_num(cfb, nan=0.0), 0, 1)

        self.cfb = _cfb_for_other(self.hfros)
        self.bros_cfb = _cfb_for_other(self.bros)

        # Clean up memory
        del is_c6, is_other, delta_sfros_c6, cfb_c6

        return

    def calcRosPercentileGrowth(self) -> None:
        """
        Calculates the percentile growth for head fire and backing fire rates of spread.
        This function adjusts the `hfros` and `bros` attributes based on the percentile growth value and
        crown/surface spread parameters.

        Implements the variance-stabilized ROS quantile model of Han, L. & Braun,
        W.J. (2014), "Dionysus: a stochastic fire growth scenario generator",
        Environmetrics 25(6):431-442. Below the crowning threshold (cfb < 0.1),
        ROS residuals are treated as log-normal and scaled by
        exp(tinv * sigma_surface). At or above it, a closed-form Box-Cox
        power-law adjustment (delta=0.6, the paper's fitted crown-fire
        transform) applies unless its radicand would go negative (outside the
        transform's valid domain), in which case it falls back to the same
        log-normal-shift form using the crown-fire sigma. Fuel types with no
        fitted sigma for the applicable regime are left unchanged.

        hfros and bros each use their own direction-specific CFB (self.cfb /
        self.bros_cfb) to decide the surface-vs-crown regime, and bros's noise
        term is additionally scaled by a wind-speed decay factor k(wsv) (the
        paper's Eq. 3): backing-spread variability shrinks as wind speed
        increases, the same way backing ROS itself does. hfros's noise is not
        wind-scaled.

        :return: None
        """

        def _tinv(probability: float, freedom: int = 9999999):
            """
            Calculates the standard-normal quantile, via a Student's t at very high freedom.

            Han & Braun (2014) specify the standard normal quantile directly; a t
            distribution at freedom=9999999 is numerically indistinguishable from
            it and is what this method's fuel-type sigmas were fit against.

            :param probability: The cumulative probability for which the quantile is calculated.
            :param freedom: The degrees of freedom for the t-distribution.
            :return: The quantile value.
            """
            return t.ppf(probability, freedom)

        def _wind_decay(w: cp.ndarray) -> cp.ndarray:
            """
            Wind-speed decay factor for backing-fire growth-percentile noise.

            Han & Braun (2014), the k(w) accompanying their Eq. 3: k(0) = 1,
            decaying as wind speed increases.

            :param w: Wind speed (wsv).
            :return: The decay factor, same shape as w.
            """
            low = cp.exp(-0.10078 * w)
            high = cp.exp(-0.05039 * w) / (12.0 * (1.0 - cp.exp(-0.0818 * (w - 28.0))))
            return cp.where(w < 40, low, high)

        if self.percentile_growth != 50:
            # Calculate the inverse t-distribution for the given percentile growth
            tinv_value = _tinv(probability=self.percentile_growth / 100, freedom=9999999)

            # Prepare default table with structured dtype. Both are fitted noise
            # standard deviations (Han & Braun 2014, Section 3): surface_vals scales
            # a log-normal shift, crown_vals scales the Box-Cox power-law adjustment.
            keys = cp.array([1, 2, 3, 4, 5, 6, 7, 8, 12], dtype=cp.uint8)
            surface_vals = cp.array([cp.nan, 0.84, 0.62, 0.74, 0.8, 0.66, 1.22, 0.716, 0.551], dtype=cp.float32)
            crown_vals = cp.array([0.95, 1.82, 1.78, 1.38, cp.nan, 1.54, 1.0, cp.nan, cp.nan], dtype=cp.float32)

            # Initialize default arrays for lookup
            surface_s = cp.full_like(self.fuel_type, cp.nan, dtype=cp.float32)
            crown_s = cp.full_like(self.fuel_type, cp.nan, dtype=cp.float32)

            # Create a mask for each valid fuel type and assign values
            for k, s_val, c_val in zip(keys, surface_vals, crown_vals):
                valid_mask = self.fuel_type == k
                surface_s = cp.where(valid_mask, s_val, surface_s)
                crown_s = cp.where(valid_mask, c_val, crown_s)

            has_surface = ~cp.isnan(surface_s)
            has_crown = ~cp.isnan(crown_s)
            wind_decay = _wind_decay(self.wsv)

            # Iterate over head fire and backing fire ROS attributes, each with its
            # own direction-specific CFB and noise-scaling factor.
            for ros_attr, regime_cfb, noise_scale in (
                ('hfros', self.cfb, 1.0),
                ('bros', self.bros_cfb, wind_decay),
            ):
                ros_in = getattr(self, ros_attr)  # Get the current ROS value

                # Surface regime: log-normal shift, scaled by the fuel type's sigma
                surface_regime = cp.where(has_surface, ros_in * cp.exp(tinv_value * surface_s * noise_scale), ros_in)

                # Crown regime: Box-Cox power-law adjustment, falling back to the
                # same log-normal-shift form (using crown_s) outside its domain
                radicand = cp.power(ros_in, 0.6) + tinv_value * crown_s * noise_scale
                power_law = cp.power(cp.where(radicand >= 0, radicand, 0.0), 1 / 0.6)
                crown_fallback = ros_in * cp.exp(tinv_value * crown_s * noise_scale)
                crown_regime = cp.where(has_crown, cp.where(radicand >= 0, power_law, crown_fallback), ros_in)

                ros_growth = cp.where(regime_cfb < 0.1, surface_regime, crown_regime)
                setattr(self, ros_attr, ros_growth)  # Update the ROS attribute with the adjusted value

        return

    def calcAccelParam(self) -> None:
        """
        Function to calculate acceleration parameter for a fire starting from a point ignition source.

        :return: None
        """
        # Mask for open fuel types that use a fixed acceleration parameter (0.115)
        fixed_mask = cp.isin(self.fuel_type, self.open_fuel_types)

        # Mask for closed fuel types that require computation
        variable_mask = cp.isin(self.fuel_type, self.ftypes) & ~fixed_mask

        # Compute acceleration parameter for open fuel types
        self.accel_param = cp.where(
            fixed_mask,
            cp.full_like(fixed_mask, 0.115, dtype=self.cupy_float_type),
            self.accel_param
        )

        # Compute acceleration parameter for closed fuel types
        self.accel_param = cp.where(
            variable_mask,
            0.115 - 18.8 * cp.power(self.cfb, 2.5) * cp.exp(-8 * self.cfb),
            self.accel_param
        )

        # Clean up memory
        del fixed_mask, variable_mask

        return

    def calcFireType(self) -> None:
        """
        Function to calculate fire type using CuPy.
            - 1: surface
            - 2: intermittent crown
            - 3: active crown
        Applies only to fuels with numeric codes < 19.

        :return: None
        """
        self.fire_type = cp.where(
            self.fuel_type < 19,
            cp.where(
                self.cfb <= 0.1,
                # Surface fire
                1,
                cp.where(
                    (self.cfb > 0.1) & (self.cfb < 0.90),
                    # Intermittent crown fire
                    2,
                    cp.where(
                        self.cfb >= 0.90,
                        # Active crown fire
                        3,
                        # No fire type
                        0
                    )
                )
            ),
            cp.zeros_like(self.fuel_type),
        ).astype(cp.int8)

        return

    def calcCFC(self) -> None:
        """
        Function to calculate crown fuel consumed (kg/m^2) using CuPy.

        :return: None
        """
        self.cfc = cp.where(
            (self.fuel_type == 10) | (self.fuel_type == 11),
            self.cfb * self.cfl * self.pc / 100,
            cp.where(
                (self.fuel_type == 12) | (self.fuel_type == 13),
                self.cfb * self.cfl * self.pdf / 100,
                self.cfb * self.cfl
            )
        )

        return

    def calcC6hfros(self) -> None:
        """
        Function to calculate crown and total head fire rate of spread for the C6 fuel type using CuPy.

        :returns: None
        """
        self.cfros = cp.where(
            self.fuel_type == 6,
            cp.where(
                self.cfc == 0,
                cp.zeros_like(self.fuel_type),
                60 * cp.power(1 - cp.exp(-0.0497 * self.isi), 1) * (self.fme / 0.778237),
            ),
            self.cfros
        )

        self.hfros = cp.where(
            self.fuel_type == 6,
            self.sfros + (self.cfb * (self.cfros - self.sfros)),
            self.hfros
        )

        return

    def calcTFC(self) -> None:
        """
        Function to calculate total fuel consumed (kg/m^2) using CuPy.

        :return: None
        """
        self.tfc = self.sfc + self.cfc

        return

    def calcHFI(self) -> None:
        """
        Function to calculate head fire intensity.
        """
        self.hfi = 300 * self.hfros * self.tfc

        return

    def calcFireIntensityClass(self) -> None:
        """
        Function to calculate the fire intensity class based on fire intensity (FI).

        :return: None
        """
        self.fi_class = cp.where(
            (self.hfi > 0) & (self.hfi <= 10), 1,
            cp.where((self.hfi > 10) & (self.hfi <= 500), 2,
                     cp.where((self.hfi > 500) & (self.hfi <= 2000), 3,
                              cp.where((self.hfi > 2000) & (self.hfi <= 4000), 4,
                                       cp.where((self.hfi > 4000) & (self.hfi <= 10000), 5,
                                                cp.where((self.hfi > 10000), 6,
                                                         -99)
                                                )
                                       )
                              )
                     )
        )

        return

    def setParams(self, set_dict: dict) -> None:
        """
        Function to set FBP parameters to specific values using CuPy.

        :param set_dict: Dictionary of FBP parameter names and the values to assign to the FBP class object
        :return: None
        """
        # Iterate through the set dictionary and assign values
        for key, value in set_dict.items():
            if hasattr(self, key):  # Check if the class has the attribute
                if isinstance(value, cp.ndarray):  # Check if value is already a CuPy array
                    # Mask NaN values using CuPy
                    setattr(self, key, cp.where(cp.isnan(value), cp.nan, value))
                else:
                    # Convert scalar or list to a CuPy array and mask NaN values
                    value_array = (cp.asarray(value)
                                   if isinstance(value, (list, tuple, np.ndarray))
                                   else cp.asarray([value]))
                    setattr(self, key, cp.where(cp.isnan(value_array), cp.nan, value_array))
        return

    def getParams(self, out_request: list[str]) -> Union[list, str, None]:
        """
        Function to output requested dataset parameters from the FBP class using CuPy.

        :param out_request: List of requested FBP parameters.
        :return: List of requested outputs.
        """
        # Dictionary of CFFBPS parameters
        fbp_params = {
            # Default output variables
            'fire_type': self.fire_type,  # Type of fire (surface, intermittent crown, active crown)
            'hfros': self.hfros,  # Head fire rate of spread (m/min)
            'hfi': self.hfi,  # Head fire intensity (kW/m)

            # Fuel type variables
            'fuel_type': self.fuel_type,  # Fuel type codes

            # Weather variables
            'ws': self.ws,  # Observed wind speed (km/h)
            'wd': self.wd,  # Wind azimuth/direction (degrees)
            'm': self.m,  # Moisture content equivalent of the FFMC (%, value from 0-100+)
            'fF': self.fF,  # Fine fuel moisture function in the ISI equation
            'fW': self.fW,  # Wind function in the ISI equation
            'ffmc': self.ffmc,  # Fine fuel moisture code
            'bui': self.bui,  # Build-up index
            'isi': self.isi,  # Final calculated ISI, accounting for wind and slope

            # Slope + wind effect variables
            'a': self.a,  # Rate of spread equation coefficient
            'b': self.b,  # Rate of spread equation coefficient
            'c': self.c,  # Rate of spread equation coefficient
            'rsz': self.rsz,  # Surface spread rate with zero wind on level terrain
            'sf': self.sf,  # Slope factor
            'rsf': self.rsf,  # Spread rate with zero wind, upslope
            'isf': self.isf,  # ISI, with zero wind upslope
            'rsi': self.rsi,  # Initial spread rate without BUI effect
            'wse1': self.wse1,  # Original slope equivalent wind speed value for cases where WSE1 <= 40
            'wse2': self.wse2,  # New slope equivalent wind speed value for cases where WSE1 > 40
            'wse': self.wse,  # Slope equivalent wind speed
            'wsx': self.wsx,  # Net vectorized wind speed in the x-direction
            'wsy': self.wsy,  # Net vectorized wind speed in the y-direction
            'wsv': self.wsv,  # Net vectorized wind speed
            'raz': self.raz,  # Net vectorized wind direction

            # BUI effect variables
            'q': self.q,  # Proportion of maximum rate of spread at BUI equal to 50
            'bui0': self.bui0,  # Average BUI for each fuel type
            'be': self.be,  # Buildup effect on spread rate
            'be_max': self.be_max,  # Maximum allowable BE value

            # Surface fuel variables
            'ffc': self.ffc,  # Estimated forest floor consumption
            'wfc': self.wfc,  # Estimated woody fuel consumption
            'sfc': self.sfc,  # Estimated total surface fuel consumption

            # Foliar moisture content variables
            'latn': self.latn,  # Normalized latitude
            'dj': self.dj,  # Julian date of day being modeled
            'd0': self.d0,  # Julian date of minimum foliar moisture content
            'nd': self.nd,  # Number of days between modeled fire date and d0
            'fmc': self.fmc,  # Foliar moisture content
            'fme': self.fme,  # Foliar moisture effect

            # Critical crown fire threshold variables
            'csfi': self.csfi,  # Critical intensity (kW/m)
            'rso': self.rso,  # Critical rate of spread (m/min)

            # Back fire spread variables
            'bfw': self.bfW,  # The back fire wind function
            'bisi': self.bisi,  # The ISI associated with the back fire rate of spread
            'bros': self.bros,  # Backing rate of spread (m/min)

            # Crown fuel parameters
            'cbh': self.cbh,  # Height to live crown base (m)
            'cfb': self.cfb,  # Crown fraction burned (proportion, value ranging from 0-1)
            'cfl': self.cfl,  # Crown fuel load (kg/m^2)
            'cfc': self.cfc,  # Crown fuel consumed

            # Final fuel parameters
            'tfc': self.tfc,  # Total fuel consumed

            # Acceleration parameter
            'accel': self.accel_param,  # Acceleration parameter for point source ignition

            # Fire Intensity Class parameter
            'fi_class': self.fi_class,  # Fire intensity class (1-6)
        }

        # Retrieve requested parameters
        if self.return_array:
            if self.return_array_as == 'cupy':
                return [
                    cp.array(fbp_params.get(var)) if fbp_params.get(var) is not None
                    else 'Invalid output variable'
                    for var in out_request
                ]
            else:  # if self.return_array_as == 'numpy':
                return [
                    np.array(cp.asnumpy(fbp_params.get(var))) if fbp_params.get(var) is not None
                    else 'Invalid output variable'
                    for var in out_request
                ]

        else:
            return [
                cp.asnumpy(fbp_params.get(var)).item() if fbp_params.get(var).ndim == 0
                else cp.asnumpy(fbp_params.get(var))[0].item() if fbp_params.get(var) is not None
                else 'Invalid output variable'
                for var in out_request
            ]

    def runFBP(self, block: Optional[cp.ndarray] = None) -> list[any]:
        """
        Function to automatically run CFFBPS modeling using CuPy.

        :param block: The array of partial data (block) to run FBP with.
        :returns:
            List of values requested through the `out_request` parameter. Default values are `fire_type`, `hfros`, and `hfi`.
        """
        if not self.initialized:
            raise ValueError('FBP class must be initialized before running calculations. Call "initialize" first.')

        if block is not None:
            self.block = cp.asarray(block)

        # Check output requests values
        if self.out_request is None:
            # Set default output requests if none provided
            self.out_request = ['hfros', 'hfi', 'fire_type']

        # ### Model fire behavior with CFFBPS
        # Invert wind direction and aspect
        self.invertWindAspect()
        # Calculate slope factor
        self.calcSF()
        # Calculate zero slope & zero wind ISI
        self.calcISZ()
        # Calculate foliar moisture content
        self.calcFMC()
        # Calculate ISI, RSI, and BE
        self.calcISI_RSI_BE()
        # Calculate ROS
        self.calcROS()
        # Calculate surface fuel consumption
        self.calcSFC()
        # Calculate canopy base height and canopy fuel load
        self.getCBH_CFL()
        # Calculate critical surface fire intensity
        self.calcCSFI()
        # Calculate critical surface fire rate of spread
        self.calcRSO()
        # Calculate crown fraction burned
        self.calcCFB()
        # Calculate ROS percentile growth
        self.calcRosPercentileGrowth()
        # Calculate acceleration parameter
        self.calcAccelParam()
        # Calculate fire type
        self.calcFireType()
        # Calculate crown fuel consumed
        self.calcCFC()
        # Calculate C6 head fire rate of spread
        self.calcC6hfros()
        # Calculate total fuel consumption
        self.calcTFC()
        # Calculate head fire intensity
        self.calcHFI()
        # Calculate fire intensity class
        self.calcFireIntensityClass()

        # Return requested values
        return self.getParams(self.out_request)


def _get_cupy_array(raster_path):
    """
    Load a raster as a CuPy array using the specified backend (Rasterio or GDAL).

    :param raster_path: Path to the raster file.
    :return: CuPy array of the raster data.
    """
    with rio.open(raster_path) as src:
        return cp.asarray(src.read(1))


def _testFBP(test_functions: list,
             wx_date: int,
             lat: Union[float, int, cp.ndarray],
             long: Union[float, int, cp.ndarray],
             elevation: Union[float, int, cp.ndarray],
             slope: Union[float, int, cp.ndarray],
             aspect: Union[float, int, cp.ndarray],
             ws: Union[float, int, cp.ndarray],
             wd: Union[float, int, cp.ndarray],
             ffmc: Union[float, int, cp.ndarray],
             bui: Union[float, int, cp.ndarray],
             pc: Optional[Union[float, int, cp.ndarray]] = 50,
             pdf: Optional[Union[float, int, cp.ndarray]] = 35,
             gfl: Optional[Union[float, int, cp.ndarray]] = 0.35,
             gcf: Optional[Union[float, int, cp.ndarray]] = 80,
             d0: Optional[int] = None,
             dj: Optional[int] = None,
             out_request: Optional[list[str]] = None,
             out_folder: Optional[str] = None) -> None:
    """
    Function to test the cffbps module with various input types
    :param test_functions: List of functions to test
        (options: ['numeric', 'array', 'raster', 'raster_multiprocessing'])
    :param wx_date: Date of weather observation (used for fmc calculation) (YYYYMMDD)
    :param lat: Latitude of area being modelled (Decimal Degrees, floating point)
    :param long: Longitude of area being modelled (Decimal Degrees, floating point)
    :param elevation: Elevation of area being modelled (m)
    :param slope: Ground slope angle/tilt of area being modelled (%)
    :param aspect: Ground slope aspect/azimuth of area (degrees)
    :param ws: Wind speed (km/h @ 10m height)
    :param wd: Wind direction (degrees, direction wind is coming from)
    :param ffmc: CFFWIS Fine Fuel Moisture Code
    :param bui: CFFWIS Buildup Index
    :param pc: Percent conifer (%, value from 0-100)
    :param pdf: Percent dead fir (%, value from 0-100)
    :param gfl: Grass fuel load (kg/m^2)
    :param gcf: Grass curing factor (%, value from 0-100)
    :param d0: Julian date of minimum foliar moisture content (if None, calculated internally)
    :param dj: Julian date of day being modelled (if None, calculated internally)
    :param out_request: Tuple or list of CFFBPS output variables
        # Default output variables
        fire_type = Type of fire predicted to occur (surface, intermittent crown, active crown)
        hfros = Head fire rate of spread (m/min)
        hfi = head fire intensity (kW/m)

        # Weather variables
        ws = Observed wind speed (km/h)
        wd = Wind azimuth/direction (degrees)
        m = Moisture content equivalent of the FFMC (%, value from 0-100+)
        fF = Fine fuel moisture function in the ISI equation
        fW = Wind function in the ISI equation
        isi = Final ISI, accounting for wind and slope

        # Slope + wind effect variables
        a = Rate of spread equation coefficient
        b = Rate of spread equation coefficient
        c = Rate of spread equation coefficient
        RSZ = Surface spread rate with zero wind on level terrain
        SF = Slope factor
        RSF = spread rate with zero wind, upslope
        ISF = ISI, with zero wind upslope
        RSI = Initial spread rate without BUI effect
        WSE1 = Original slope equivalent wind speed value
        WSE2 = New slope equivalent wind speed value for cases where WSE1 > 40 (capped at max of 112.45)
        WSE = Slope equivalent wind speed
        WSX = Net vectorized wind speed in the x-direction
        WSY = Net vectorized wind speed in the y-direction
        WSV = (aka: slope-adjusted wind speed) Net vectorized wind speed (km/h)
        RAZ = (aka: slope-adjusted wind direction) Net vectorized wind direction (degrees)

        # BUI effect variables
        q = Proportion of maximum rate of spread at BUI equal to 50
        bui0 = Average BUI for each fuel type
        BE = Buildup effect on spread rate
        be_max = Maximum allowable BE value

        # Surface fuel variables
        ffc = Estimated forest floor consumption
        wfc = Estimated woody fuel consumption
        sfc = Estimated total surface fuel consumption

        # Foliar moisture content variables
        latn = Normalized latitude
        d0 = Julian date of minimum foliar moisture content
        nd = number of days between modelled fire date and d0
        fmc = foliar moisture content
        fme = foliar moisture effect

        # Critical crown fire threshold variables
        csfi = critical intensity (kW/m)
        rso = critical rate of spread (m/min)

        # Crown fuel parameters
        cbh = Height to live crown base (m)
        cfb = Crown fraction burned (proportion, value ranging from 0-1)
        cfl = Crown fuel load (kg/m^2)
        cfc = Crown fuel consumed
    :param out_folder: Location to save test rasters (Default: <location of script>/Test_Data/Outputs)
    :return: None
    """
    import sys

    import ProcessRasters as pr

    # generate_test_fbp_rasters moved from the repo root to tools/ (unpackaged);
    # make it importable when this script runs from the repo root.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools'))
    import generate_test_fbp_rasters as genras

    fbp = FBP()

    # Create fuel type list
    fuel_type_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'D1', 'D2', 'M1', 'M2', 'M3', 'M4',
                      'O1a', 'O1b', 'S1', 'S2', 'S3', 'NF', 'WA']

    # Put inputs into list
    input_data = [wx_date, lat, long,
                  elevation, slope, aspect, ws, wd, ffmc, bui,
                  pc, pdf, gfl, gcf, d0, dj, out_request]

    # ### Test non-raster modelling
    if any(var in test_functions for var in ['numeric', 'all']):
        print('Testing non-raster modelling')
        for ft in fuel_type_list:
            fbp.initialize(*([fbpFTCode_AlphaToNum_LUT.get(ft)] + input_data))
            print('\t' + ft, fbp.runFBP())

    # ### Test array modelling
    if any(var in test_functions for var in ['array', 'all']):
        print('Testing array modelling')
        fbp.initialize(*([cp.array([fbpFTCode_AlphaToNum_LUT.get(ft) for ft in fuel_type_list])] + input_data))
        print('\t', fbp.runFBP())

    # Get test folders
    input_folder = os.path.join(os.path.dirname(__file__), 'tests', 'cffbps', 'data', 'inputs')
    multiprocess_folder = os.path.join(input_folder, 'multiprocessing')
    if out_folder is None:
        output_folder = os.path.join(os.path.dirname(__file__), 'tests', 'cffbps', 'data', 'outputs', 'cupy')
    else:
        output_folder = out_folder
    os.makedirs(output_folder, exist_ok=True)

    # ### Test simple raster GPU processing
    if any(var in test_functions for var in ['raster', 'all']):
        print('Testing simple raster GPU processing')
        # Generate test raster datasets using user-provided input values
        genras.gen_test_data(*input_data[:-3], dtype=cp.float32)

        # Get input dataset paths
        raster_paths = {
            'fuel_type': os.path.join(input_folder, 'FuelType.tif'),
            'lat': os.path.join(input_folder, 'LAT.tif'),
            'long': os.path.join(input_folder, 'LONG.tif'),
            'elevation': os.path.join(input_folder, 'ELV.tif'),
            'slope': os.path.join(input_folder, 'GS.tif'),
            'aspect': os.path.join(input_folder, 'Aspect.tif'),
            'ws': os.path.join(input_folder, 'WS.tif'),
            'wd': os.path.join(input_folder, 'WD.tif'),
            'ffmc': os.path.join(input_folder, 'FFMC.tif'),
            'bui': os.path.join(input_folder, 'BUI.tif'),
            'pc': os.path.join(input_folder, 'PC.tif'),
            'pdf': os.path.join(input_folder, 'PDF.tif'),
            'gfl': os.path.join(input_folder, 'GFL.tif'),
            'gcf': os.path.join(input_folder, 'cc.tif'),
        }

        # Create a reference raster profile for final raster outputs
        with rio.open(raster_paths['gfl']) as src:
            ref_ras_profile = src.profile

        # Read raster data into CuPy arrays
        raster_data = {key: cp.asarray(rio.open(path).read()) for key, path in raster_paths.items()}

        # Run the FBP modeling
        fbp.initialize(
            fuel_type=raster_data['fuel_type'], wx_date=wx_date,
            lat=raster_data['lat'], long=raster_data['long'], elevation=raster_data['elevation'],
            slope=raster_data['slope'], aspect=raster_data['aspect'],
            ws=raster_data['ws'], wd=raster_data['wd'], ffmc=raster_data['ffmc'],
            bui=raster_data['bui'], pc=raster_data['pc'], pdf=raster_data['pdf'],
            gfl=raster_data['gfl'], gcf=raster_data['gcf'],
            d0=d0, dj=dj,
            out_request=out_request,
            convert_fuel_type_codes=False
        )
        fbp_result = fbp.runFBP()

        # Get output dataset paths
        output_rasters = [
            os.path.join(output_folder, name + '.tif')
            for name in out_request
        ]

        for dset, path in zip(fbp_result, output_rasters):
            # Save output datasets
            pr.arrayToRaster(array=cp.asnumpy(dset),  # Convert to NumPy for rasterio compatibility
                             out_file=path,
                             ras_profile=ref_ras_profile,
                             dtype=np.float32)

    # ### Test larger raster GPU processing
    if any(var in test_functions for var in ['raster_multiprocessing', 'all']):
        print('Testing larger raster GPU processing')
        if not os.path.exists(os.path.join(output_folder, 'gpu_large_raster')):
            os.mkdir(os.path.join(output_folder, 'gpu_large_raster'))

        # Get input dataset paths
        raster_paths = {
            'fuel_type': os.path.join(multiprocess_folder, 'FuelType.tif'),
            'lat': os.path.join(multiprocess_folder, 'LAT.tif'),
            'long': os.path.join(multiprocess_folder, 'LONG.tif'),
            'elevation': os.path.join(multiprocess_folder, 'ELV.tif'),
            'slope': os.path.join(multiprocess_folder, 'GS.tif'),
            'aspect': os.path.join(multiprocess_folder, 'Aspect.tif'),
            'ws': os.path.join(multiprocess_folder, 'WS.tif'),
            # 'wd': os.path.join(multiprocess_folder, 'WD.tif'),
            # 'ffmc': os.path.join(multiprocess_folder, 'FFMC.tif'),
            # 'bui': os.path.join(multiprocess_folder, 'BUI.tif'),
            'pc': os.path.join(multiprocess_folder, 'PC.tif'),
            'pdf': os.path.join(multiprocess_folder, 'PDF.tif'),
            'gfl': os.path.join(multiprocess_folder, 'GFL.tif'),
            # 'gcf': os.path.join(multiprocess_folder, 'cc.tif'),
        }

        # Create a reference raster profile for final raster outputs
        with rio.open(raster_paths['gfl']) as src:
            ref_ras_profile = src.profile

        # Read raster data into CuPy arrays
        # Check dtype and conditionally convert datasets > float32 to float32
        raster_data = {}
        for key, path in raster_paths.items():
            with rio.open(path) as src:
                # Get the dtype of the raster
                raster_dtype = src.dtypes[0]  # Assuming single-band rasters; use indexing for multi-band rasters
                # Read raster data
                raster_array = src.read()
                # Conditionally convert to float32
                if np.dtype(raster_dtype).itemsize > np.dtype(np.float32).itemsize:
                    raster_data[key] = cp.asarray(raster_array.astype(np.float32))
                else:
                    raster_data[key] = cp.asarray(raster_array)

        # Run the FBP modeling
        fbp.initialize(
            fuel_type=raster_data['fuel_type'], wx_date=wx_date,
            lat=raster_data['lat'], long=raster_data['long'], elevation=raster_data['elevation'],
            slope=raster_data['slope'], aspect=raster_data['aspect'],
            ws=raster_data['ws'], wd=wd, ffmc=ffmc,
            bui=bui, pc=raster_data['pc'], pdf=raster_data['pdf'],
            gfl=raster_data['gfl'], gcf=getSeasonGrassCuring(season='summer', province='BC'),
            d0=d0, dj=dj,
            out_request=out_request,
            convert_fuel_type_codes=True
        )
        fbp_result = fbp.runFBP()

        # Get output dataset paths
        output_rasters = [
            os.path.join(output_folder, 'gpu_large_raster', name + '.tif')
            for name in out_request
        ]

        for dset, path in zip(fbp_result, output_rasters):
            dset_profile = ref_ras_profile
            if dset.dtype.itemsize > cp.float32().itemsize:
                # Set raster profile
                dset_profile['nodata'] = np.finfo(np.float32).max
                dtype = np.float32
            else:
                dtype = dset.dtype
            # Save output datasets
            pr.arrayToRaster(array=dset,  # Convert to NumPy for rasterio compatibility
                             out_file=path,
                             ras_profile=dset_profile,
                             dtype=dtype)


if __name__ == '__main__':
    # _test_functions options: ['all', 'numeric', 'array', 'raster', 'raster_multiprocessing']
    _test_functions = ['all']
    _wx_date = 20160516
    _lat = 62.245533
    _long = -133.840363
    _elevation = 1180
    _slope = 8
    _aspect = 60
    _ws = 24
    _wd = 266
    _ffmc = 92
    _bui = 31
    _pc = 50
    _pdf = 50
    _gfl = 0.35
    _gcf = 80
    _d0 = None
    _dj = None
    _out_request = ['wsv', 'raz', 'isi', 'rsi', 'sfc', 'csfi', 'rso', 'cfb', 'hfros', 'hfi', 'fire_type', 'fi_class']
    _out_folder = None

    # Test the FBP functions
    _testFBP(test_functions=_test_functions,
             wx_date=_wx_date, lat=_lat, long=_long,
             elevation=_elevation, slope=_slope, aspect=_aspect,
             ws=_ws, wd=_wd, ffmc=_ffmc, bui=_bui,
             pc=_pc, pdf=_pdf, gfl=_gfl, gcf=_gcf,
             d0=_d0, dj=_dj,
             out_request=_out_request,
             out_folder=_out_folder)
