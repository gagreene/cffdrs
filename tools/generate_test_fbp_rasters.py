"""
Created on Thur Aug 8 17:30:00 2024

@author: Gregory A. Greene
"""

import os
from datetime import datetime as dt

import numpy as np
import ProcessRasters as pr

# Get input folder (this script lives in tools/; the fixtures live at the repo root)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_folder = os.path.join(_REPO_ROOT, 'tests', 'cffbps', 'data', 'inputs')

# Get input dataset paths
fuel_type_path = os.path.join(input_folder, 'FuelType.tif')

# Get input data
fuel_type_ras = pr.getRaster(fuel_type_path)
fuel_type_profile = fuel_type_ras.profile
fuel_type_array = fuel_type_ras.read()


def gen_test_data(wx_date: int,
                  lat: float | int,
                  long: float | int,
                  elevation: float | int,
                  slope: float | int,
                  aspect: float | int,
                  ws: float | int,
                  wd: float | int,
                  ffmc: float | int,
                  bui: float | int,
                  pc: float | int | None = 50,
                  pdf: float | int | None = 35,
                  gfl: float | int | None = 0.35,
                  gcf: float | int | None = 80,
                  dtype=np.float32):
    # ### VERIFY ALL INPUTS
    # Verify wx_date
    if not isinstance(wx_date, int):
        raise TypeError('wx_date must be either int or numpy ndarray data types')
    try:
        date_string = str(wx_date)
        dt.fromisoformat(f'{date_string[:4]}-{date_string[4:6]}-{date_string[6:]}')
    except ValueError:
        raise ValueError('wx_date must be formatted as: YYYYMMDD')

    # Verify lat
    if not isinstance(lat, (int, float)):
        raise TypeError('lat must be either int or float data types')

    # Verify long
    if not isinstance(long, (int, float)):
        raise TypeError('long must be either int or float data types')

    # Verify elevation
    if not isinstance(elevation, (int, float)):
        raise TypeError('elevation must be either int or float data types')

    # Verify slope
    if not isinstance(slope, (int, float)):
        raise TypeError('slope must be either int or float data types')

    # Verify aspect
    if not isinstance(aspect, (int, float)):
        raise TypeError('aspect must be either int or float data types')

    # Verify ws
    if not isinstance(ws, (int, float)):
        raise TypeError('ws must be either int or float data types')

    # Verify wd
    if not isinstance(wd, (int, float)):
        raise TypeError('wd must be either int or float data types')

    # Verify ffmc
    if not isinstance(ffmc, (int, float)):
        raise TypeError('ffmc must be either int or float data types')

    # Verify bui
    if not isinstance(bui, (int, float)):
        raise TypeError('bui must be either int or float data types')

    # Verify pc
    if not isinstance(pc, (int, float)):
        raise TypeError('pc must be either int or float data types')

    # Verify pdf
    if not isinstance(pdf, (int, float)):
        raise TypeError('pdf must be either int or float data types')

    # Verify gfl
    if not isinstance(gfl, (int, float)):
        raise TypeError('gfl must be either int or float data types')

    # Verify gcf
    if not isinstance(gcf, (int, float)):
        raise TypeError('gcf must be either int or float data types')

    # Generate output dataset dictionary
    data_dict = {
        'Dj': dt.strptime(str(wx_date), '%Y%m%d').timetuple().tm_yday,
        'LAT': lat,
        'LONG': long,
        'ELV': elevation,
        'GS': slope,
        'Aspect': aspect,
        'WS': ws,
        'WD': wd,
        'FFMC': ffmc,
        'BUI': bui,
        'PC': pc,
        'PDF': pdf,
        'GFL': gfl,
        'cc': gcf
    }

    for dset in list(data_dict.keys()):
        dset_array = fuel_type_array.astype(np.float64) * 0 + data_dict.get(dset)
        # Save output datasets
        pr.arrayToRaster(array=dset_array,
                         out_file=os.path.join(input_folder, f'{dset}.tif'),
                         ras_profile=fuel_type_profile,
                         dtype=dtype)


if __name__ == '__main__':
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
    _pc = 0
    _pdf = 0
    _gfl = 0
    _gcf = 60

    # Generate FBP rasters
    gen_test_data(wx_date=_wx_date, lat=_lat, long=_long,
                  elevation=_elevation, slope=_slope, aspect=_aspect,
                  ws=_ws, wd=_wd, ffmc=_ffmc, bui=_bui,
                  pc=_pc, pdf=_pdf, gfl=_gfl, gcf=_gcf)
