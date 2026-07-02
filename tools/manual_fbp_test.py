"""Manual, raster-based smoke test for CFFBPS (NOT part of the package or the pytest suite).

Requires the external `ProcessRasters` module and `generate_test_fbp_rasters`
(both excluded from the installable package). Retained
for reference / interactive use. The automated regression gate lives in
tests/cffbps/test_fbp_regression.py.
"""
import os

import numpy as np

from cffdrs.cffbps import (
    FBP,
    fbpFTCode_AlphaToNum_LUT,
    fbpMultiprocessArray,
    getSeasonGrassCuring,
)

# This script lives in tools/; the raster fixtures live at the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _testFBP(test_functions: list,
             wx_date: int,
             lat: float | int | np.ndarray,
             long: float | int | np.ndarray,
             elevation: float | int | np.ndarray,
             slope: float | int | np.ndarray,
             aspect: float | int | np.ndarray,
             ws: float | int | np.ndarray,
             wd: float | int | np.ndarray,
             ffmc: float | int | np.ndarray,
             bui: float | int | np.ndarray,
             pc: float | int | np.ndarray | None = 50,
             pdf: float | int | np.ndarray | None = 35,
             gfl: float | int | np.ndarray | None = 0.35,
             gcf: float | int | np.ndarray | None = 80,
             d0: int | None = None,
             dj: int | None = None,
             out_request: list[str] | None = None,
             out_folder: str | None = None,
             num_processors: int = 2,
             block_size: int | None = None) -> None:
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
    :param d0: Julian date of minimum foliar moisture content (if None, will be calculated based on latitude)
    :param dj: Julian date of the day being modelled (if None, will be
    :param out_request: Tuple or list of CFFBPS output variables
        # Default output variables
        fire_type = Type of fire predicted to occur (surface, intermittent crown, active crown)
        hros = Head fire rate of spread (m/min)
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
    :param num_processors: Number of cores for multiprocessing
    :param block_size: Size of blocks (# raster cells) for multiprocessing
    :return: None
    """
    import ProcessRasters as pr

    import generate_test_fbp_rasters as genras

    # Create fuel type list
    fuel_type_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'D1', 'D2', 'M1', 'M2', 'M3', 'M4',
                      'O1a', 'O1b', 'S1', 'S2', 'S3', 'NF', 'WA']

    # Put inputs into list
    input_data = [wx_date, lat, long,
                  elevation, slope, aspect, ws, wd, ffmc, bui,
                  pc, pdf, gfl, gcf, d0, dj, out_request]

    # ### Test non-raster modelling
    if any(var in test_functions for var in ['numeric', 'all']):
        fbp = FBP()
        print('Testing non-raster modelling')
        for ft in fuel_type_list:
            fbp.initialize(*([fbpFTCode_AlphaToNum_LUT.get(ft)] + input_data))
            print('\t' + ft, fbp.runFBP())

    # ### Test array modelling
    if any(var in test_functions for var in ['array', 'all']):
        fbp = FBP()
        print('Testing array modelling')
        fbp.initialize(*([np.array(fuel_type_list)] + input_data))
        print('\t', fbp.runFBP())

    # Get test folders (fixtures live at the repo root, not under tools/)
    input_folder = os.path.join(_REPO_ROOT, 'tests', 'cffbps', 'data', 'inputs')
    multiprocess_folder = os.path.join(input_folder, 'multiprocessing')
    if out_folder is None:
        output_folder = os.path.join(_REPO_ROOT, 'tests', 'cffbps', 'data', 'outputs')
    else:
        output_folder = out_folder
    os.makedirs(output_folder, exist_ok=True)

    # ### Test raster modelling
    if any(var in test_functions for var in ['raster', 'all']):
        print('Testing raster modelling')
        # Generate test raster datasets using user-provided input values
        genras.gen_test_data(*input_data[:-3], dtype=np.float64)

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
        ref_ras_profile = pr.getRaster(raster_paths['gfl']).profile

        # Read raster data into CuPy arrays
        raster_data = {key: pr.getRaster(path).read() for key, path in raster_paths.items()}

        # Generate the output request
        out_request = ['wsv', 'raz', 'fire_type', 'hfi', 'hros', 'bros', 'ffc', 'wfc', 'sfc']

        # Run the FBP modeling
        fbp = FBP()
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
        out_path_list = [
            os.path.join(output_folder, name + '.tif') for name in out_request
        ]

        for dset, path in zip(fbp_result, out_path_list):
            if any(f'{name}.tif' in path for name in ['fuel_type', 'fire_type']):
                dtype = np.int8
            else:
                dtype = np.float64

            # Convert dset to dtype
            dset = dset.astype(dtype)

            # Save output datasets
            pr.arrayToRaster(array=dset,
                             out_file=path,
                             ras_profile=ref_ras_profile,
                             dtype=dtype)

    # ### Test raster multiprocessing
    if any(var in test_functions for var in ['raster_multiprocessing', 'all']):
        print('Testing raster multiprocessing')
        if not os.path.exists(os.path.join(output_folder, 'multiprocessing')):
            os.mkdir(os.path.join(output_folder, 'multiprocessing'))

        # Get input dataset paths
        fuel_type_path = os.path.join(multiprocess_folder, 'FuelType.tif')
        lat_path = os.path.join(multiprocess_folder, 'LAT.tif')
        long_path = os.path.join(multiprocess_folder, 'LONG.tif')
        elev_path = os.path.join(multiprocess_folder, 'ELV.tif')
        slope_path = os.path.join(multiprocess_folder, 'GS.tif')
        aspect_path = os.path.join(multiprocess_folder, 'Aspect.tif')
        ws_path = os.path.join(multiprocess_folder, 'WS.tif')
        # wd_path = os.path.join(multiprocess_folder, 'WD.tif')
        # ffmc_path = os.path.join(multiprocess_folder, 'FFMC.tif')
        # bui_path = os.path.join(multiprocess_folder, 'BUI.tif')
        pc_path = os.path.join(multiprocess_folder, 'PC.tif')
        pdf_path = os.path.join(multiprocess_folder, 'PDF.tif')
        gfl_path = os.path.join(multiprocess_folder, 'GFL.tif')
        # gcf_path = os.path.join(multiprocess_folder, 'cc.tif')

        # Create a reference raster profile for final raster outputs
        ref_ras_profile = pr.getRaster(gfl_path).profile

        # Get input dataset arrays
        fuel_type_array = pr.getRaster(fuel_type_path).read()
        lat_array = pr.getRaster(lat_path).read()
        long_array = pr.getRaster(long_path).read()
        elev_array = pr.getRaster(elev_path).read()
        slope_array = pr.getRaster(slope_path).read()
        aspect_array = pr.getRaster(aspect_path).read()
        ws_array = pr.getRaster(ws_path).read()
        # wd_array = pr.getRaster(wd_path).read()
        # ffmc_array = pr.getRaster(ffmc_path).read()
        # bui_array = pr.getRaster(bui_path).read()
        pc_array = pr.getRaster(pc_path).read()
        pdf_array = pr.getRaster(pdf_path).read()
        gfl_array = pr.getRaster(gfl_path).read()
        # gcf_array = pr.getRaster(gcf_path).read()

        # Generate the output request
        out_request = ['wsv', 'raz', 'fire_type', 'hfi', 'hros', 'bros', 'ffc', 'wfc', 'sfc']

        # Run the FBP multiprocessing
        fbp_multiprocess_result = fbpMultiprocessArray(
            fuel_type=fuel_type_array, wx_date=wx_date, lat=lat_array, long=long_array,
            elevation=elev_array, slope=slope_array, aspect=aspect_array,
            ws=ws_array, wd=wd, ffmc=ffmc, bui=bui,
            pc=pc_array, pdf=pdf_array, gfl=gfl_array, gcf=getSeasonGrassCuring(season='summer', province='BC'),
            d0=d0, dj=dj,
            out_request=out_request,
            convert_fuel_type_codes=True,
            num_processors=num_processors,
            block_size=block_size
        )

        # Get output dataset paths
        out_path_list = [
            os.path.join(output_folder, 'multiprocessing', name + '.tif') for name in out_request
        ]

        for dset, path in zip(fbp_multiprocess_result, out_path_list):
            if any(f'{name}.tif' in path for name in ['fuel_type', 'fire_type']):
                dtype = np.int8
            else:
                dtype = np.float64

            # Convert dset to dtype
            dset = dset.astype(dtype)

            # Save output datasets
            pr.arrayToRaster(array=dset,
                             out_file=path,
                             ras_profile=ref_ras_profile,
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
    _out_request = ['bros', 'wsv', 'raz', 'isi', 'rsi', 'sfc', 'csfi', 'rso', 'cfb', 'hros', 'hfi', 'fire_type', 'fi_class']
    _out_folder = None
    _num_processors = os.cpu_count() - 1 if os.cpu_count() > 2 else 2
    _block_size = None

    # Test the FBP functions
    _testFBP(test_functions=_test_functions,
             wx_date=_wx_date, lat=_lat, long=_long,
             elevation=_elevation, slope=_slope, aspect=_aspect,
             ws=_ws, wd=_wd, ffmc=_ffmc, bui=_bui,
             pc=_pc, pdf=_pdf, gfl=_gfl, gcf=_gcf,
             d0=_d0, dj=_dj,
             out_request=_out_request,
             out_folder=_out_folder,
             num_processors=_num_processors,
             block_size=_block_size)
