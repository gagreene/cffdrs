"""Block-based multiprocessing driver for CFFBPS raster runs.

Splits large input arrays into blocks and runs :class:`cffbps.facade.FBP` on each
block in a worker pool. Kept separate from the pure equation modules because it
owns process orchestration (not referentially transparent).
"""
from __future__ import annotations

from multiprocessing import Pool, current_process
from operator import itemgetter
import warnings

import numpy as np
import psutil

from .facade import FBP


def _estimate_optimal_block_size(array_shape, num_processors, memory_fraction=0.8):
    # Total available memory
    available_memory = psutil.virtual_memory().available * memory_fraction

    # Estimate memory needed for one block
    element_size = np.dtype(np.float64).itemsize  # Assuming float64 data type

    # Calculate the maximum possible block size based on available memory and the number of processors
    max_block_size = int(np.sqrt(available_memory / (element_size * array_shape[0] * num_processors)))

    # Ensure block size is practical and does not exceed array dimensions
    block_size = min(max_block_size, array_shape[1], array_shape[2])

    # If block size exceeds a reasonable portion of the array, reduce it further
    while block_size > 0 and block_size > array_shape[1] // 4 and block_size > array_shape[2] // 4:
        block_size //= 2

    # Never return a degenerate block size: the halving loop reaches 0 for arrays
    # with both spatial dims < 4 (and _gen_blocks would crash on stride=0).
    return max(1, block_size)


def _gen_blocks(array: np.ndarray, block_size: int, stride: int) -> tuple:
    blocks = []
    block_positions = []
    layers, rows, cols = array.shape

    for i in range(0, rows, stride):
        for j in range(0, cols, stride):
            # Adjust block size for edge cases
            end_i = min(i + block_size, rows)
            end_j = min(j + block_size, cols)

            # Extract the block, keeping all layers
            block = array[:, i:end_i, j:end_j]
            blocks.append(block)
            block_positions.append((i, j))  # Save the top-left position of each block

    return blocks, block_positions


def _process_block(block: tuple, position: tuple) -> tuple:
    # Get ID of the multiprocessing Pool Worker
    process_id = current_process().name
    print(f'\t\t[{process_id}] Processing Block at Cell {position}')

    # Get top-left cell position
    row, col = position

    # Initialize FBP class with parameters
    fbp = FBP()
    fbp.initialize(*block)

    # Process the block and return results
    result = fbp.runFBP()

    return result, (row, col)


def fbpMultiprocessArray(fuel_type: int | str | np.ndarray,
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
                         convert_fuel_type_codes: bool | None = False,
                         num_processors: int = 2,
                         block_size: int | None = None) -> list:
    """
    Function breaks input arrays into blocks and processes each block with a different worker/processor.
    Uses the runFBP function in the FBP class.
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
    :param dj: Julian date of the day being modelled (if None, will be calculated from wx_date)
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
    :param convert_fuel_type_codes: Convert from CFS cffdrs R fuel type grid codes
        to the grid codes used in this module
    :param num_processors: Number of cores for multiprocessing
    :param block_size: Size of blocks (# raster cells) for multiprocessing.
        If block_size is None, an optimal block size will be estimated automatically.
    :return: Concatenated output array from all workers
    """
    # Apply the same default output request runFBP uses; the output-array sizing
    # below iterates out_request, so a None here would otherwise raise TypeError.
    if out_request is None:
        out_request = ['hros', 'hfi', 'fire_type']

    # Add input parameters to list
    input_list = [fuel_type, wx_date, lat, long, elevation, slope, aspect,
                  ws, wd, ffmc, bui, pc, pdf, gfl, gcf, d0, dj, out_request,
                  convert_fuel_type_codes]

    # Split input arrays into chunks for each worker
    array_indices = [i for i in range(len(input_list)) if isinstance(input_list[i], np.ndarray)]
    nonarray_indices = [i for i in range(len(input_list)) if i not in array_indices]
    array_list = list(itemgetter(*array_indices)(input_list))

    # Verify there is at least one input array
    if len(array_indices) == 0:
        raise ValueError('Unable to use the multiprocessing function. There are no arrays in the inputs')

    # If more than one array, verify they are all the same shape
    if len(array_indices) > 1:
        shapes = {arr.shape for arr in array_list}
        if len(shapes) > 1:
            raise ValueError(f'All arrays must have the same dimensions. '
                             f'The following range of dimensions exists: {shapes}')

    # Verify num_processors is greater than 1
    if num_processors < 2:
        num_processors = 2
        warnings.warn(
            'Multiprocessing requires at least two cores. '
            'Defaulting num_processors to 2 for this run.',
            stacklevel=2,
        )

    # Verify block size
    if block_size is None:
        block_size = _estimate_optimal_block_size(array_shape=array_list[0].shape,
                                                  num_processors=num_processors)

    # Split input arrays into blocks and track their positions
    array_blocks = []
    block_positions = None  # Will hold the block positions from the first array

    for array in array_list:
        blocks, positions = _gen_blocks(array=array, block_size=block_size, stride=block_size)
        array_blocks.append(blocks)
        if block_positions is None:
            block_positions = positions

    # Generate final input_block list for multiprocessing
    input_blocks = []
    num_blocks = len(array_blocks[0])  # Number of blocks should be the same for all arrays

    for idx in range(num_blocks):
        block_set = [array_blocks[i][idx] for i in range(len(array_blocks))]
        row = [None] * len(input_list)

        # Assign blocks to the correct indices
        for i, block in zip(array_indices, block_set, strict=False):
            row[i] = block

        # Assign non-array inputs
        for i in nonarray_indices:
            row[i] = input_list[i]

        input_blocks.append((row, block_positions[idx]))  # Attach the position to each block

    del array_list

    output_arrays = []
    for _ in out_request:
        output_arrays.append(np.zeros(input_list[array_indices[0]].shape, dtype=np.float64))

    # Initialize a multiprocessing pool
    with Pool(num_processors) as pool:
        try:
            print('\tStarting FBP multiprocessing...')
            # Process each block using runFBP in parallel
            results = pool.starmap(_process_block, input_blocks)
        finally:
            pool.close()  # Stop accepting new tasks
            pool.join()  # Wait for all tasks to finish

    # Place the processed blocks back into the output array
    for result, (i, j) in results:
        for idx, _ in enumerate(out_request):
            result_shape = result[idx].shape
            slice_i_end = i + result_shape[1]
            slice_j_end = j + result_shape[2]

            output_arrays[idx][:, i:slice_i_end, j:slice_j_end] = result[idx]

    return output_arrays
