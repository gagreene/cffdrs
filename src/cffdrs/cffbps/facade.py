"""
Created on Mon July 22 10:00:00 2024

@author: Gregory A. Greene
"""
__author__ = ['Gregory A. Greene, map.n.trowel@gmail.com']


import numpy as np
from numpy import ma as mask

from . import constants, inputs
from .equations import consumption as consumption_eq
from .equations import crown as crown_eq
from .equations import fmc as fmc_eq
from .equations import growth as growth_eq
from .equations import ros as ros_eq
from .equations import slope_wind as sw_eq
from .equations import surface as surface_eq


def getSeasonGrassCuring(season: str,
                         province: str,
                         subregion: str | None = None) -> int | None:
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


##################################################################################################
# #### CLASS FOR CANADIAN FOREST FIRE BEHAVIOR PREDICTION SYSTEM (CFFBPS) MODELLING ####
##################################################################################################
class FBP:
    """
    Class to model fire behavior with the Canadian Forest Fire Behavior Prediction System.
    """

    def __init__(self):
        # Initialize CFFBPS input parameters
        self.fuel_type = None
        self.wx_date = None
        self.lat = None
        self.long = None
        self.elevation = None
        self.slope = None
        self.aspect = None
        self.ws = None
        self.wd = None
        self.ffmc = None
        self.bui = None
        self.pc = None
        self.pdf = None
        self.gfl = None
        self.gcf = None
        self.d0 = None
        self.dj = None
        self.out_request = None
        self.convert_fuel_type_codes = False
        self.percentile_growth = None

        # Array verification parameter
        self.return_array = None
        self.ref_array = None
        self.initialized = False

        # Initialize multiprocessing block variable
        self.block = None

        # Initialize unique fuel types list
        self.ftypes = None

        # Initialize weather parameters
        self.isi = None
        self.m = None
        self.fF = None
        self.fW = None

        # Initialize slope effect parameters
        self.a = None
        self.b = None
        self.c = None
        self.rsz = None
        self.isz = None
        self.sf = None
        self.rsf = None
        self.isf = None
        self.rsi = None
        self.wse1 = None
        self.wse2 = None
        self.wse = None
        self.wsx = None
        self.wsy = None
        self.wsv = None
        self.raz = None

        # Initialize BUI effect parameters
        self.q = None
        self.bui0 = None
        self.be = None
        self.be_max = None

        # Initialize surface parameters
        self.cf = None
        self.ffc = None
        self.wfc = None
        self.sfc = None
        self.rss = None

        # Initialize foliar moisture content parameters
        self.latn = None
        self.nd = None
        self.fmc = None
        self.fme = None

        # Initialize crown and total fuel consumed parameters
        self.cbh = None
        self.csfi = None
        self.rso = None
        self.rsc = None
        self.cfb = None
        self.bros_cfb = None
        self.cfl = None
        self.cfc = None
        self.tfc = None

        # Initialize the backing fire rate of spread parameters
        self.bfW = None
        self.brsi = None
        self.bisi = None
        self.bros = None

        # Initialize default CFFBPS output parameters
        self.fire_type = None
        self.hros = None
        self.hfi = None

        # Initialize C-6 rate of spread parameters
        self.sros = None
        self.cros = None

        # Initialize point ignition acceleration parameter
        self.accel_param = None

        # Initialize fire intensity class parameter
        self.fi_class = None

        # ### CFFBPS reference tables — per-instance mutable copies of the read-only
        # module constants (see cffbps.constants). Instance-local copies preserve the
        # historical ability to calibrate one FBP instance (e.g. tweak rosParams for
        # a sensitivity run) without leaking the change into other instances or the
        # shared module tables.
        # Lists for CFFBPS Crown Fire Metric variables
        self.csfiVarList = list(constants.csfiVarList)
        self.rsoVarList = list(constants.rsoVarList)
        self.cfbVarList = list(constants.cfbVarList)
        self.cfcVarList = list(constants.cfcVarList)
        self.cfiVarList = list(constants.cfiVarList)

        # List of open fuel type codes
        self.open_fuel_types = list(constants.open_fuel_types)

        # List of non-crowning fuel type codes
        self.non_crowning_fuels = list(constants.non_crowning_fuels)

        # CFFBPS Canopy Base Height & Canopy Fuel Load Lookup Table (cbh, cfl, ht)
        self.fbpCBH_CFL_HT_LUT = dict(constants.fbpCBH_CFL_HT_LUT)

        # CFFBPS Surface Fire Rate of Spread Parameters (a, b, c, q, BUI0, be_max)
        self.rosParams = dict(constants.rosParams)

    def _checkArray(self) -> None:
        """
        Detect array inputs and build the reference arrays (delegates to
        :func:`cffbps.inputs.check_array`).
        :return: None
        """
        input_list = [
            self.fuel_type, self.lat, self.long,
            self.elevation, self.slope, self.aspect,
            self.ws, self.wd, self.ffmc, self.bui,
            self.pc, self.pdf,
            self.gfl, self.gcf
        ]
        self.return_array, self.ref_array, self.ref_int_array = inputs.check_array(input_list)
        return

    def _verifyInputs(self) -> None:
        """
        Validate all inputs and coerce them to masked numpy arrays (delegates to
        :func:`cffbps.inputs.verify_inputs`), writing the results back onto self.
        :return: None
        """
        validated = inputs.verify_inputs(
            fuel_type=self.fuel_type, wx_date=self.wx_date,
            lat=self.lat, long=self.long, elevation=self.elevation,
            slope=self.slope, aspect=self.aspect, ws=self.ws, wd=self.wd,
            ffmc=self.ffmc, bui=self.bui, pc=self.pc, pdf=self.pdf,
            gfl=self.gfl, gcf=self.gcf, d0=self.d0, dj=self.dj,
            out_request=self.out_request,
            convert_fuel_type_codes=self.convert_fuel_type_codes,
        )
        # VerifiedInputs field names match the facade attribute names by contract.
        for key, value in validated._asdict().items():
            setattr(self, key, value)

    def initialize(self,
                   fuel_type: int | str | np.ndarray | None = None,
                   wx_date: int | None = None,
                   lat: float | int | np.ndarray | None = None,
                   long: float | int | np.ndarray | None = None,
                   elevation: float | int | np.ndarray | None = None,
                   slope: float | int | np.ndarray | None = None,
                   aspect: float | int | np.ndarray | None = None,
                   ws: float | int | np.ndarray | None = None,
                   wd: float | int | np.ndarray | None = None,
                   ffmc: float | int | np.ndarray | None = None,
                   bui: float | int | np.ndarray | None = None,
                   pc: float | int | np.ndarray | None = 50,
                   pdf: float | int | np.ndarray | None = 35,
                   gfl: float | int | np.ndarray | None = 0.35,
                   gcf: float | int | np.ndarray | None = 80,
                   d0: int | None = None,
                   dj: int | None = None,
                   out_request: list | tuple | None = None,
                   convert_fuel_type_codes: bool | None = False,
                   percentile_growth: float | int | None = 50) -> None:
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
        :param d0: Julian date of minimum foliar moisture content (if None, it will be calculated based on latitude)
        :param dj: Julian date of modelled fire (if None, it will be calculated from wx_date)
        :param out_request: Tuple or list of CFFBPS output variables
            # Default output variables
            fire_type = Type of fire predicted to occur (surface, intermittent crown, active crown)
            hros = Head fire rate of spread (m/min)
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

            # Backing fire rate of spread variables
            bfW = backing fire wind speed component (km/h)
            brsi = backing fire spread rate without BUI effect (m/min)
            bisi = backing fire ISI without BUI effect
            bros = backing fire rate of spread (m/min)

            # Critical crown fire threshold variables
            csfi = critical intensity (kW/m)
            rso = critical rate of spread (m/min)

            # Crown fuel parameters
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
            to the grid codes used in this module
        :param percentile_growth: The ROS percentile growth (0-100) for the fire growth model
        """
        # Initialize CFFBPS input parameters
        self.fuel_type = fuel_type
        self.wx_date = wx_date  # For FMC calculations
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
        self.d0 = d0  # For FMC calculations
        self.dj = dj  # For FMC calculations
        self.out_request = out_request
        self.convert_fuel_type_codes = convert_fuel_type_codes
        self.percentile_growth = percentile_growth

        # Verify input parameters
        self._checkArray()
        self._verifyInputs()

        # ### Bind result attributes to the zero-filled template.
        # runFBP rebinds every one of these to a fresh array returned by its
        # equation, and the equations never mutate their inputs (enforced
        # own-and-return convention), so a single shared template preserves the
        # historical pre-run values (zeros via getParams) without ~40 full-size
        # array copies per initialize() — a real cost on large rasters.
        # NOTE: do not mutate these arrays in place before runFBP; assign whole
        # attributes instead (as setParams and all internal code do).
        template = self.ref_array

        # Weather parameters
        self.isi = template
        self.m = template
        self.fF = template
        self.fW = template

        # Slope effect parameters
        self.a = template
        self.b = template
        self.c = template
        self.rsz = template
        self.isz = template
        self.sf = template
        self.rsf = template
        self.isf = template
        self.rsi = template
        self.wse1 = template
        self.wse2 = template
        self.wse = template
        self.wsx = template
        self.wsy = template
        self.wsv = template
        self.raz = template

        # BUI effect parameters
        self.q = template
        self.bui0 = template
        self.be = template
        self.be_max = template

        # Surface parameters (ffc/wfc/sfc default to NaN, matching calcSFC's
        # unknown-fuel semantics)
        self.ffc = np.full_like(template, np.nan, dtype=np.float64)
        self.wfc = np.full_like(template, np.nan, dtype=np.float64)
        self.sfc = np.full_like(template, np.nan, dtype=np.float64)

        # Foliar moisture content parameters
        self.latn = template
        self.nd = template
        self.fmc = template
        self.fme = template

        # Crown and total fuel consumed parameters
        self.cbh = template
        self.csfi = template
        self.rso = template
        self.cfb = template
        self.bros_cfb = template
        self.cfl = template
        self.cfc = template
        self.tfc = template

        # Backing fire rate of spread parameters
        self.bfW = template
        self.brsi = template
        self.bisi = template
        self.bros = template

        # Default CFFBPS output parameters
        self.fire_type = template
        self.hros = template
        self.hfi = template

        # C-6 rate of spread parameters
        self.sros = template
        self.cros = template

        # Point ignition acceleration parameter
        self.accel_param = template

        # Fire intensity class parameter
        self.fi_class = self.ref_int_array

        # List of required parameters
        required_params = [
            'fuel_type', 'wx_date', 'lat', 'long', 'elevation', 'slope', 'aspect', 'ws', 'wd', 'ffmc', 'bui'
        ]

        # Check for missing required parameters
        missing_params = [param for param in required_params if getattr(self, param) is None]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        # Set initialized to True
        self.initialized = True

        return

    def invertWindAspect(self):
        """
        Function to invert/flip wind direction and aspect by 180 degrees
        :return: None
        """
        self.wd, self.aspect = sw_eq.invert_wind_aspect(self.wd, self.aspect)
        return

    def calcSF(self) -> None:
        """
        Function to calculate the slope factor
        :return: None
        """
        self.sf = sw_eq.calc_sf(self.slope)
        return

    def calcISZ(self) -> None:
        """
        Function to calculate the initial spread index with no wind/no slope effects
        :return: None
        """
        self.m, self.fF, self.isz = sw_eq.calc_isz(self.ffmc)
        return

    def calcFMC(self,
                d0: int | None = None,
                dj: int | None = None,
                lat: float | None = None,
                long: float | None = None,
                elevation: float | None = None,
                wx_date: int | None = None) -> None:
        """
        Function to calculate foliar moisture content (FMC) and foliar moisture effect (FME).
        :return: None
        """
        if lat is not None:
            self.lat = lat
        if long is not None:
            self.long = long
        if elevation is not None:
            self.elevation = elevation
        if wx_date is not None:
            self.wx_date = wx_date

        self.latn, self.d0, self.dj, self.nd, self.fmc, self.fme = fmc_eq.calc_fmc(
            lat=self.lat, long=self.long, elevation=self.elevation, wx_date=self.wx_date,
            d0=self.d0, dj=self.dj, d0_override=d0, dj_override=dj,
        )
        return

    def calcISI_RSI_BE(self) -> None:
        """
        Function to calculate the slope-/wind-adjusted Initial Spread Index (ISI),
        rate of spread (RSI), and the BUI buildup effect (BE) using NumPy masked arrays.

        :return: None
        """
        result = sw_eq.calc_isi_rsi_be(
            fuel_type=self.fuel_type, ros_params=self.rosParams, gcf=self.gcf,
            isz=self.isz, sf=self.sf, pc=self.pc, pdf=self.pdf, bui=self.bui,
            fF=self.fF, wd=self.wd, aspect=self.aspect, ws=self.ws,
            ref_array=self.ref_array,
        )
        # ISIRSIBEResult field names match the facade attribute names by contract.
        for key, value in result._asdict().items():
            setattr(self, key, value)
        return

    def calcROS(self) -> None:
        """
        Function to model the fire rate of spread (m/min).
        For C6, this is the surface fire heading and backing rate of spread.
        For all other fuel types, this is the overall heading and backing fire rate of spread.

        :return: None
        """
        self.hros, self.bros, self.sros = ros_eq.calc_ros(
            rsi=self.rsi, brsi=self.brsi, be=self.be,
            fuel_type=self.fuel_type, bui=self.bui, sros=self.sros,
        )
        return

    def calcSFC(self) -> None:
        """
        Function to calculate forest floor consumption (FFC), woody fuel consumption (WFC),
        and total surface fuel consumption (SFC) for all fuel types.
        :return: None
        """
        self.ffc, self.wfc, self.sfc = surface_eq.calc_sfc(
            fuel_type=self.fuel_type, ffmc=self.ffmc, bui=self.bui,
            pc=self.pc, gfl=self.gfl, ref_array=self.ref_array,
        )
        return

    def getCBH_CFL(self, ftype: int | None = None, cbh: float | None = None,
                   cfl: float | None = None) -> None:
        """
        Function to get the default CFFBPS canopy base height (CBH) and canopy fuel load (CFL)
        values for a specified fuel type.

        :param ftype: The numeric FBP fuel type code.
        :param cbh: A specific cbh value to use instead of the default (only for C6 fuel types)
        :param cfl: A specific cfl value to use instead of the default (only for C6 fuel types)
        :return: None
        """
        self.cbh, self.cfl = crown_eq.calc_cbh_cfl(
            fuel_type=self.fuel_type, cbh=self.cbh, cfl=self.cfl,
            cbh_cfl_ht_lut=self.fbpCBH_CFL_HT_LUT,
            ftype=ftype, cbh_override=cbh, cfl_override=cfl,
        )
        return

    def calcCSFI(self) -> None:
        """
        Function to calculate the critical surface fire intensity (CSFI).

        :return: None
        """
        self.csfi = crown_eq.calc_csfi(fuel_type=self.fuel_type, cbh=self.cbh, fmc=self.fmc)
        return

    def calcRSO(self) -> None:
        """
        Function to calculate the critical surface fire rate of spread (RSO).

        :return: None
        """
        self.rso = crown_eq.calc_rso(sfc=self.sfc, csfi=self.csfi)
        return

    def calcCFB(self) -> None:
        """
        Function calculates crown fraction burned using equation in Forestry Canada Fire Danger Group (1992).

        Also computes a backing-fire-specific CFB (self.bros_cfb), using bros in
        place of hros, for calcRosPercentileGrowth's backing-fire regime decision
        (matches WISE FBPFuel::BROS computing its own CFB from brss, distinct
        from FBPFuel::ROS's head-fire CFB from rss). For C6, sros (the C6-specific
        surface ROS used in place of hros for CFB) is head-fire-derived only —
        no backing-fire equivalent is computed elsewhere in this pipeline, so
        C6's backing CFB reuses the same sros as a documented simplification.

        :return: None
        """
        self.cfb = crown_eq.calc_cfb(
            fuel_type=self.fuel_type, ftypes=self.ftypes,
            non_crowning_fuels=self.non_crowning_fuels,
            sros=self.sros, rso=self.rso, hros=self.hros,
        )
        self.bros_cfb = crown_eq.calc_cfb(
            fuel_type=self.fuel_type, ftypes=self.ftypes,
            non_crowning_fuels=self.non_crowning_fuels,
            sros=self.sros, rso=self.rso, hros=self.bros,
        )
        return

    def calcRosPercentileGrowth(self) -> None:
        """
        Calculates the rate of spread (ROS) percentile growth for head fire and backing fire rates of spread.

        :return: None
        """
        self.hros, self.bros = growth_eq.calc_ros_percentile_growth(
            percentile_growth=self.percentile_growth, fuel_type=self.fuel_type,
            hros_cfb=self.cfb, bros_cfb=self.bros_cfb, wsv=self.wsv,
            hros=self.hros, bros=self.bros,
        )
        return

    def calcAccelParam(self) -> None:
        """
        Function to calculate acceleration parameter for a fire starting from a point ignition source.

        :return: None
        """
        self.accel_param = growth_eq.calc_accel_param(
            fuel_type=self.fuel_type, ftypes=self.ftypes,
            open_fuel_types=self.open_fuel_types,
            cfb=self.cfb, accel_param=self.accel_param,
        )
        return

    def calcFireType(self) -> None:
        """
        Function to calculate fire type (1: surface, 2: intermittent crown, 3: active crown)

        :return: None
        """
        self.fire_type = crown_eq.calc_fire_type(fuel_type=self.fuel_type, cfb=self.cfb)
        return

    def calcCFC(self) -> None:
        """
        Function calculates crown fuel consumed (kg/m^2).

        :return: None
        """
        self.cfc = crown_eq.calc_cfc(
            fuel_type=self.fuel_type, cfb=self.cfb, cfl=self.cfl, pc=self.pc, pdf=self.pdf,
        )
        return

    def calcC6hros(self) -> None:
        """
        Function to calculate crown and total head fire rate of spread for the C6 fuel type

        :returns: None
        """
        self.cros, self.hros = crown_eq.calc_c6hros(
            fuel_type=self.fuel_type, cfc=self.cfc, isi=self.isi, fme=self.fme,
            cros=self.cros, sros=self.sros, cfb=self.cfb, hros=self.hros,
        )
        return

    def calcTFC(self) -> None:
        """
        Function to calculate total fuel consumed (kg/m^2)

        :return: None
        """
        self.tfc = consumption_eq.calc_tfc(sfc=self.sfc, cfc=self.cfc)
        return

    def calcHFI(self) -> None:
        """
        Function to calculate fire type, total fuel consumption, and head fire intensity

        :returns: None
        """
        self.hfi = consumption_eq.calc_hfi(hros=self.hros, tfc=self.tfc)
        return

    def calcFireIntensityClass(self) -> None:
        """
        Function to calculate the fire intensity class based on fire intensity (FI).

        :return: None
        """
        self.fi_class = consumption_eq.calc_fire_intensity_class(hfi=self.hfi)
        return

    def setParams(self, set_dict: dict) -> None:
        """
        Function to set FBP parameters to specific values.

        :param set_dict: Dictionary of FBP parameter names and the values to assign to the FBP class object
        :return: None
        """
        # Iterate through the set dictionary and assign values
        for key, value in set_dict.items():
            if hasattr(self, key):  # Check if the class has the attribute
                if isinstance(value, np.ndarray):
                    setattr(self, key, mask.array(value, mask=np.isnan(value)))
                else:
                    setattr(self, key, mask.array([value], mask=np.isnan([value])))
        return

    def getParams(self, out_request: list[str]) -> list:
        """
        Function to output requested dataset parameters from the FBP class.

        :param out_request: List of requested FBP parameters.
        :return: List of requested outputs.
        """
        # Dictionary of CFFBPS parameters
        fbp_params = {
            # Default output variables
            'fire_type': self.fire_type,  # Type of fire (surface, intermittent crown, active crown)
            'hros': self.hros,  # Head fire rate of spread (m/min)
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
            'dj': self.dj,  # Julian date of day being modelled
            'd0': self.d0,  # Julian date of minimum foliar moisture content
            'nd': self.nd,  # number of days between modelled fire date and d0
            'fmc': self.fmc,  # foliar moisture content
            'fme': self.fme,  # foliar moisture effect

            # Critical crown fire threshold variables
            'csfi': self.csfi,  # Critical intensity (kW/m)
            'rso': self.rso,  # Critical rate of spread (m/min)

            # Backing fire spread variables
            'bfw': self.bfW,  # The backing fire wind function
            'bisi': self.bisi,  # The ISI associated with the backing fire rate of spread
            'bros': self.bros,  # Backing rate of spread (m/min)

            # C-6 specific variables
            'sros': self.sros,  # Surface fire rate of spread (m/min)
            'cros': self.cros,  # Crown fire rate of spread (m/min)

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

        def _to_plain(arr):
            # Guard to fill returned arrays with NaN at masked cells so downstream nanmax/isfinite guards
            # treat them as missing instead of as real values.
            if isinstance(arr, np.ma.MaskedArray) and np.issubdtype(arr.dtype, np.floating):
                return arr.filled(np.nan)
            return arr.data

        # Retrieve requested parameters
        if self.return_array:
            return [
                _to_plain(fbp_params.get(var))[0] if fbp_params.get(var, None) is not None
                                                     and fbp_params.get(var).ndim > 3
                else _to_plain(fbp_params.get(var)) if fbp_params.get(var, None) is not None
                else np.nan
                for var in out_request
            ]
        else:
            return [
                fbp_params.get(var).item() if fbp_params.get(var, None) is not None
                                              and fbp_params.get(var).ndim == 0
                else (fbp_params.get(var))[0].item() if fbp_params.get(var, None) is not None
                else np.nan
                for var in out_request
            ]

    def runFBP(self, block: np.ndarray | None = None) -> list:
        """
        Function to automatically run CFFBPS modelling.

        :param block: The array of partial data (block) to run FBP with.
        :returns:
            Tuple of values requested through out_request parameter. Default values are fire_type, hros, and hfi.
        :raises ValueError: if out_request contains any name not in cffbps.constants.valid_outputs.
        """
        if not self.initialized:
            raise ValueError('FBP class must be initialized before running calculations. Call "initialize" first.')

        if block is not None:
            self.block = block

        # Check output requests values
        if self.out_request is None:
            # Set default output requests if none provided
            self.out_request = ['hros', 'hfi', 'fire_type']
        else:
            unknown = [var for var in self.out_request if var not in constants.valid_outputs]
            if unknown:
                raise ValueError(
                    f'Unknown out_request value(s): {unknown}. '
                    f'Valid values are: {sorted(constants.valid_outputs)}'
                )

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
        self.calcC6hros()
        # Calculate total fuel consumption
        self.calcTFC()
        # Calculate head fire intensity
        self.calcHFI()
        # Calculate fire intensity class
        self.calcFireIntensityClass()

        # Return requested values
        return self.getParams(self.out_request)
