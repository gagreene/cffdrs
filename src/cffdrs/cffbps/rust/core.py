"""Owned Rust FBP facade.

Inputs are copied. Modify them through assignments or the explicit setters; editing
an original input buffer in place does not update the model. Computed results are
available through getParams. After runFBP, call reset before using individual stages.
"""
import importlib

import numpy as np

from ..common import inputs
from ..common.constants import open_fuel_types, valid_outputs
from ..common.fmc_inputs import coerce_override, resolve_dj_override
from ._convert import finish, to_nan
from ._scalar_masks import ScalarMasks

try:
    _rust = importlib.import_module('cffdrs._rust')
except ImportError as error:
    raise ImportError('Rust FBP requires the cffdrs extension: install with pip install cffdrs '
                      'or build the checkout with uv sync --extra test --extra dev') from error

_INPUTS = ('lat', 'long', 'elevation', 'slope', 'aspect', 'ws', 'wd', 'ffmc', 'bui',
           'pc', 'pdf', 'gfl', 'gcf')
_ALIASES = {'bfW': 'bfw', 'accel_param': 'accel'}
_INTERNAL = {'isz', 'brsi', 'c6_blend_cfb', 'c6_blend_cfc', 'percentile_cfb',
             'percentile_bros_cfb', 'bros_cfb'}
# setParams uses actual Python attribute names, whereas getParams uses output names.
_SETTABLE = (set(valid_outputs) - {'bfw', 'accel'}) | set(_ALIASES) | _INTERNAL | set(_INPUTS)
_UNSUPPORTED = {'rosParams', 'open_fuel_types', 'non_crowning_fuels', 'fbpCBH_CFL_HT_LUT',
                'fbpFTCode_AlphaToNum_LUT', 'fbpFTCode_NumToAlpha_LUT', 'fbpMultiprocessArray',
                'csfiVarList', 'rsoVarList', 'cfbVarList', 'cfcVarList', 'cfiVarList', 'ftypes'}
_STAGE_WRITES = {
    'invertWindAspect': 'wd aspect', 'calcSF': 'sf', 'calcISZ': 'm fF isz',
    'calcFMC': 'latn d0 dj nd fmc fme',
    'calcISI_RSI_BE': 'a b c q bui0 be_max rsz rsf isf rsi brsi be wse1 wse2 wse wsx wsy wsv raz fW bfw isi bisi',
    'calcROS': 'hros bros sros', 'calcSFC': 'ffc wfc sfc', 'getCBH_CFL': 'cbh cfl',
    'calcCSFI': 'csfi', 'calcRSO': 'rso', 'calcC6BlendCFB': 'c6_blend_cfb',
    'calcC6BlendCFC': 'c6_blend_cfc', 'calcC6CROS': 'cros', 'calcC6HROS': 'hros',
    'calcPercentileCFB': 'percentile_cfb percentile_bros_cfb',
    'calcRosPercentileGrowth': 'hros bros', 'calcCFB': 'cfb bros_cfb',
    'calcAccelParam': 'accel', 'calcFireType': 'fire_type', 'calcCFC': 'cfc',
    'calcTFC': 'tfc', 'calcHFI': 'hfi', 'calcFireIntensityClass': 'fi_class',
}
_STAGE_WRITES = {name: fields.split() for name, fields in _STAGE_WRITES.items()}
_DEFAULT = ['hros', 'hfi', 'fire_type']


def _percentile(value):
    """The native percentile: ``None`` means no adjustment, as in the Python reference (the median, 50)."""
    return 50 if value is None else value


class FBP:
    """FBP with Rust calculations and explicit owned-input update routes."""

    def __init__(self):
        object.__setattr__(self, '_grid', None)
        object.__setattr__(self, 'initialized', False)

    def __getattr__(self, name):
        if name in _INPUTS and self._grid is not None:
            return self._grid.get(name)
        if name == 'fuel_type' and self._grid is not None:
            return self._fuel.copy()
        if name in ('d0', 'dj') and self._grid is not None:
            if not self._fmc_done:
                supplied = self._overrides[name]
                return None if supplied is None else supplied.copy()
            return self.getParams([name])[0]
        if name in _UNSUPPORTED or name in _SETTABLE or name in valid_outputs:
            raise AttributeError(f'{name} is unavailable on Rust FBP; use getParams([{name!r}]) '
                                 "or backend='python' for result attributes and mutable tables")
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in _UNSUPPORTED:
            raise NotImplementedError(f"{name} is unsupported on Rust FBP; use backend='python'")
        if not name.startswith('_') and self._grid is not None and (
                name in _SETTABLE or name in ('wx_date', 'percentile_growth')):
            self.set(name, value)
        else:
            object.__setattr__(self, name, value)

    def initialize(self, fuel_type=None, wx_date=None, lat=None, long=None, elevation=None,
                   slope=None, aspect=None, ws=None, wd=None, ffmc=None, bui=None,
                   pc=50, pdf=35, gfl=0.35, gcf=80, d0=None, dj=None, out_request=None,
                   convert_fuel_type_codes=False, percentile_growth=50, threads=1):
        prepared = inputs.prepare_inputs(
            fuel_type=fuel_type, wx_date=wx_date, lat=lat, long=long, elevation=elevation,
            slope=slope, aspect=aspect, ws=ws, wd=wd, ffmc=ffmc, bui=bui, pc=pc, pdf=pdf,
            gfl=gfl, gcf=gcf, d0=d0, dj=dj, out_request=out_request,
            convert_fuel_type_codes=convert_fuel_type_codes)
        verified = prepared.verified
        request = _DEFAULT.copy() if out_request is None else list(out_request)
        unknown = [name for name in request if name not in valid_outputs]
        if unknown:
            raise ValueError(f'Invalid out_request values: {unknown}. Valid values are: {sorted(valid_outputs)}')
        shape = prepared.ref_array.shape if prepared.return_array else ()
        values = {name: self._boundary(getattr(verified, name), shape) for name in _INPUTS}
        overrides = {name: getattr(verified, name) for name in ('d0', 'dj')}
        grid = _rust.FbpGrid(
            fuel_type=self._fuel_boundary(verified.fuel_type, shape), **values, wx_date=wx_date,
            percentile_growth=_percentile(percentile_growth),
            d0=self._boundary(overrides['d0'], shape), dj=self._boundary(overrides['dj'], shape),
            out_request=request, threads=threads)
        object.__setattr__(self, '_grid', grid)
        object.__setattr__(self, 'wx_date', wx_date)
        object.__setattr__(self, 'percentile_growth', percentile_growth)
        self.out_request = request
        self.convert_fuel_type_codes = convert_fuel_type_codes
        self.return_array = prepared.return_array
        self._fuel = verified.fuel_type.copy()
        self._inputs = {name: getattr(verified, name).copy() for name in _INPUTS}
        self._overrides = overrides
        self._dj_warned = set()
        self._fmc_done = self._sfc_done = False
        self._injected = {}
        self._accel_done = False
        self._accel_fused = False
        self._reset_masks()
        self.initialized = True
        self._warn_dj()

    @staticmethod
    def _boundary(value, shape):
        arr = to_nan(value)
        if arr is None:
            return None
        if np.size(arr) == 1 and (not shape or np.shape(arr) != shape):
            return float(np.asarray(arr).item())
        if np.shape(arr) != shape:
            raise ValueError(f'input shape {np.shape(arr)} != grid shape {shape}')
        return arr

    @staticmethod
    def _fuel_boundary(fuel, shape):
        arr = np.asarray(fuel.filled(0), dtype=np.int32)
        if arr.size == 1 and (not shape or arr.shape != shape):
            return int(arr.item())
        if arr.shape != shape:
            raise ValueError(f'fuel_type shape {arr.shape} != grid shape {shape}')
        return arr

    def _require(self):
        if self._grid is None:
            raise ValueError('FBP class must be initialized before running calculations. Call "initialize" first.')

    def _warn_dj(self):
        supplied = self._overrides['dj']
        if supplied is not None and self._grid.dj_override_clipped:
            resolve_dj_override(supplied, self.wx_date, self._dj_warned)

    def set_override(self, name, values):
        self._require()
        if name not in ('d0', 'dj'):
            raise ValueError("set_override name must be 'd0' or 'dj'")
        supplied = coerce_override(name, values)
        self._grid.set_override(name, self._boundary(supplied, self._grid.shape))
        self._overrides[name] = supplied
        if self._scalar_masks is not None:
            self._scalar_masks.set(name, supplied)
        self._fmc_done = False
        self._warn_dj()

    def set_param(self, name, value):
        self._require()
        if name == 'wx_date':
            value = inputs.validate_wx_date(value)
        self._grid.set_param(name, _percentile(value) if name == 'percentile_growth' else value)
        object.__setattr__(self, name, value)
        if name == 'wx_date':
            self._fmc_done = False
            self._warn_dj()

    def set_fuel(self, value):
        self._require()
        fuel, _ = inputs.prepare_fuel_type(value, self.convert_fuel_type_codes)
        self._grid.set_fuel(self._fuel_boundary(fuel, self._grid.shape))
        self._fuel = fuel.copy()
        if self._scalar_masks is not None:
            self._scalar_masks.set("fuel_type", fuel)

    def set(self, name, value):
        self._require()
        if name in ('d0', 'dj'):
            self.set_override(name, value)
        elif name in ('wx_date', 'percentile_growth'):
            self.set_param(name, value)
        elif name == 'fuel_type':
            self.set_fuel(value)
        else:
            arr = (inputs.normalize_input(name, value) if name in inputs.NORMALISED_INPUTS
                   else inputs.coerce_input(name, value))
            if name == 'long':
                arr = np.absolute(arr)
            self._grid.set(_ALIASES.get(name, name), self._boundary(arr, self._grid.shape))
            if self._scalar_masks is not None:
                self._scalar_masks.set(_ALIASES.get(name, name), arr)
            if name in _INPUTS:
                self._inputs[name] = arr.copy()
            else:
                self._injected[_ALIASES.get(name, name)] = arr.copy()

    def setParams(self, set_dict):
        for name, value in set_dict.items():
            if name in _SETTABLE or name in ('wx_date', 'percentile_growth'):
                self.set(name, value)

    def _reset_masks(self):
        self._scalar_masks = (None if self.return_array else
                              ScalarMasks(self._inputs, self._fuel, self._overrides))

    def reset(self):
        self._require()
        self._grid.reset()
        self._reset_masks()
        self._fmc_done = self._sfc_done = False
        self._injected.clear()
        self._accel_done = False
        self._accel_fused = False

    def runFBP(self, block=None):
        self._require()
        if block is not None:
            raise NotImplementedError('block processing is a Python-backend feature; the Rust backend uses threads; '
                                      "use backend='python' for block processing")
        self.reset()
        self._grid.run()
        if self._scalar_masks is not None:
            for stage in _STAGE_WRITES:
                self._scalar_masks.stage(stage, self._fuel, _percentile(self.percentile_growth))
        self._fmc_done = self._sfc_done = self._accel_done = True
        self._accel_fused = True
        self._accel_fuel = self._fuel
        return self.getParams(self.out_request)

    def _stage(self, name):
        self._require()
        if self._accel_fused:
            self._grid.run_stage(name)  # raises the native AfterFused error
        if name == 'calcAccelParam' and np.ma.getmaskarray(self._fuel).any():
            codes = np.broadcast_to(self._fuel.data, self._grid.shape or (1,))
            # Python computes closed acceleration only for valid models present
            # in the grid, while open fuel membership includes masked codes.
            selected = np.isin(codes, open_fuel_types) | np.isin(codes, self._fuel.compressed())
            codes = np.asarray(np.where(selected, codes, 0), dtype=np.int32)
            if not self._grid.shape:
                codes = int(codes.item())
            self._grid.run_acceleration_for_fuel(codes)
        else:
            self._grid.run_stage(name)
        if self._scalar_masks is not None:
            self._scalar_masks.stage(name, self._fuel, _percentile(self.percentile_growth))
        for field in _STAGE_WRITES[name]:
            self._injected.pop(field, None)
        if name == 'calcAccelParam':
            self._accel_fused = False
            self._accel_done = True
            self._accel_fuel = self._fuel
        if name == 'calcFMC':
            self._fmc_done = True
        elif name == 'calcSFC':
            self._sfc_done = True

    def calcFMC(self, d0=None, dj=None, lat=None, long=None, elevation=None, wx_date=None):
        for name, value in [('d0', d0), ('dj', dj), ('lat', lat), ('long', long),
                            ('elevation', elevation), ('wx_date', wx_date)]:
            if value is not None:
                self.set(name, value)
        self._stage('calcFMC')

    def getCBH_CFL(self, ftype=None, cbh=None, cfl=None):
        if any(value is not None for value in (ftype, cbh, cfl)):
            raise NotImplementedError('Rust getCBH_CFL overrides require setParams({"cbh": ..., "cfl": ...}); '
                                      "use backend='python' for lookup overrides")
        self._stage('getCBH_CFL')

    def getParams(self, out_request):
        self._require()
        results = []
        for name in out_request:
            if name not in valid_outputs:
                results.append(np.nan)
                continue
            # Even fuel_type must surface native lifecycle errors before exporting metadata.
            result = self._grid.get(name)
            dtype = None
            if name in ('d0', 'dj') and not self._fmc_done:
                supplied = self._overrides[name]
                result = (np.full(self._grid.shape or (1,), np.nan) if supplied is None
                          else np.broadcast_to(to_nan(supplied), self._grid.shape or (1,)))
                if supplied is not None and not np.ma.getmaskarray(supplied).any():
                    dtype = supplied.dtype
            if name in ('ws', 'ffmc', 'bui', 'wd'):
                dtype = self._inputs[name].dtype
            elif name == 'nd' and self._overrides['d0'] is not None:
                supplied = self._overrides['d0']
                dj_supplied = self._overrides['dj']
                if not np.ma.getmaskarray(supplied).any() and (
                        dj_supplied is None or not np.ma.getmaskarray(dj_supplied).any()):
                    dtype = np.result_type(supplied.dtype, np.int64)
            if name == 'fuel_type':
                result = np.broadcast_to(self._fuel.data, self._grid.shape or (1,))
                dtype = self._fuel.dtype
            elif name == 'd0' and self._overrides['d0'] is not None:
                supplied = self._overrides['d0']
                if not np.ma.getmaskarray(supplied).any():
                    dtype = supplied.dtype
            elif name == 'dj' and self._overrides['dj'] is None and self._fmc_done:
                result = np.nan_to_num(result, nan=0.)
            if name in self._injected:
                supplied = self._injected[name]
                dtype = supplied.dtype
                if not self.return_array and np.ma.getmaskarray(supplied).any():
                    result, dtype = np.array(0.), np.float64
            if name == 'accel' and self._accel_done and self._accel_fused and name not in self._injected:
                fuel_data = np.broadcast_to(self._accel_fuel.data, self._grid.shape or (1,))
                fuel_mask = np.broadcast_to(np.ma.getmaskarray(self._accel_fuel), np.shape(fuel_data))
                result = np.where(fuel_mask & np.isin(fuel_data, open_fuel_types), 0.115, result)
                result = np.where(fuel_mask & np.isin(fuel_data, self._accel_fuel.compressed())
                                  & np.isin(fuel_data, range(1, 19))
                                  & ~np.isin(fuel_data, open_fuel_types), np.nan, result)
            if self._scalar_masks is not None:
                result, dtype = self._scalar_masks.export(name, result, dtype)
            results.append(finish(name, result, not self.return_array, dtype=dtype))
        return results


def _stage_method(name):
    def method(self):
        self._stage(name)
    method.__name__ = name
    method.__doc__ = f'Run the {name} stage on the owned Rust grid.'
    return method


for _name in (
        'invertWindAspect calcSF calcISZ calcISI_RSI_BE calcROS calcSFC calcCSFI calcRSO '
        'calcC6BlendCFB calcC6BlendCFC calcC6CROS calcC6HROS calcPercentileCFB '
        'calcRosPercentileGrowth calcCFB calcAccelParam calcFireType calcCFC calcTFC '
        'calcHFI calcFireIntensityClass').split():
    setattr(FBP, _name, _stage_method(_name))





