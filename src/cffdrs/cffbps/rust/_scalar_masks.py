"""Scalar export provenance; numeric equations remain entirely in Rust.

Only scalar models need this metadata. Each producer snapshots masks from its
selected dependencies, so later input edits cannot reinterpret an older result.
The native getter still controls availability: metadata cannot resurrect storage.
"""
import numpy as np


def masked(value):
    return value is not None and bool(np.ma.getmaskarray(value).any())


class ScalarMasks:
    """Track missing-input provenance separately from unmasked NaN arithmetic."""

    def __init__(self, inputs, fuel, overrides):
        self.fields = {name: masked(value) for name, value in inputs.items()}
        self.fields['fuel_type'] = masked(fuel)
        self.date_nan = set()
        self.produced = set()
        self.overrides = overrides
        if masked(overrides['fmc']):
            # the fmc (and the fme it gives) is present before calcFMC runs
            self.set('fmc', overrides['fmc'])

    def export(self, name, result, dtype):
        producer = {'fire_type': 'calcFireType', 'fi_class': 'calcFireIntensityClass'}.get(name)
        if producer is not None and producer not in self.produced and dtype is None:
            dtype = np.float64
        if name == 'fuel_type' and self.fields[name]:
            return np.array(0.), np.float64
        if name in self.date_nan:
            return result, dtype
        if self.fields.get(name, False):
            # Native integer sentinels represent a masked produced value only
            # when this producer consumed a mask. Infinity remains unmasked.
            sentinel = 0 if name in ('fire_type', 'dj') else -99 if name == 'fi_class' else None
            if np.isnan(result).any() or (sentinel is not None and np.asarray(result).item() == sentinel):
                return np.array(0.), np.float64
        return result, dtype

    def set(self, name, value):
        self.fields[name] = masked(value)
        self.date_nan.discard(name)
        if name == 'fmc':
            # a given fmc also gives the matching fme
            self.fields['fme'] = self.fields[name]
            self.date_nan.discard('fme')
        if name in ('d0', 'dj', 'fmc') and self.fields[name]:
            # a missing override exports NaN straight away, as in the Python reference
            self.date_nan.add(name)
            if name == 'fmc':
                self.date_nan.add('fme')

    def stage(self, name, fuel, percentile):
        m = self.fields
        self.produced.add(name)
        code = int(fuel.data.item())
        invalid = masked(fuel)

        def any_mask(*fields):
            return any(m.get(field, False) for field in fields)

        def put(fields, value):
            for field in fields.split():
                m[field] = bool(value)
                self.date_nan.discard(field)

        if name == 'invertWindAspect':
            return
        if name == 'calcSF':
            put('sf', any_mask('slope'))
        elif name == 'calcISZ':
            put('m fF isz', any_mask('ffmc'))
        elif name == 'calcFMC':
            put('latn', any_mask('long', 'elevation'))
            d0, dj, fmc = (self.overrides[n] for n in ('d0', 'dj', 'fmc'))
            put('d0', masked(d0) if d0 is not None else any_mask('lat', 'latn', 'elevation'))
            put('dj', masked(dj) if dj is not None else any_mask('latn'))
            put('nd', any_mask('d0', 'dj'))
            put('fmc fme', masked(fmc) if fmc is not None else any_mask('nd'))
            if masked(d0) or masked(dj) or masked(fmc):
                self.date_nan.update(n for n in ('d0', 'dj', 'fmc', 'fme') if m[n])
        elif name == 'calcISI_RSI_BE':
            put('a b c q bui0 be_max', False)
            mixture = ('pc',) if code in (10, 11) else ('gcf',) if code in (14, 15) else ()
            put('rsz', invalid or any_mask('isz', *mixture))
            put('rsf', any_mask('rsz', 'sf'))
            put('isf', code not in range(1, 19) or any_mask('rsf', 'pdf' if code in (12, 13) else 'rsf'))
            put('wse1 wse2 wse', any_mask('isf', 'fF'))
            put('wsx wsy wsv raz fW bfw', any_mask('wse', 'ws', 'wd', 'aspect'))
            put('isi bisi', any_mask('fF', 'wsv'))
            mixture = ('pdf',) if code in (12, 13) else mixture
            put('rsi brsi', invalid or any_mask('isi', *mixture))
            put('be', code in (19, 20) or any_mask('bui'))
        elif name == 'calcROS':
            put('hros', invalid or any_mask('rsi', 'be') or (code == 9 and any_mask('bui')))
            put('bros', invalid or any_mask('brsi', 'be') or (code == 9 and any_mask('bui')))
            put('sros', invalid or any_mask('hros' if code == 6 else 'sros'))
        elif name == 'calcSFC':
            put('ffc', invalid or code not in (7, 16, 17, 18) or any_mask('ffmc' if code == 7 else 'bui'))
            put('wfc', invalid or code not in (7, 16, 17, 18) or any_mask('bui'))
            deps = ('ffmc',) if code == 1 else ('gfl',) if code in (14, 15) else ('bui',)
            if code == 7:
                deps = ('ffc', 'wfc')
            elif code in (10, 11):
                deps = ('bui', 'pc')
            put('sfc', invalid or code not in range(1, 19) or any_mask(*deps))
        elif name == 'getCBH_CFL':
            if not invalid:
                put('cbh cfl', False)
        elif name == 'calcCSFI':
            put('csfi', invalid or (code < 14 and any_mask('cbh', 'fmc')))
        elif name == 'calcRSO':
            put('rso', any_mask('sfc', 'csfi'))
        elif name == 'calcC6BlendCFB':
            put('c6_blend_cfb', invalid or (code == 6 and any_mask('sros', 'rso')))
        elif name in ('calcC6BlendCFC', 'calcCFC'):
            source = 'c6_blend_cfb' if name == 'calcC6BlendCFC' else 'cfb'
            target = 'c6_blend_cfc' if name == 'calcC6BlendCFC' else 'cfc'
            deps = ('pc',) if code in (10, 11) else ('pdf',) if code in (12, 13) else ()
            put(target, invalid or any_mask(source, 'cfl', *deps))
        elif name == 'calcC6CROS':
            deps = ('c6_blend_cfc', 'isi', 'fme') if code == 6 else ('cros',)
            put('cros', invalid or any_mask(*deps))
        elif name == 'calcC6HROS':
            deps = ('sros', 'cros', 'c6_blend_cfb') if code == 6 else ('hros',)
            put('hros', invalid or any_mask(*deps))
        elif name in ('calcPercentileCFB', 'calcCFB'):
            crowning = code in (*range(1, 8), *range(10, 14))
            targets = ('percentile_cfb', 'percentile_bros_cfb') if name == 'calcPercentileCFB' else ('cfb', 'bros_cfb')
            for target, source in zip(targets, ('hros', 'bros'), strict=True):
                put(target, invalid or (crowning and any_mask(source, 'rso')))
        elif name == 'calcRosPercentileGrowth':
            if percentile != 50 and code in (*range(1, 8), *range(10, 14)):
                put('hros', any_mask('hros', 'percentile_cfb'))
                put('bros', any_mask('bros', 'percentile_bros_cfb', 'wsv'))
        elif name == 'calcAccelParam':
            if code in (1, 7, 9, 14, 15, 16, 17, 18):
                put('accel', False)
            elif not invalid and code in range(1, 19):
                put('accel', any_mask('cfb'))
        elif name == 'calcFireType':
            put('fire_type', invalid or (code < 19 and any_mask('cfb')))
        elif name == 'calcTFC':
            # consumption follows the spread: a masked rate of spread masks it
            for target in ('ffc', 'wfc', 'sfc', 'cfc'):
                put(target, any_mask(target, 'hros'))
            put('tfc', any_mask('sfc', 'cfc'))
        elif name == 'calcHFI':
            put('hfi', any_mask('hros', 'tfc'))
        elif name == 'calcFireIntensityClass':
            put('fi_class', any_mask('hfi'))
