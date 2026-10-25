"""Differential comparison of the Python and Rust FBP backends.

Runs both backends over the same scenarios (every fuel code plus invalid codes, each input
missing in turn, two percentiles, scalar and array input, and a few partial-stage states) and
tabulates, per output name, where they differ: value (beyond rtol 1e-9, NaN equal to NaN), type
(Python type for scalar input, dtype for arrays) and shape. The table is the systematic list of
known differences kept in ``docs/BACKEND_DIFFERENCES.md``: run this tool again after any change
to either backend and compare.

    uv run --no-sync python tools/compare_backends.py [--markdown]
"""
import argparse
import warnings
from collections import defaultdict

import numpy as np

from cffdrs.cffbps import common
from cffdrs.cffbps.python import FBP as PythonFBP
from cffdrs.cffbps.rust import FBP as RustFBP

NAMES = sorted(common.constants.valid_outputs)
FUELS = [*range(1, 21), 0, 99]
INPUTS = ['lat', 'long', 'elevation', 'slope', 'aspect', 'ws', 'wd', 'ffmc', 'bui']
BASE = dict(wx_date=20230615, lat=55.0, long=-110.0, elevation=500.0, slope=10.0, aspect=180.0,
            ws=20.0, wd=0.0, ffmc=90.0, bui=80.0)


def compare(py, rs):
    """Return the set of difference kinds between one Python and one Rust output."""
    kinds = set()
    a, b = np.asarray(py), np.asarray(rs)
    if isinstance(py, np.ndarray) != isinstance(rs, np.ndarray):
        kinds.add('container')
    elif isinstance(py, np.ndarray):
        if a.dtype != b.dtype:
            kinds.add('dtype')
        if a.shape != b.shape:
            kinds.add('shape')
    elif type(py) is not type(rs):
        kinds.add('type')
    try:
        if a.shape == b.shape or a.size == 1 or b.size == 1:
            if not np.allclose(a.astype(float), b.astype(float), rtol=1e-9, atol=1e-12, equal_nan=True):
                kinds.add('value')
    except (TypeError, ValueError):
        kinds.add('value')
    return kinds


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--markdown', action='store_true', help='print a Markdown table')
    args = parser.parse_args()
    warnings.simplefilter('ignore')
    table = defaultdict(lambda: defaultdict(lambda: [0, None]))
    total = 0
    for label, kw in scenarios():
        for stage in ('init', 'run', 'calcFMC'):
            try:
                py = run(PythonFBP, dict(kw), stage)
                rs = run(RustFBP, dict(kw), stage)
            except Exception as exc:  # report, do not hide, a scenario one backend rejects
                table['(whole scenario)'][f'{type(exc).__name__}'][0] += 1
                table['(whole scenario)'][f'{type(exc).__name__}'][1] = f'{label} {stage}: {exc}'[:90]
                continue
            total += 1
            for name, p, r in zip(NAMES, py, rs, strict=True):
                for kind in compare(p, r):
                    cell = table[name][kind]
                    cell[0] += 1
                    cell[1] = cell[1] or f'{label} {stage}'
    print(f'{total} scenario runs compared ({len(NAMES)} outputs each)')
    if args.markdown:
        print('\n| Output | Kind | Cases | First example |\n|---|---|---|---|')
    for name in sorted(table):
        for kind, (count, example) in sorted(table[name].items()):
            if args.markdown:
                print(f'| {name} | {kind} | {count} | {example} |')
            else:
                print(f'{name:12s} {kind:10s} {count:6d}  {example}')


def run(cls, kw, stage):
    model = cls()
    model.initialize(**kw)
    if stage == 'init':
        return model.getParams(NAMES)
    if stage == 'run':
        return model.runFBP()
    for step in ('invertWindAspect', 'calcSF', 'calcISZ', 'calcFMC'):
        getattr(model, step)()
        if step == stage:
            break
    return model.getParams(NAMES)


def scenarios():
    """Yield (label, kwargs) for scalar and array input."""
    for shape in ('scalar', 'array', 'mixed'):
        scalar = shape == 'scalar'
        for fuel in FUELS:
            for missing in [None, *INPUTS, 'fuel']:
                for percentile in (50, 90):
                    kw = dict(BASE, percentile_growth=percentile, out_request=NAMES)
                    kw['fuel_type'] = fuel
                    if missing in INPUTS:
                        kw[missing] = np.nan
                    if shape == 'mixed':
                        kw['fuel_type'] = np.array([[fuel, fuel]])
                        if missing == 'fuel':
                            kw['fuel_type'] = np.ma.masked_array(kw['fuel_type'], mask=[[True, False]])
                        yield f'{shape} fuel={fuel} missing={missing} p={percentile}', kw
                        continue
                    if not scalar:
                        kw = {k: (np.full((1, 2), v) if k in INPUTS else v) for k, v in kw.items()}
                        kw['fuel_type'] = np.array([[fuel, fuel]])
                        if missing in INPUTS:
                            kw[missing][0, 0] = np.nan
                    if missing == 'fuel':
                        kw['fuel_type'] = np.ma.masked_array(np.array([[fuel, fuel]]), mask=[[True, False]]) \
                            if not scalar else -1
                    yield f'{shape} fuel={fuel} missing={missing} p={percentile}', kw


if __name__ == '__main__':
    main()
