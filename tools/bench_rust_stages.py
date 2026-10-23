"""Release-extension stage timing and owned-memory measurements (not a test).

Run with the configured Python SDK: python tools/bench_rust_stages.py.
The scalar gate is measured separately with the release bench_run Rust example.
RSS is sampled in separate memory passes, so sampling does not affect timing.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import threading
import time
from contextlib import contextmanager

import psutil

from bench_rust_grid import PERCENTILE, SCALARS, build_grids, run_once
from cffdrs import _rust
from cffdrs.cffbps.common.constants import valid_outputs
from cffdrs.cffbps.python import FBP as PythonFBP

STAGES = (
    'invertWindAspect', 'calcSF', 'calcISZ', 'calcFMC', 'calcISI_RSI_BE',
    'calcROS', 'calcSFC', 'getCBH_CFL', 'calcCSFI', 'calcRSO',
    'calcC6BlendCFB', 'calcC6BlendCFC', 'calcC6CROS', 'calcC6HROS',
    'calcPercentileCFB', 'calcRosPercentileGrowth', 'calcCFB',
    'calcAccelParam', 'calcFireType', 'calcCFC', 'calcTFC', 'calcHFI',
    'calcFireIntensityClass',
)
REQUEST = ['hros', 'hfi', 'fire_type']
FMC_OUTPUTS = ['latn', 'd0', 'dj', 'nd', 'fmc', 'fme']


def construct(inputs, outputs=REQUEST, threads=1):
    return _rust.FbpGrid(
        **inputs, **SCALARS, percentile_growth=PERCENTILE,
        out_request=list(outputs), threads=threads,
    )


def distribution(samples):
    milliseconds = [sample * 1000 for sample in samples]
    return dict(samples_ms=milliseconds, min_ms=min(milliseconds),
                median_ms=statistics.median(milliseconds), max_ms=max(milliseconds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error('warmup must be nonnegative and repeats must be positive')
    inputs = build_grids()
    cells = inputs['fuel_type'].size
    result = dict(
        metadata=dict(cells=cells, shape=list(inputs['fuel_type'].shape),
                      warmup=args.warmup, repeats=args.repeats,
                      python=platform.python_version(), platform=platform.platform(),
                      extension=_rust.__file__, logical_cpus=os.cpu_count(),
                      physical_cpus=psutil.cpu_count(logical=False),
                      rss_sampling_interval_ms=5),
        fused={}, stepwise={}, memory={},
    )
    for threads in (1, 4, 0):
        grid = construct(inputs, threads=threads)
        result['fused'][str(threads)] = measure(grid.run, args.warmup, args.repeats)
        del grid
    serial = result['fused']['1']
    for threads in (4, 0):
        timing = result['fused'][str(threads)]
        timing['speedup_min'] = serial['min_ms'] / timing['min_ms']
        timing['speedup_median'] = serial['median_ms'] / timing['median_ms']
    for label, outputs in (('three_outputs', REQUEST), ('all_54_outputs', valid_outputs)):
        grid = construct(inputs, outputs)

        def stages(model=grid):
            for name in STAGES:
                model.run_stage(name)

        result['stepwise'][label] = measure(stages, args.warmup, args.repeats, grid.reset)
        del stages, grid
        result['memory'][label] = dict(
            stepwise=memory_pass(inputs, outputs, False),
            fused=memory_pass(inputs, outputs, True),
        )
    # Keep all six outputs: freeing one FMC output deliberately invalidates its cache.
    grid = construct(inputs, FMC_OUTPUTS)
    date = SCALARS['wx_date']

    def invalidate(model=grid):
        nonlocal date
        date = 20230616 if date == 20230615 else 20230615
        model.set_param('wx_date', date)

    result['fmc'] = dict(
        miss=measure(lambda model=grid: model.run_stage('calcFMC'), args.warmup, args.repeats, invalidate),
        hit=measure(lambda model=grid: model.run_stage('calcFMC'), args.warmup, args.repeats),
        retained_outputs=FMC_OUTPUTS,
        python_task4_recorded_ms=dict(hit_approx=85, miss_range=[414, 450]),
    )
    del invalidate, grid
    python_grid = PythonFBP()
    python_grid.initialize(**inputs, **SCALARS, percentile_growth=PERCENTILE,
                           out_request=list(valid_outputs))

    def invalidate_python(model=python_grid):
        model.wx_date = 20230616 if model.wx_date == 20230615 else 20230615

    result['fmc']['python_same_grid'] = dict(
        miss=measure(python_grid.calcFMC, args.warmup, args.repeats, invalidate_python),
        hit=measure(python_grid.calcFMC, args.warmup, args.repeats),
    )
    del invalidate_python, python_grid
    result['construction'] = {}
    for count in (90_000, 1_000):
        window = {name: values.reshape(-1)[:count].copy() for name, values in inputs.items()}
        result['construction'][str(count)] = measure(
            lambda values=window: construct(values), args.warmup, args.repeats,
        )
    # Original ten-field function includes construction and returned NumPy allocation.
    def legacy_boundary():
        run_once(inputs)

    result['legacy_ten_field_boundary'] = measure(
        legacy_boundary, args.warmup, args.repeats,
    )
    baseline = 1490.2
    timing = result['legacy_ten_field_boundary']
    result['grid_gate'] = dict(baseline_min_ms=baseline, limit_ms=baseline * 1.10,
                               current_min_ms=timing['min_ms'],
                               regression_percent=(timing['min_ms'] / baseline - 1) * 100,
                               passed=timing['min_ms'] <= baseline * 1.10)
    print(json.dumps(result, indent=2), flush=True)


def measure(operation, warmup, repeats, prepare=lambda: None):
    for _ in range(warmup):
        prepare()
        operation()
    samples = []
    for _ in range(repeats):
        prepare()
        start = time.perf_counter()
        returned = operation()
        samples.append(time.perf_counter() - start)
        del returned
    return distribution(samples)


def memory_pass(inputs, outputs, fused):
    gc.collect()
    with rss_peak() as rss:
        grid = construct(inputs, outputs)
        live = [dict(stage='construction', live_bytes=grid.live_bytes())]
        if fused:
            grid.run()
            live.append(dict(stage='run', live_bytes=grid.live_bytes()))
        else:
            for name in STAGES:
                grid.run_stage(name)
                live.append(dict(stage=name, live_bytes=grid.live_bytes()))
        result = dict(peak_live_bytes=max(item['live_bytes'] for item in live),
                      final_live_bytes=live[-1]['live_bytes'], after_stages=live)
        del grid
    result['rss'] = rss
    return result


@contextmanager
def rss_peak():
    """Sample whole-process RSS every 5 ms; this is not native live_bytes."""
    process = psutil.Process()
    result = dict(baseline_bytes=process.memory_info().rss, peak_bytes=0)
    result['peak_bytes'] = result['baseline_bytes']
    stopped = threading.Event()

    def sample():
        while not stopped.wait(0.005):
            result['peak_bytes'] = max(result['peak_bytes'], process.memory_info().rss)

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    try:
        yield result
    finally:
        result['peak_bytes'] = max(result['peak_bytes'], process.memory_info().rss)
        stopped.set()
        worker.join()


if __name__ == '__main__':
    main()



