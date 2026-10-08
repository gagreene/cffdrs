"""Boundary-inclusive timing of the compiled grid pass (``cffdrs._rust.run_fbp_grid``).

Includes input copies and output array allocation, which the scalar
``bench_run`` example cannot see. Usage: ``uv run --no-sync python tools/bench_rust_grid.py``
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from cffdrs import _rust as rust_backend

SHAPE = (1000, 1000)
WARMUP = 3
REPEATS = 7
SCALARS = dict(wx_date=20230615, ffmc=91.2, bui=76.4, pdf=42.0, gfl=0.41)
PERCENTILE = 50.0


def build_grids():
    rng = np.random.default_rng(42)
    n = SHAPE[0] * SHAPE[1]
    fuel = np.tile(np.arange(1, 19, dtype=np.int32), n // 18 + 1)[:n].reshape(SHAPE)
    return dict(
        fuel_type=np.ascontiguousarray(fuel),
        lat=rng.uniform(48, 60, SHAPE),
        long=rng.uniform(-120, -95, SHAPE),
        elevation=rng.uniform(0, 1500, SHAPE),
        slope=rng.uniform(0, 60, SHAPE),
        aspect=rng.uniform(0, 360, SHAPE),
        pc=rng.uniform(0, 100, SHAPE),
        gcf=rng.uniform(0, 100, SHAPE),
        ws=rng.uniform(0, 45, SHAPE),
        wd=rng.uniform(0, 360, SHAPE),
    )


def run_once(g):
    return rust_backend.run_fbp_grid(
        g['fuel_type'],
        g['lat'], g['long'], g['elevation'],
        g['slope'], g['aspect'],
        g['pc'], g['gcf'],
        g['ws'], g['wd'],
        SCALARS['wx_date'], SCALARS['ffmc'], SCALARS['bui'],
        SCALARS['pdf'], SCALARS['gfl'], PERCENTILE,
    )


def main():
    g = build_grids()
    for _ in range(WARMUP):
        run_once(g)
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        run_once(g)
        times.append(time.perf_counter() - t0)
    cells = SHAPE[0] * SHAPE[1]
    mn, med = min(times), statistics.median(times)
    print(
        f'min {mn * 1e3:.1f} ms ({mn * 1e9 / cells:.1f} ns/cell), '
        f'median {med * 1e3:.1f} ms ({med * 1e9 / cells:.1f} ns/cell)'
    )


if __name__ == '__main__':
    main()
