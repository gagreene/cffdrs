"""The backend comparison tool must really apply the missing-fuel mask in every input shape."""
import importlib.util
from pathlib import Path

import numpy as np

TOOL = Path(__file__).resolve().parents[1] / 'tools' / 'compare_backends.py'
_spec = importlib.util.spec_from_file_location('compare_backends', TOOL)
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)


# ---- missing fuel tests ----

def test_every_missing_fuel_scenario_has_a_masked_or_invalid_fuel():
    seen = {'scalar': 0, 'array': 0, 'mixed': 0}
    for label, kw in compare.scenarios():
        if 'missing=fuel' not in label:
            continue
        shape = label.split()[0]
        fuel = kw['fuel_type']
        masked = np.ma.getmaskarray(fuel).any() if isinstance(fuel, np.ma.MaskedArray) else fuel == -1
        assert masked, label
        seen[shape] += 1
    assert all(seen.values()), seen


def test_a_mixed_missing_fuel_scenario_differs_from_its_unmasked_baseline():
    kws = {label: kw for label, kw in compare.scenarios() if label.startswith('mixed fuel=1 ') and 'p=50' in label}
    masked = kws['mixed fuel=1 missing=fuel p=50']['fuel_type']
    baseline = kws['mixed fuel=1 missing=None p=50']['fuel_type']
    assert np.ma.getmaskarray(masked).tolist() == [[True, False]]
    assert not np.ma.getmaskarray(baseline).any()
