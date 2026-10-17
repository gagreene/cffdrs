"""Record reference integer exports; never rewrite existing golden fixtures."""
import json
from pathlib import Path

import numpy as np

from cffdrs.cffbps.python import FBP, valid_outputs

BASE = dict(fuel_type=2, wx_date=20230615, lat=52., long=-115., elevation=800.,
            slope=10., aspect=180., ws=20., wd=270., ffmc=90., bui=80.)
INTEGER = ['fire_type', 'fi_class', 'dj', 'fuel_type']


def main():
    records = {}
    for name in ['fuel_type', 'lat', 'long', 'elevation', 'slope', 'aspect', 'ws', 'wd', 'ffmc', 'bui']:
        kw = BASE | {'fuel_type': np.full((1, 3), 2, dtype=np.int8)}
        kw[name] = np.ma.array(np.full((1, 3), BASE[name]), mask=[[False, True, False]])
        model = FBP()
        model.initialize(**kw, out_request=INTEGER)
        records['masked_' + name] = {field: {'value': arr.ravel()[min(1, arr.size - 1)].item(), 'dtype': str(arr.dtype)}
                                    for field, arr in zip(INTEGER, model.runFBP(), strict=True)}
    for code in [0, 19, 20, 99, -1]:
        for scalar in [False, True]:
            model = FBP()
            model.initialize(**(BASE | {'fuel_type': code if scalar else np.full((1, 3), code)}),
                             out_request=INTEGER)
            records[f'fuel_{code}_' + ('scalar' if scalar else 'array')] = {
                field: {'value': arr if scalar else arr.ravel()[min(1, arr.size - 1)].item(),
                        'dtype': type(arr).__name__ if scalar else str(arr.dtype)}
                for field, arr in zip(INTEGER, model.runFBP(), strict=True)}
    path = Path(__file__).resolve().parents[1] / 'tests/cffbps/data/golden/integer_outputs_snapshot.json'
    if path.exists():
        raise FileExistsError('Refusing to overwrite the integer snapshot')
    path.write_text(json.dumps(records, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for code in range(0, 21):
        model = FBP()
        model.initialize(**(BASE | {'fuel_type': code}), out_request=list(valid_outputs))
        results = dict(zip(valid_outputs, model.runFBP(), strict=True))
        print(code, {key: value for key, value in results.items() if key in ('ffc', 'wfc', 'sfc', 'dj')})
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()

