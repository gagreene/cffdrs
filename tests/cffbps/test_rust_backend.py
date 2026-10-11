"""Public Rust FBP behavior against the explicit Python reference."""
import importlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from _wotton_helpers import load_input_frame, row_to_kwargs
from test_rust_fbpgrid import STAGES
from test_rust_parity import SCALARS, grids  # noqa: F401

from cffdrs.cffbps.python import FBP as PythonFBP
from cffdrs.cffbps.python import valid_outputs

ALL = list(valid_outputs)
BASE = dict(fuel_type=2, wx_date=20230615, lat=52., long=-115., elevation=800.,
            slope=10., aspect=180., ws=20., wd=270., ffmc=90., bui=80.)


def make(backend='rust', **kw):
    cls = importlib.import_module('cffdrs.cffbps.rust').FBP if backend == 'rust' else PythonFBP
    model = cls()
    model.initialize(**(BASE | {'out_request': ALL} | kw))
    return model


def equal(actual, expected, scalar=False):
    for name, left, right in zip(ALL, actual, expected, strict=True):
        np.testing.assert_allclose(left, np.broadcast_to(right, np.shape(left)),
                                   rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=name)
        if scalar:
            assert type(left) is type(right), (name, type(left), type(right))
        else:
            assert type(left) is np.ndarray, name
            assert left.dtype == np.asarray(right).dtype, name


def test_wotton_all_outputs():
    for _, row in load_input_frame().iterrows():
        kw = row_to_kwargs(row, ALL)
        equal(make(**kw).runFBP(), make('python', **kw).runFBP(), scalar=True)


@pytest.mark.parametrize('code', [*range(1, 21), 0, 99, -1])
def test_scalar_fuels(code):
    equal(make(fuel_type=code).runFBP(), make('python', fuel_type=code).runFBP(), scalar=True)


def test_shared_grid(grids):  # noqa: F811
    kw = grids | SCALARS
    equal(make(**kw).runFBP(), make('python', **kw).runFBP())


@pytest.mark.parametrize('shape', [(7,), (2, 3)])
@pytest.mark.parametrize('layout', ['c', 'fortran', 'negative'])
def test_mixed_shape_layout_ownership(shape, layout):
    lat = np.linspace(50, 55, np.prod(shape), dtype=np.float32).reshape(shape)
    if layout == 'fortran':
        lat = np.asfortranarray(lat)
    elif layout == 'negative':
        lat = np.ascontiguousarray(lat[..., ::-1])[..., ::-1]
    model = make(lat=lat)
    reference = make('python', lat=lat.copy(), fuel_type=np.full(shape, 2)).runFBP()
    actual = model.runFBP()
    for name, a, b in zip(ALL, actual, reference, strict=True):
        assert a.shape == shape, name
        np.testing.assert_allclose(a, np.broadcast_to(b, shape), rtol=1e-9, equal_nan=True, err_msg=name)
    lat[...] = 80
    equal(model.runFBP(), actual)


@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64, np.uint16, np.float32, np.float64])
def test_fuel_dtypes(dtype):
    values = [2, 6, 19, 20, 0] if dtype == np.int8 else [2, 6, 19, 20, 258]
    fuel = np.ma.array(np.array(values, dtype=dtype), mask=[False, True, False, False, False])
    equal(make(fuel_type=fuel).runFBP(), make('python', fuel_type=fuel).runFBP())


def test_fractional_fuel():
    fuel = np.array([2., 2.9, 258.])
    equal(make(fuel_type=fuel).runFBP(), make('python', fuel_type=fuel).runFBP())


@pytest.mark.parametrize('name',
                         ['fuel_type', 'lat', 'long', 'elevation', 'slope', 'aspect', 'ws', 'wd', 'ffmc', 'bui'])
def test_masked_integer_snapshot(name):
    snapshot = json.loads((Path(__file__).parent / 'data/golden/integer_outputs_snapshot.json').read_text())
    kw = {'fuel_type': np.full((1, 3), 2, dtype=np.int8)}
    kw[name] = np.ma.array(np.full((1, 3), BASE[name]), mask=[[False, True, False]])
    model = make(**kw)
    model.runFBP()
    for field, record in snapshot['masked_' + name].items():
        arr = model.getParams([field])[0]
        assert arr[0, 1] == record['value'], (name, field)
        assert str(arr.dtype) == record['dtype']


def test_stepwise_and_lifecycle():
    model = make()
    for stage in STAGES:
        getattr(model, stage)()
    equal(model.getParams(ALL), make().runFBP(), scalar=True)
    equal(model.runFBP(), make().runFBP(), scalar=True)
    equal(model.runFBP(), make().runFBP(), scalar=True)
    with pytest.raises(RuntimeError, match='reset'):
        model.calcFMC()
    model.reset()
    model.calcFMC()
    assert model.getParams(['fmc'])[0] > 0


def pyrocell(model, growth=True):
    for stage in ['invertWindAspect', 'calcSF', 'calcISZ']:
        getattr(model, stage)()
    model.setParams({'fmc': np.full(4, 95.)})
    for stage in ['calcISI_RSI_BE', 'calcSFC', 'getCBH_CFL', 'calcROS']:
        getattr(model, stage)()
    model.setParams({'hros': np.array([np.nan, 4., 9., 14.])})
    for stage in STAGES[8:]:
        if growth or stage != 'calcRosPercentileGrowth':
            getattr(model, stage)()


@pytest.mark.parametrize('percentile', [5, 50, 90])
@pytest.mark.parametrize('growth', [False, True])
@pytest.mark.parametrize('default', [False, True])
def test_pyrocell(percentile, growth, default):
    kw = dict(fuel_type=np.array([2, 6, 10, 14]), percentile_growth=percentile,
              out_request=None if default else ALL)
    a, b = make(**kw), make('python', **kw)
    pyrocell(a, growth)
    pyrocell(b, growth)
    if default:
        for x, y in zip(a.getParams(['hros', 'hfi', 'fire_type']),
                        b.getParams(['hros', 'hfi', 'fire_type']), strict=True):
            np.testing.assert_allclose(x, y, rtol=1e-9, equal_nan=True)
    else:
        equal(a.getParams(ALL), b.getParams(ALL))


@pytest.mark.parametrize('name,value', [('slope', -10), ('aspect', -1), ('ws', -5), ('ffmc', -5),
                                      ('bui', -1), ('pc', -1), ('pdf', -1), ('gfl', -1), ('gcf', 0),
                                      ('lat', 55), ('long', -100), ('elevation', 100), ('wd', 30),
                                      ('wx_date', 20240615), ('percentile_growth', 90), ('fuel_type', 6)])
@pytest.mark.parametrize('route', ['setParams', 'attribute', 'set'])
def test_updates(name, value, route):
    model = make()
    if route == 'setParams':
        model.setParams({name: value, 'bogus': 3})
    elif route == 'attribute':
        setattr(model, name, value)
    else:
        model.set(name, value)
    equal(model.runFBP(), make('python', **{name: value}).runFBP(), scalar=True)
    model.reset()
    equal(model.runFBP(), make('python', **{name: value}).runFBP(), scalar=True)


@pytest.mark.parametrize('name', ['d0', 'dj'])
@pytest.mark.parametrize('route', ['initialize', 'calcFMC', 'setParams', 'attribute', 'set_override'])
def test_override_routes(name, route):
    values = np.ma.array([0., 166.9, 400.], mask=[False, False, True])
    kw = dict(fuel_type=np.full(3, 2))
    a, b = make(**kw), make('python', **kw)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        if route == 'initialize':
            a, b = make(**kw, **{name: values}), make('python', **kw, **{name: values})
        else:
            for model in [a, b]:
                if route == 'calcFMC':
                    model.calcFMC(**{name: values})
                elif route == 'setParams':
                    model.setParams({name: values})
                elif route == 'attribute':
                    setattr(model, name, values)
                else:
                    model.set_override(name, values)
        equal(a.runFBP(), b.runFBP())
    a.set_override(name, None)
    equal(a.runFBP(), make('python', **kw).runFBP())


@pytest.mark.parametrize('value', [0, -5, 366, 400, 365, 166.9, np.nan])
@pytest.mark.parametrize('year', [2023, 2024])
def test_dj_warning_and_values(value, year):
    kw = dict(dj=value, wx_date=year * 10000 + 615)
    with warnings.catch_warnings(record=True) as reference:
        warnings.simplefilter('always')
        expected = make('python', **kw).runFBP()
    with warnings.catch_warnings(record=True) as native:
        warnings.simplefilter('always')
        model = make(**kw)
        actual = model.runFBP()
        model.set_override('dj', value)
        model.runFBP()
    assert [str(w.message) for w in native] == [str(w.message) for w in reference]
    equal(actual, expected, scalar=True)


@pytest.mark.parametrize('name', ['d0', 'dj'])
def test_scalar_missing_fmc(name):
    for backend in ['python', 'rust']:
        model = make(backend, **{name: np.nan})
        model.runFBP()
        assert np.isnan(model.getParams(['fmc'])[0])
        assert np.isnan(model.getParams(['fme'])[0])
        assert np.isnan(model.getParams([name])[0])


@pytest.mark.parametrize('kw', [dict(lat=83, long=-52, elevation=5959),
                               dict(lat=0, long=-110, elevation=0), dict(d0=400), dict(d0=0)])
def test_d0_unrestricted(kw):
    equal(make(**kw).runFBP(), make('python', **kw).runFBP(), scalar=True)


def test_restrictions_and_freed():
    model = make(out_request=None)
    with pytest.raises(NotImplementedError, match='threads'):
        model.runFBP(block=np.ones(1))
    with pytest.raises(NotImplementedError, match='setParams'):
        model.getCBH_CFL(cbh=2)
    with pytest.raises(AttributeError, match="backend='python'"):
        _ = model.hros
    model.runFBP()
    with pytest.raises(ValueError, match='out_request'):
        model.getParams(['sf'])
    assert np.isnan(model.getParams(['bogus'])[0])


def test_missing_extension(monkeypatch):
    importlib.import_module('cffdrs.cffbps.rust')
    monkeypatch.setitem(sys.modules, 'cffdrs._rust', None)
    monkeypatch.delitem(sys.modules, 'cffdrs.cffbps.rust')
    monkeypatch.delitem(sys.modules, 'cffdrs.cffbps.rust.core')
    with pytest.raises(ImportError, match='pip|uv'):
        importlib.import_module('cffdrs.cffbps.rust')


def test_threads():
    fuel = np.tile(np.arange(1, 19, dtype=np.int32), 300)
    equal(make(fuel_type=fuel, threads=1).runFBP(), make(fuel_type=fuel, threads=4).runFBP())


def test_fmc_input_update_and_injected_reset():
    model = make()
    model.calcFMC(lat=57, long=-101, elevation=300, wx_date=20240615)
    reference = make('python', lat=57, long=-101, elevation=300, wx_date=20240615)
    reference.calcFMC()
    for a, b in zip(model.getParams(['fmc', 'fme', 'd0', 'dj']),
                    reference.getParams(['fmc', 'fme', 'd0', 'dj']), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-9)
    model.setParams({'fmc': 9, 'hros': np.nan, 'cbh': 90})
    equal(model.runFBP(), reference.runFBP(), scalar=True)
    model.initialize(**BASE, out_request=ALL)
    equal(model.runFBP(), make('python').runFBP(), scalar=True)


def test_aliases_use_reference_attribute_names():
    a, b = make(), make('python')
    for model in (a, b):
        model.setParams({'bfW': 12., 'accel_param': 0.3, 'bfw': 99., 'accel': 99.})
    assert a.getParams(['bfw', 'accel']) == b.getParams(['bfw', 'accel']) == [12., 0.3]


def test_warning_remembers_value_year_pairs():
    model = make(dj=365)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        model.set_override('dj', 400)
        model.set_param('wx_date', 20240615)
        model.set_param('wx_date', 20230615)
        model.set_override('dj', 365)
        model.set_override('dj', 400)
    assert [str(w.message) for w in seen] == [
        'dj override: 1 value(s) outside 1..365 (year 2023) were clipped',
        'dj override: 1 value(s) outside 1..366 (year 2024) were clipped']


def test_override_infinity_and_fractional_floor():
    model = make()
    for route in [model.set_override, lambda name, value: model.setParams({name: value})]:
        with pytest.raises(ValueError, match='infinity'):
            route('dj', np.inf)
    for value in [165.5, 166.5, 166.9]:
        model.set_override('dj', value)
        model.calcFMC()
        assert model.getParams(['dj'])[0] == int(np.floor(value))


def test_uncomputed_dates_and_nonmodeled_integer_snapshot():
    model = make(fuel_type=np.full(3, 2))
    assert all(np.isnan(arr).all() for arr in model.getParams(['d0', 'dj']))
    snapshot = json.loads((Path(__file__).parent / 'data/golden/integer_outputs_snapshot.json').read_text())
    for code in [0, 19, 20, 99, -1]:
        for scalar in [False, True]:
            model = make(fuel_type=code if scalar else np.full((1, 3), code))
            model.runFBP()
            for name, record in snapshot[f'fuel_{code}_' + ('scalar' if scalar else 'array')].items():
                value = model.getParams([name])[0]
                assert (value if scalar else value[0, 1]) == record['value']
                assert (type(value).__name__ if scalar else str(value.dtype)) == record['dtype']

@pytest.mark.parametrize('name,value', [('hros', 7), ('fmc', 90), ('cbh', 3), ('bfW', 9), ('accel_param', 1)])
def test_injected_result_types_and_stage_replacement(name, value):
    a, b = make(), make('python')
    for model in (a, b):
        model.setParams({name: value})
    output = {'bfW': 'bfw', 'accel_param': 'accel'}.get(name, name)
    x, y = a.getParams([output])[0], b.getParams([output])[0]
    assert type(x) is type(y)
    assert x == y
    equal(a.runFBP(), b.runFBP(), scalar=True)


def test_masked_float_exports_all_fields():
    for name in ['fuel_type', 'lat', 'long', 'elevation', 'slope', 'aspect', 'ws', 'wd', 'ffmc', 'bui']:
        kw = {'fuel_type': np.full((1, 3), 2, dtype=np.int8)}
        kw[name] = np.ma.array(np.full((1, 3), BASE[name]), mask=[[False, True, False]])
        equal(make(**kw).runFBP(), make('python', **kw).runFBP())

@pytest.mark.parametrize('code', range(1, 21))
def test_masked_fuel_acceleration(code):
    fuel = np.ma.array([code, code, code], mask=[False, True, False])
    a, b = make(fuel_type=fuel), make('python', fuel_type=fuel)
    equal(a.runFBP(), b.runFBP())
    a.reset()
    b.reset()
    for stage in STAGES:
        getattr(a, stage)()
        getattr(b, stage)()
    equal(a.getParams(ALL), b.getParams(ALL))


@pytest.mark.parametrize('shape', [(1,), (1, 1), (0,)])
def test_single_cell_and_empty_shapes(shape):
    model = make(fuel_type=np.full(shape, 2, dtype=np.int8))
    for result in model.runFBP():
        assert result.shape == shape


def test_threads_are_bit_identical():
    fuel = np.tile(np.arange(1, 19, dtype=np.int32), 300)
    for a, b in zip(make(fuel_type=fuel, threads=1).runFBP(),
                    make(fuel_type=fuel, threads=4).runFBP(), strict=True):
        np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))


def test_acceleration_injection_overrides_export_mask():
    fuel = np.ma.array([2, 2, 2], mask=[False, True, False])
    a, b = make(fuel_type=fuel), make('python', fuel_type=fuel)
    for model in (a, b):
        for stage in STAGES:
            getattr(model, stage)()
        model.setParams({'accel_param': np.full(3, 0.7)})
    np.testing.assert_array_equal(a.getParams(['accel'])[0], b.getParams(['accel'])[0])

@pytest.mark.parametrize('cfb', [None, 0.2, np.nan])
def test_masked_closed_acceleration_reads_current_cfb(cfb):
    fuel = np.ma.array([[2, 2]], mask=[[True, False]], dtype=np.int8)
    a, b = make(fuel_type=fuel, out_request=['accel', 'cfb', 'tfc']), make('python', fuel_type=fuel)
    for model in (a, b):
        if cfb is not None:
            model.setParams({'cfb': np.full((1, 2), cfb)})
        model.calcAccelParam()
    np.testing.assert_allclose(a.getParams(['accel'])[0], b.getParams(['accel'])[0], rtol=1e-9, equal_nan=True)


@pytest.mark.parametrize('code', [19, 20, 0])
def test_nonmodeled_staged_injected_consumption(code):
    fuel = np.full((1, 2), code, dtype=np.int8)
    a, b = make(fuel_type=fuel), make('python', fuel_type=fuel)
    for model in (a, b):
        model.setParams({'sfc': np.full((1, 2), 2.), 'cfc': np.full((1, 2), 3.),
                         'hros': np.full((1, 2), 4.)})
        model.calcTFC()
        model.calcHFI()
        model.calcFireIntensityClass()
    for x, y in zip(a.getParams(['tfc', 'hfi', 'fi_class']), b.getParams(['tfc', 'hfi', 'fi_class']), strict=True):
        np.testing.assert_array_equal(x, y)

@pytest.mark.parametrize('code', [19, 20, 0])
def test_scalar_nonmodeled_injected_consumption_class(code):
    a, b = make(fuel_type=code), make('python', fuel_type=code)
    for model in (a, b):
        model.setParams({'sfc': 2., 'cfc': 3., 'hros': 4.})
        model.calcTFC()
        model.calcHFI()
        model.calcFireIntensityClass()
    for x, y in zip(a.getParams(['tfc', 'hfi', 'fi_class']), b.getParams(['tfc', 'hfi', 'fi_class']), strict=True):
        assert x == y
        assert type(x) is type(y)
    for model in (a, b):
        model.setParams({'hfi': 6000.})
        model.calcFireIntensityClass()
    assert a.getParams(['fi_class']) == b.getParams(['fi_class']) == [5]
    assert type(a.getParams(['fi_class'])[0]) is int


def test_masked_acceleration_default_retention():
    model = make(fuel_type=np.ma.array([[2, 2]], mask=[[True, False]]), out_request=None)
    model.calcAccelParam()
    with pytest.raises(ValueError, match='out_request'):
        model.getParams(['accel'])


@pytest.mark.parametrize('code', [19, 20, 0])
def test_nonmodeled_acceleration_keeps_injection(code):
    a, b = make(fuel_type=np.full(2, code)), make('python', fuel_type=np.full(2, code))
    for model in (a, b):
        model.setParams({'accel_param': np.full(2, 0.7)})
        model.calcAccelParam()
    np.testing.assert_array_equal(a.getParams(['accel'])[0], b.getParams(['accel'])[0])


@pytest.mark.parametrize('code', [2, 19, 20, 0])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, 6000., 0.])
def test_scalar_classification_uses_consumed_hfi_mask(code, value):
    a, b = make(fuel_type=code, out_request=['fi_class', 'hfi']), make('python', fuel_type=code)
    for model in (a, b):
        model.setParams({'hfi': value})
        model.calcFireIntensityClass()
    x, y = a.getParams(['fi_class'])[0], b.getParams(['fi_class'])[0]
    assert x == y
    assert type(x) is type(y)


@pytest.mark.parametrize('initialized', [False, True])
@pytest.mark.parametrize('name', [
    'rosParams', 'open_fuel_types', 'non_crowning_fuels', 'fbpCBH_CFL_HT_LUT',
    'fbpFTCode_AlphaToNum_LUT', 'fbpFTCode_NumToAlpha_LUT',
    'csfiVarList', 'rsoVarList', 'cfbVarList', 'cfcVarList', 'cfiVarList', 'ftypes',
])
def test_rust_rejects_calibration_assignment(name, initialized):
    model = make() if initialized else importlib.import_module('cffdrs.cffbps.rust').FBP()
    with pytest.raises(NotImplementedError, match="backend='python'"):
        setattr(model, name, {})


@pytest.mark.parametrize('operation', ['block', 'canopy'])
def test_rust_unsupported_operations_advise_python(operation):
    model = make()
    with pytest.raises(NotImplementedError, match="backend='python'"):
        if operation == 'block':
            model.runFBP(block=1)
        else:
            model.getCBH_CFL(cbh=3.)
