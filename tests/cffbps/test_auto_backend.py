"""Automatic backend selection and the public no-switching contract."""
import importlib
import subprocess
import sys
import warnings

import numpy as np
import pytest
from _wotton_helpers import load_input_frame, row_to_kwargs
from test_rust_backend import ALL, BASE, equal, pyrocell

from cffdrs.cffbps import FBP
from cffdrs.cffbps.auto import AutoFBP, available_backends
from cffdrs.cffbps.python import FBP as PythonFBP


def make(cls=FBP, **kwargs):
    model = cls()
    model.initialize(**(BASE | {'out_request': ALL} | kwargs))
    return model


def test_public_dispatching_class_and_read_only_status():
    assert FBP is AutoFBP
    model = FBP()
    assert model.backend_in_use is None
    with pytest.raises(AttributeError):
        model.backend_in_use = 'rust'
    model.initialize(**BASE)
    assert model.backend_in_use == 'rust'
    with pytest.raises(AttributeError):
        model.backend_in_use = 'python'


def test_wotton_all_54_outputs_choose_rust():
    rows = load_input_frame()
    assert len(rows) == 20
    for _, row in rows.iterrows():
        kwargs = row_to_kwargs(row, ALL)
        model = make(**kwargs)
        assert model.backend_in_use == 'rust'
        equal(model.runFBP(), make(PythonFBP, **kwargs).runFBP(), scalar=True)


def test_missing_extension_falls_back_silently(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cffdrs._rust', None)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model = make()
        equal(model.runFBP(), make(PythonFBP).runFBP(), scalar=True)
    assert model.backend_in_use == 'python'
    assert not captured


def test_forced_python_never_imports_extension():
    code = """
import sys
class BlockRust:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cffdrs._rust' or fullname.startswith('cffdrs.cffbps.rust'):
            raise AssertionError('explicit Python tried Rust')
sys.meta_path.insert(0, BlockRust())
from cffdrs.cffbps import FBP
model = FBP()
model.initialize(fuel_type=2, wx_date=20230615, lat=52., long=-115.,
                 elevation=800., slope=10., aspect=180., ws=20., wd=270.,
                 ffmc=90., bui=80., backend='python', threads=-99)
assert model.backend_in_use == 'python'
assert model.runFBP()[0] > 0
assert 'cffdrs._rust' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', code], check=True)


def test_forced_rust_requires_extension(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cffdrs._rust', None)
    with pytest.raises(ImportError, match='pip install|uv sync'):
        make(backend='rust')


TABLES = ['rosParams', 'open_fuel_types', 'non_crowning_fuels', 'fbpCBH_CFL_HT_LUT',
          'csfiVarList', 'rsoVarList', 'cfbVarList', 'cfcVarList', 'cfiVarList', 'ftypes']


@pytest.mark.parametrize('name', TABLES)
@pytest.mark.parametrize('route', ['read', 'assign'])
def test_preinitialize_tables_pin_python(name, route):
    model = FBP()
    if route == 'read':
        getattr(model, name)
    else:
        setattr(model, name, getattr(PythonFBP(), name))
    model.initialize(**BASE, out_request=ALL)
    assert model.backend_in_use == 'python'
    equal(model.runFBP(), make(PythonFBP).runFBP(), scalar=True)


def test_calibrated_table_is_preserved_and_forced_rust_rejected():
    model, reference = FBP(), PythonFBP()
    for item in (model, reference):
        original = item.rosParams[2]
        item.rosParams[2] = (original[0] * 1.5, *original[1:])
    model.initialize(**BASE, out_request=ALL)
    reference.initialize(**BASE, out_request=ALL)
    assert model.backend_in_use == 'python'
    actual = model.runFBP()
    equal(actual, reference.runFBP(), scalar=True)
    assert actual[ALL.index('hros')] != make(PythonFBP).runFBP()[ALL.index('hros')]
    with pytest.raises(ValueError, match="backend='python'"):
        model.initialize(**BASE, backend='rust')
    pinned = FBP()
    pinned.rosParams[2] = (150., *pinned.rosParams[2][1:])
    with pytest.raises(ValueError, match="backend='python'"):
        pinned.initialize(**BASE, backend='rust')


@pytest.mark.parametrize('name', TABLES)
@pytest.mark.parametrize('route', ['read', 'assign'])
def test_rust_table_operations_fail_without_switching(name, route):
    model = make()
    with pytest.raises((AttributeError, NotImplementedError), match="backend='python'"):
        if route == 'read':
            getattr(model, name)
        else:
            setattr(model, name, {})
    assert model.backend_in_use == 'rust'
    equal(model.runFBP(), make(PythonFBP).runFBP(), scalar=True)


@pytest.mark.parametrize('operation', ['hros', 'block', 'cbh'])
def test_rust_unsupported_operations_fail_without_switching(operation):
    model = make()
    with pytest.raises((AttributeError, NotImplementedError), match="backend='python'"):
        if operation == 'hros':
            _ = model.hros
        elif operation == 'block':
            model.runFBP(block=(0, 1))
        else:
            model.getCBH_CFL(cbh=3.)
    assert model.backend_in_use == 'rust'
    equal(model.runFBP(), make(PythonFBP).runFBP(), scalar=True)


@pytest.mark.parametrize('backend', ['python', 'rust'])
def test_reinitialization_keeps_backend(backend, monkeypatch):
    model = make(backend=backend)
    if backend == 'python':
        # An extension becoming available must not migrate a Python instance.
        monkeypatch.delitem(sys.modules, 'cffdrs._rust', raising=False)
    model.initialize(**(BASE | {'fuel_type': 6, 'out_request': ALL}))
    assert model.backend_in_use == backend
    equal(model.runFBP(), make(PythonFBP, fuel_type=6).runFBP(), scalar=True)
    other = 'rust' if backend == 'python' else 'python'
    with pytest.raises(ValueError, match='chosen|switch|once'):
        model.initialize(**BASE, backend=other)
    assert model.backend_in_use == backend


@pytest.mark.parametrize('backend', ['auto', 'python'])
@pytest.mark.parametrize('percentile', [5, 50, 90])
@pytest.mark.parametrize('growth', [False, True])
def test_pyrocell_injections(backend, percentile, growth):
    kwargs = dict(fuel_type=np.array([2, 6, 10, 14]), percentile_growth=percentile)
    model, reference = make(backend=backend, **kwargs), make(PythonFBP, **kwargs)
    pyrocell(model, growth)
    pyrocell(reference, growth)
    if backend == 'auto':
        equal(model.getParams(ALL), reference.getParams(ALL))
    else:
        # Python leaves skipped FMC fields as scalar NaN; forwarding preserves it.
        for name, actual, expected in zip(ALL, model.getParams(ALL), reference.getParams(ALL), strict=True):
            assert type(actual) is type(expected), name
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=name)


@pytest.mark.parametrize('name,value', [('slope', -10), ('aspect', -1), ('ws', -5), ('ffmc', -5),
                                      ('bui', -1), ('pc', -1), ('pdf', -1), ('gfl', -1), ('gcf', 0),
                                      ('lat', 55), ('long', -100), ('elevation', 100), ('wd', 30),
                                      ('wx_date', 20240615), ('percentile_growth', 90), ('fuel_type', 6)])
@pytest.mark.parametrize('route', ['setParams', 'attribute', 'set'])
def test_updates_forward_to_chosen_backend(name, value, route):
    model = make()
    if route == 'setParams':
        model.setParams({name: value})
    elif route == 'attribute':
        setattr(model, name, value)
    else:
        model.set(name, value)
    equal(model.runFBP(), make(PythonFBP, **{name: value}).runFBP(), scalar=True)


@pytest.mark.parametrize('name', ['d0', 'dj'])
@pytest.mark.parametrize('route', ['setParams', 'attribute', 'set_override', 'calcFMC'])
def test_d16_override_routes(name, route):
    model, reference = make(fuel_type=np.full(3, 2)), make(PythonFBP, fuel_type=np.full(3, 2))
    value = np.ma.array([166., 167., 168.], mask=[False, True, False])
    for item in (model, reference):
        if route == 'setParams':
            item.setParams({name: value})
        elif route == 'attribute':
            setattr(item, name, value)
        elif route == 'set_override':
            item.set_override(name, value)
        else:
            item.calcFMC(**{name: value})
    equal(model.runFBP(), reference.runFBP())


def test_threads_reach_rust():
    model = make(threads=2)
    assert model.backend_in_use == 'rust'
    equal(model.runFBP(), make(PythonFBP).runFBP(), scalar=True)
    with pytest.raises((ValueError, OverflowError)):
        make(threads=-1)


@pytest.mark.parametrize('backend', ['cupy', '', None, 3])
def test_invalid_backend(backend):
    with pytest.raises(ValueError, match='backend'):
        make(backend=backend)


def test_available_backends_reports_importable_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', None)
    assert available_backends() == ['python', 'rust']
    from cffdrs.cffbps import available_backends as public_available
    assert public_available is available_backends
    monkeypatch.setitem(sys.modules, 'cffdrs._rust', None)
    assert available_backends() == ['python']
    monkeypatch.setitem(sys.modules, 'cupy', importlib)
    assert available_backends() == ['python', 'cupy']


def test_public_fbp_import_is_lazy():
    code = """
import sys
from cffdrs.cffbps import FBP
assert 'cffdrs._rust' not in sys.modules
assert 'cffdrs.cffbps.rust' not in sys.modules
assert 'cupy' not in sys.modules
model = FBP()
assert model.backend_in_use is None
assert 'cffdrs._rust' not in sys.modules
assert 'cupy' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', code], check=True)
