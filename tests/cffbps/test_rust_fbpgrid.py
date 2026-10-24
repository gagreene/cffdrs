"""Contracts for the owned, stateful Rust pipeline binding."""

import numpy as np
import pytest
from test_rust_parity import SCALARS, grids, rust_grid  # noqa: F401, F811

from cffdrs import _rust

OUTPUTS = (
    'ws wd wse wse1 wse2 wsx wsy wsv raz m fF fW ffmc isi bui a b c q bui0 be be_max sf rsz rsf isf rsi '
    'latn dj d0 nd fmc fme ffc wfc sfc cfl cfc tfc cbh csfi rso cfb fire_type hros sros cros bfw bisi bros '
    'hfi fi_class accel fuel_type'
).split()
STAGES = (
    'invertWindAspect calcSF calcISZ calcFMC calcISI_RSI_BE calcROS calcSFC getCBH_CFL calcCSFI calcRSO '
    'calcC6BlendCFB calcC6BlendCFC calcC6CROS calcC6HROS calcPercentileCFB calcRosPercentileGrowth '
    'calcCFB calcAccelParam calcFireType calcCFC calcTFC calcHFI calcFireIntensityClass'
).split()


def inputs(shape=()):
    return dict(fuel_type=np.full(shape, 2, dtype=np.int32), lat=52., long=-115.,
                elevation=800., slope=10., aspect=180., pc=50., pdf=35., gfl=.35,
                gcf=80., ws=20., wd=270., ffmc=90., bui=80., wx_date=20230615)


def make(**kwargs):
    return _rust.FbpGrid(**(inputs() | kwargs))


# ---- rust fbpgrid tests ----

@pytest.mark.parametrize('shape', [(), (7,), (2, 3)])
def test_shape_scalar_array_mix(shape):
    grid = _rust.FbpGrid(**inputs(shape))
    assert grid.shape == shape
    grid.run()
    assert grid.get('hros').shape == shape
    assert grid.get('hros').dtype == np.float64
    assert grid.get('hros').item(0) > 0


def test_array_input_can_define_shape_with_scalar_fuel():
    grid = make(fuel_type=2, ws=np.full((2, 3), 20.))
    assert grid.shape == (2, 3)


def test_parity_existing_grid(grids):  # noqa: F811
    grid = _rust.FbpGrid(**grids, **SCALARS, out_request=['hros'])
    grid.run()
    np.testing.assert_array_equal(
        grid.get('hros').view(np.uint64), rust_grid(grids)['hros'].view(np.uint64)
    )


@pytest.mark.parametrize('layout', ['fortran', 'strided', 'negative'])
def test_layout_and_owned_inputs(layout):
    data = inputs((3, 4))
    data['ws'] = np.arange(12, dtype=float).reshape(3, 4)
    expected = _rust.FbpGrid(**data)
    expected.run()
    if layout == 'fortran':
        data = {k: np.asfortranarray(v) if isinstance(v, np.ndarray) else v for k, v in data.items()}
    else:
        for name in ['fuel_type', 'ws']:
            source = np.repeat(data[name], 2, axis=1)
            data[name] = source[:, ::2] if layout == 'strided' else np.ascontiguousarray(data[name][:, ::-1])[:, ::-1]
            if layout == 'negative':
                assert any(stride < 0 for stride in data[name].strides)
    grid = _rust.FbpGrid(**data)
    data['ws'][...] = 100
    data['fuel_type'][...] = 6
    grid.run()
    np.testing.assert_array_equal(grid.get('hros'), expected.get('hros'))


@pytest.mark.parametrize('dtype', [np.float32, np.int8])
def test_numeric_dtypes(dtype):
    grid = make(ws=np.full((3,), 20, dtype=dtype), fuel_type=np.full(3, 2, dtype=np.int32))
    grid.run()
    expected = make(fuel_type=np.full(3, 2, dtype=np.int32))
    expected.run()
    np.testing.assert_array_equal(grid.get('hros'), expected.get('hros'))


@pytest.mark.parametrize(
    'route', ['get', 'set', 'run_stage', 'out_request', 'stages', 'keep', 'set_param', 'set_override']
)
def test_bad_names(route):
    grid = make()
    with pytest.raises(ValueError, match='valid'):
        if route in ['out_request', 'stages', 'keep']:
            make(**{route: ['bogus']})
        elif route in ['set', 'set_param', 'set_override']:
            getattr(grid, route)('bogus', 1)
        else:
            getattr(grid, route)('bogus')


def test_errors():
    with pytest.raises(ValueError, match='threads'):
        make(threads=-1)
    with pytest.raises(ValueError, match='shape'):
        make(fuel_type=np.full((2, 3), 2, dtype=np.int32), lat=np.zeros(6))
    grid = make(fuel_type=np.full((2, 3), 2, dtype=np.int32))
    with pytest.raises(ValueError, match='shape'):
        grid.set_override('dj', np.zeros(6))
    with pytest.raises(ValueError, match='shape'):
        grid.set('ws', np.zeros(6))
    with pytest.raises(ValueError, match='shape'):
        grid.set_fuel(np.full(6, 2, dtype=np.int32))
    with pytest.raises(ValueError):
        grid.set_param('wx_date', 20230229)
    with pytest.raises(ValueError):
        grid.set_param('wx_date', 20230615.5)
    with pytest.raises(ValueError):
        grid.set_override('dj', np.inf)


def test_all_outputs_stepwise_and_planned():
    data = inputs((3, 18))
    data['fuel_type'] = np.tile(np.arange(1, 19, dtype=np.int32), (3, 1))
    fused = _rust.FbpGrid(**data, out_request=OUTPUTS)
    stepped = _rust.FbpGrid(**data, out_request=OUTPUTS)
    planned = _rust.FbpGrid(**data, out_request=OUTPUTS)
    fused.run()
    for name in STAGES:
        stepped.run_stage(name)
    planned.run_planned()
    for name in OUTPUTS:
        np.testing.assert_array_equal(stepped.get(name), fused.get(name), err_msg=name)
        np.testing.assert_array_equal(planned.get(name), fused.get(name), err_msg=name)


def test_lifecycle_freed_fields_and_keep():
    grid = make(out_request=['hros'])
    grid.run()
    expected = grid.get('hros')
    grid.run()
    np.testing.assert_array_equal(grid.get('hros'), expected)
    with pytest.raises(RuntimeError, match=r'call reset\(\) before running stages after run\(\)'):
        grid.run_stage('calcSF')
    grid.reset()
    grid.run_planned()
    with pytest.raises(ValueError, match='sfc.*out_request'):
        grid.get('sfc')
    with pytest.raises(ValueError, match='ffc.*out_request'):   # calcTFC reads ffc first of the freed fields
        grid.run_stage('calcTFC')
    kept = make(out_request=['hros'], keep=['sfc'])
    kept.run_planned()
    assert kept.get('sfc') > 0
    partial = make(stages=['calcSF'], keep=['sf'], out_request=[])
    partial.run_planned()
    assert partial.get('sf') > 1
    assert partial.live_bytes() > 0


def test_set_param_and_set_persist_across_reset():
    grid = make(out_request=['hros', 'dj'])
    grid.set('ws', 0)
    grid.set_fuel(np.array(6, dtype=np.int32))
    grid.set_param('wx_date', 20240615)
    grid.set_param('percentile_growth', 90)
    grid.reset()
    grid.run()
    expected = make(fuel_type=6, ws=0, wx_date=20240615, percentile_growth=90, out_request=['hros', 'dj'])
    expected.run()
    for name in ['hros', 'dj']:
        np.testing.assert_array_equal(grid.get(name), expected.get(name))
    result = grid.get('hros')
    result[...] = -1
    assert grid.get('hros') > 0


def test_overrides_mask_clip_clear_and_readonly_property():
    grid = make(fuel_type=np.full(4, 2, dtype=np.int32), out_request=['dj', 'd0', 'fmc'])
    grid.set_override('d0', np.ma.array([370., 370., 370., 370.], mask=[0, 0, 1, 0]))
    grid.set_override('dj', np.ma.array([0.3, 166.9, 400., 200.], mask=[0, 0, 0, 1]))
    assert grid.dj_override_clipped == 2
    grid.run()
    np.testing.assert_array_equal(grid.get('dj'), [1., 166., 365., np.nan])
    assert np.isnan(grid.get('fmc')[2:]).all()
    grid.set_param('wx_date', 20240615)
    grid.run()
    assert grid.get('dj')[2] == 366
    grid.set_override('dj', None)
    grid.set_override('d0', None)
    assert grid.dj_override_clipped == 0
    grid.run()
    assert np.isfinite(grid.get('fmc')).all()
    with pytest.raises(AttributeError):
        grid.dj_override_clipped = 0


@pytest.mark.parametrize('size', [4096, 100003])
def test_threads_identical(size):
    data = inputs((size,))
    data['fuel_type'] = np.resize(np.arange(1, 19, dtype=np.int32), size)
    data['ws'] = np.linspace(0., 45., size)
    serial = _rust.FbpGrid(**data, threads=1, out_request=OUTPUTS)
    parallel = _rust.FbpGrid(**data, threads=4, out_request=OUTPUTS)
    serial.run()
    parallel.run()
    for name in OUTPUTS:
        np.testing.assert_array_equal(serial.get(name), parallel.get(name), err_msg=name)
    serial.reset()
    parallel.reset()
    serial.run_planned()
    parallel.run_planned()
    for name in OUTPUTS:
        np.testing.assert_array_equal(serial.get(name), parallel.get(name), err_msg=name)


def test_core_version():
    assert _rust.__core_version__ == '0.3.0'






