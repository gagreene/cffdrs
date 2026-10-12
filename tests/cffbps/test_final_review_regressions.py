"""Parity regressions for the final stateful and scalar export review findings."""
import numpy as np
import pytest
from test_rust_backend import ALL, equal, make


@pytest.mark.parametrize('scalar', [True, False])
@pytest.mark.parametrize('fuel', [2, 19, 20, 0])
@pytest.mark.parametrize(('stage', 'injected', 'fields'), [
    ('calcC6HROS', {'hros': 10.}, ['hros']),
    ('calcC6CROS', {'cros': 12.}, ['cros']),
    ('calcROS', {'rsi': 2., 'brsi': 3., 'be': 4., 'sros': 5.}, ['hros', 'bros', 'sros']),
    ('calcRSO', {'sfc': 2., 'csfi': 300.}, ['rso']),
    ('getCBH_CFL', {'cbh': 5., 'cfl': 3.}, ['cbh', 'cfl']),
])
def test_nonmodeled_stage_consumes_injections(scalar, fuel, stage, injected, fields):
    kwargs = {'fuel_type': fuel if scalar else np.array([fuel], dtype=np.int8)}
    models = [make(backend, **kwargs) for backend in ('rust', 'python')]
    for model in models:
        model.setParams(injected)
        getattr(model, stage)()
    for name, actual, expected in zip(fields, *(m.getParams(fields) for m in models), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=name)
        if scalar:
            assert type(actual) is type(expected), name


@pytest.mark.parametrize('scalar', [True, False])
@pytest.mark.parametrize('fuel', [2, 6, 10, 16])
@pytest.mark.parametrize('missing', ['ws', 'wd', 'ffmc', 'bui', 'slope', 'aspect'])
def test_weather_mask_all_outputs(scalar, fuel, missing):
    kwargs = {'fuel_type': fuel if scalar else np.array([fuel], dtype=np.int8), missing: np.nan}
    equal(make(**kwargs).runFBP(), make('python', **kwargs).runFBP(), scalar=scalar)


@pytest.mark.parametrize('missing', ['ws', 'wd', 'ffmc', 'bui', 'slope'])
def test_scalar_producer_mask_snapshot(missing):
    models = [make(backend, **{missing: np.nan}) for backend in ('rust', 'python')]
    for model in models:
        for stage in ('invertWindAspect', 'calcSF', 'calcISZ', 'calcISI_RSI_BE', 'calcROS'):
            getattr(model, stage)()
        model.setParams({missing: 30.})
    equal(models[0].getParams(ALL), models[1].getParams(ALL), scalar=True)


@pytest.mark.parametrize('missing', ['ws', 'lat', 'fuel_type', 'd0'])
def test_scalar_independent_injected_producer(missing):
    models = [make(backend, **{missing: 0 if missing == 'fuel_type' else np.nan}) for backend in ('rust', 'python')]
    for model in models:
        model.setParams({'hros': 2., 'tfc': 3.})
        model.calcHFI()
        model.calcFireIntensityClass()
    fields = ['hros', 'tfc', 'hfi', 'fi_class']
    for name, actual, expected in zip(fields, *(m.getParams(fields) for m in models), strict=True):
        assert actual == expected, name
        assert type(actual) is type(expected), name


@pytest.mark.parametrize('scalar', [True, False])
@pytest.mark.parametrize('activation', [0., 1., np.nan, np.inf])
def test_c6_activation_consumes_injection_not_current_weather(scalar, activation):
    kwargs = {'fuel_type': 6 if scalar else np.array([6]), 'bui': np.nan}
    models = [make(backend, **kwargs) for backend in ('rust', 'python')]
    for model in models:
        model.setParams({'c6_blend_cfc': activation, 'isi': 10., 'fme': 1.})
        model.calcC6CROS()
        model.setParams({'bui': 80.})
    actual, expected = (model.getParams(['cros'])[0] for model in models)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12, equal_nan=True)
    if scalar:
        assert type(actual) is type(expected)


@pytest.mark.parametrize('missing', ['d0', 'dj', 'lat', 'long', 'elevation'])
def test_scalar_date_and_geography_controls(missing):
    equal(make(**{missing: np.nan}).runFBP(), make('python', **{missing: np.nan}).runFBP(), scalar=True)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, 1000.])
def test_scalar_injected_intensity_mask_and_infinity(value):
    models = [make(backend, ws=np.nan) for backend in ('rust', 'python')]
    for model in models:
        model.setParams({'hfi': value})
        model.calcFireIntensityClass()
        model.setParams({'hfi': 100.})
    actual, expected = (model.getParams(['fi_class'])[0] for model in models)
    assert actual == expected
    assert type(actual) is type(expected)


def test_unmasked_growth_nan_remains_nonfinite():
    equal(make(percentile_growth=np.nan).runFBP(),
          make('python', percentile_growth=np.nan).runFBP(), scalar=True)


def test_scalar_new_missing_input_does_not_remask_old_producer():
    models = [make(backend) for backend in ('rust', 'python')]
    for model in models:
        for stage in ('invertWindAspect', 'calcSF', 'calcISZ', 'calcISI_RSI_BE', 'calcROS'):
            getattr(model, stage)()
        model.setParams({'ws': np.nan, 'lat': np.nan, 'fuel_type': 0})
        model.setParams({'tfc': 3.})
        model.calcHFI()
        model.calcFireIntensityClass()
    fields = ['hros', 'hfi', 'fi_class']
    for name, actual, expected in zip(fields, *(model.getParams(fields) for model in models), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12, err_msg=name)
        assert type(actual) is type(expected), name


@pytest.mark.parametrize('keep', [False, True])
@pytest.mark.parametrize('fuel', [0, 19, 20])
def test_native_nonmodeled_rso_retention(keep, fuel):
    from test_rust_fbpgrid import make as native_make

    grid = native_make(fuel_type=fuel, stages=['calcRSO'], out_request=[], keep=['rso'] if keep else [])
    grid.set('sfc', 2.)
    grid.set('csfi', 300.)
    grid.run_stage('calcRSO')
    if keep:
        assert grid.get('rso').item() == 0.5
    else:
        with pytest.raises(ValueError, match='freed'):
            grid.get('rso')
    with pytest.raises(ValueError, match='freed'):
        grid.run_stage('calcRSO')


def test_scalar_provenance_cannot_restore_freed_storage():
    model = make(out_request=['hfi'], ws=np.nan)
    model.runFBP()
    with pytest.raises(ValueError, match='freed'):
        model.getParams(['hros'])


@pytest.mark.parametrize('scalar', [True, False])
@pytest.mark.parametrize('route', ['set_override', 'setParams', 'attribute'])
@pytest.mark.parametrize('name', ['d0', 'dj'])
def test_missing_date_override_exports_nan_before_and_after_calcfmc(name, route, scalar):
    kwargs = {} if scalar else {'fuel_type': np.array([[2, 2]]), 'lat': np.array([[55., 55.]])}
    models = [make(backend, **kwargs) for backend in ('rust', 'python')]
    for model in models:
        if route == 'set_override':
            model.set_override(name, np.nan)
        elif route == 'setParams':
            model.setParams({name: np.nan})
        else:
            setattr(model, name, np.nan)
    for stage in (None, 'calcFMC'):
        if stage:
            for model in models:
                model.calcFMC()
        actual, expected = (np.asarray(m.getParams([name])[0], dtype=float) for m in models)
        assert np.isnan(expected).all()
        assert np.isnan(actual).all(), f'{name} via {route} after {stage}: {actual}'


@pytest.mark.parametrize('missing', ['lat', 'long', 'elevation'])
@pytest.mark.parametrize('fuel', [10, 11, 12, 13])
@pytest.mark.parametrize('percentile', [10, 90])
def test_scalar_mixedwood_growth_keeps_the_foliar_moisture_mask(missing, fuel, percentile):
    kwargs = {'fuel_type': fuel, 'percentile_growth': percentile, missing: np.nan}
    equal(make(**kwargs).runFBP(), make('python', **kwargs).runFBP(), scalar=True)


@pytest.mark.parametrize('route', ['initialize', 'setParams', 'attribute'])
@pytest.mark.parametrize('kwargs', [{}, {'fuel_type': 10, 'lat': np.nan}])
def test_percentile_growth_none_means_no_adjustment(route, kwargs):
    models = []
    for backend in ('rust', 'python'):
        model = make(backend, **kwargs, **({'percentile_growth': None} if route == 'initialize' else {}))
        if route == 'setParams':
            model.setParams({'percentile_growth': None})
        elif route == 'attribute':
            model.percentile_growth = None
        models.append(model)
    equal(models[0].runFBP(), models[1].runFBP(), scalar=True)
    reference = make('python', **kwargs, percentile_growth=50).runFBP()
    equal(models[1].runFBP(), reference, scalar=True)
