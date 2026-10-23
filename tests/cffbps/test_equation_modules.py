"""Where each equation lives: one module per concern (consumption, crown determination, behavior, ...)."""
import importlib

import pytest

PACKAGE = 'cffdrs.cffbps.python.equations'
LAYOUT = {
    'consumption': ['zero_without_spread', 'calc_sfc', 'calc_cfc', 'calc_tfc'],
    'crown': ['calc_cbh_cfl', 'calc_csfi', 'calc_rso', 'calc_cfb', 'calc_c6_blend_cfb'],
    'behavior': ['calc_fire_type', 'calc_hfi', 'calc_fire_intensity_class'],
    'growth': ['calc_ros_percentile_growth', 'calc_accel_param'],
    'ros': ['calc_ros', 'calc_c6_cros', 'calc_c6_hros'],
    'slope_wind': ['invert_wind_aspect', 'calc_sf', 'calc_isz', 'calc_slope_wind_isi', 'calc_isi_rsi_be'],
    'foliar_moisture': ['calc_latn', 'calc_d0', 'calc_dj', 'calc_nd', 'calc_fmc', 'calc_fme', 'calc_foliar_moisture'],
}


@pytest.mark.parametrize('module,names', LAYOUT.items())
def test_every_equation_is_defined_in_its_module(module, names):
    mod = importlib.import_module(f'{PACKAGE}.{module}')
    for name in names:
        assert getattr(mod, name).__module__ == mod.__name__, (module, name)


def test_a_function_has_one_home_only():
    owners = {}
    for module, names in LAYOUT.items():
        for name in names:
            assert name not in owners, (name, module, owners.get(name))
            owners[name] = module
    for module, names in LAYOUT.items():
        mod = importlib.import_module(f'{PACKAGE}.{module}')
        public = {n for n, v in vars(mod).items()
                  if n.startswith('calc_') and getattr(v, '__module__', None) == mod.__name__}
        assert public <= set(names), (module, public - set(names))


def test_the_surface_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f'{PACKAGE}.surface')
