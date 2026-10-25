"""Static lookup tables and parameter sets for CFFBPS.

These are read-only reference data (fuel-type codes, rate-of-spread parameters,
canopy tables). They were previously defined at module scope and inside
``FBP.__init__``; centralizing them here lets the equation functions import them
directly. Tables are runtime-immutable: dicts are exposed as MappingProxyType views and
sequences as tuples, so accidental mutation raises instead of silently leaking
across FBP instances.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

# CFFBPS Fuel Type Numeric->Alphanumeric Code Lookup Table
fbpFTCode_NumToAlpha_LUT: Final[Mapping[int, str]] = MappingProxyType({
    1: 'C1',    # C-1
    2: 'C2',    # C-2
    3: 'C3',    # C-3
    4: 'C4',    # C-4
    5: 'C5',    # C-5
    6: 'C6',    # C-6
    7: 'C7',    # C-7
    8: 'D1',    # D-1
    9: 'D2',    # D-2
    10: 'M1',   # M-1
    11: 'M2',   # M-2
    12: 'M3',   # M-3
    13: 'M4',   # M-4
    14: 'O1a',  # O-1a
    15: 'O1b',  # O-1b
    16: 'S1',   # S-1
    17: 'S2',   # S-2
    18: 'S3',   # S-3
    19: 'NF',   # NF (non-fuel)
    20: 'WA',   # WA (water)
})

# CFFBPS Fuel Type Alphanumeric->Numeric Code Lookup Table
fbpFTCode_AlphaToNum_LUT: Final[Mapping[str, int]] = MappingProxyType({
    'C1': 1,    # C-1
    'C2': 2,    # C-2
    'C3': 3,    # C-3
    'C4': 4,    # C-4
    'C5': 5,    # C-5
    'C6': 6,    # C-6
    'C7': 7,    # C-7
    'D1': 8,    # D-1
    'D2': 9,    # D-2
    'M1': 10,   # M-1
    'M2': 11,   # M-2
    'M3': 12,   # M-3
    'M4': 13,   # M-4
    'O1a': 14,  # O-1a
    'O1b': 15,  # O-1b
    'S1': 16,   # S-1
    'S2': 17,   # S-2
    'S3': 18,   # S-3
    'NF': 19,   # NF (non-fuel)
    'WA': 20,   # WA (water)
})

# List of valid CFFBPS output parameters
valid_outputs: Final[tuple[str, ...]] = (
    'fire_type', 'hros', 'hfi', 'fuel_type', 'ws', 'wd', 'm', 'fF', 'fW', 'ffmc', 'bui', 'isi',
    'a', 'b', 'c', 'rsz', 'sf', 'rsf', 'isf', 'rsi', 'wse1', 'wse2', 'wse', 'wsx', 'wsy', 'wsv', 'raz',
    'q', 'bui0', 'be', 'be_max', 'ffc', 'wfc', 'sfc', 'latn', 'dj', 'd0', 'nd', 'fmc', 'fme',
    'csfi', 'rso', 'bfw', 'bisi', 'bros', 'sros', 'cros', 'cbh', 'cfb', 'cfl', 'cfc', 'tfc', 'accel', 'fi_class'
)

# ### Lists for CFFBPS Crown Fire Metric variables
csfiVarList: Final[tuple[str, ...]] = ('cbh', 'fmc')
rsoVarList: Final[tuple[str, ...]] = ('csfi', 'sfc')
cfbVarList: Final[tuple[str, ...]] = ('cros', 'rso')
cfcVarList: Final[tuple[str, ...]] = ('cfb', 'cfl')
cfiVarList: Final[tuple[str, ...]] = ('cros', 'cfc')

# List of open fuel type codes
open_fuel_types: Final[tuple[int, ...]] = (1, 7, 9, 14, 15, 16, 17, 18)

# List of non-crowning fuel type codes
non_crowning_fuels: Final[tuple[int, ...]] = (8, 9, 14, 15, 16, 17, 18)

# CFFBPS Canopy Base Height & Canopy Fuel Load Lookup Table (cbh, cfl, ht)
fbpCBH_CFL_HT_LUT: Final[Mapping[int, tuple[float | None, float | None, float | None]]] = MappingProxyType({
    1: (2, 0.75, 10),
    2: (3, 0.8, 7),
    3: (8, 1.15, 18),
    4: (4, 1.2, 10),
    5: (18, 1.2, 25),
    6: (7, 1.8, 14),
    7: (10, 0.5, 20),
    8: (0, 0, 0),
    9: (0, 0, 0),
    10: (6, 0.8, 13),
    11: (6, 0.8, 13),
    12: (6, 0.8, 8),
    13: (6, 0.8, 8),
    14: (0, 0, 0),
    15: (0, 0, 0),
    16: (0, 0, 0),
    17: (0, 0, 0),
    18: (0, 0, 0),
    19: (None, None, None),
    20: (None, None, None),
})

# CFFBPS Surface Fire Rate of Spread Parameters (a, b, c, q, BUI0, be_max)
rosParams: Final[Mapping[int, tuple]] = MappingProxyType({
    1: (90, 0.0649, 4.5, 0.9, 72, 1.076),    # C-1
    2: (110, 0.0282, 1.5, 0.7, 64, 1.321),   # C-2
    3: (110, 0.0444, 3, 0.75, 62, 1.261),    # C-3
    4: (110, 0.0293, 1.5, 0.8, 66, 1.184),   # C-4
    5: (30, 0.0697, 4, 0.8, 56, 1.220),      # C-5
    6: (30, 0.08, 3, 0.8, 62, 1.197),        # C-6
    7: (45, 0.0305, 2, 0.85, 106, 1.134),    # C-7
    8: (30, 0.0232, 1.6, 0.9, 32, 1.179),    # D-1
    9: (30, 0.0232, 1.6, 0.9, 32, 1.179),    # D-2
    10: (None, None, None, 0.8, 50, 1.250),  # M-1
    11: (None, None, None, 0.8, 50, 1.250),  # M-2
    12: (120, 0.0572, 1.4, 0.8, 50, 1.250),  # M-3
    13: (100, 0.0404, 1.48, 0.8, 50, 1.250),  # M-4
    14: (190, 0.0310, 1.4, 1, None, 1),      # O-1a
    15: (250, 0.0350, 1.7, 1, None, 1),      # O-1b
    16: (75, 0.0297, 1.3, 0.75, 38, 1.460),  # S-1
    17: (40, 0.0438, 1.7, 0.75, 63, 1.256),  # S-2
    18: (55, 0.0829, 3.2, 0.75, 31, 1.590),  # S-3
})
