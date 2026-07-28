"""ROS percentile growth and point-ignition acceleration parameter."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy import ma as mask
from scipy.stats import t

MaskedArray = mask.MaskedArray


def _tinv(probability: float | int, freedom: int = 9999999):
    """Inverse of the Student's t-distribution (quantile function)."""
    return t.ppf(probability, freedom)


def calc_ros_percentile_growth(*,
                               percentile_growth: float | int | None,
                               fuel_type: MaskedArray,
                               cfb: MaskedArray,
                               hros: MaskedArray,
                               bros: MaskedArray) -> tuple[MaskedArray, MaskedArray]:
    """Adjust head/backing ROS by a percentile-growth factor (from the WISE code base).

    Returns ``(hros, bros)`` unchanged when ``percentile_growth`` is None or 50.
    """
    if (percentile_growth is not None) and (percentile_growth != 50):
        # Calculate the inverse t-distribution for the given percentile growth
        tinv_value = _tinv(probability=percentile_growth / 100, freedom=9999999)

        # Prepare default table with structured dtype
        keys = np.array([1, 2, 3, 4, 5, 6, 7, 8, 12], dtype=np.uint8)
        surface_vals = np.array([-1.0, 0.84, 0.62, 0.74, 0.8, 0.66, 1.22, 0.716, 0.551], dtype=np.float32)
        crown_vals = np.array([0.95, 1.82, 1.78, 1.38, -1.0, 1.54, 1.0, -1.0, -1.0], dtype=np.float32)

        # Initialize default arrays for lookup
        surface_s = np.full_like(fuel_type, np.nan, dtype=np.float32)
        crown_s = np.full_like(fuel_type, np.nan, dtype=np.float32)

        # Create a mask for each valid fuel type and assign values
        for k, s_val, c_val in zip(keys, surface_vals, crown_vals, strict=False):
            valid_mask = fuel_type == k
            surface_s[valid_mask] = s_val
            crown_s[valid_mask] = c_val

        e = tinv_value * crown_s

        # Iterate over head fire and backing fire ROS values
        out = {}
        for name, ros_in in (('hros', hros), ('bros', bros)):
            d = mask.power(ros_in, 0.6)  # Apply a power transformation to the ROS value

            # Calculate the adjusted ROS growth based on crown and surface spread parameters
            ros_growth = mask.where(~np.isnan(crown_s),
                                    mask.where(cfb < 0.1,
                                               mask.where(surface_s < 0,
                                                          # No adjustment if surface_s is invalid
                                                          ros_in,
                                                          # Adjust using surface_s
                                                          np.exp(tinv_value) * ros_in),
                                               mask.where(crown_s < 0,
                                                          # No adjustment if crown_s is invalid
                                                          ros_in,
                                                          mask.where(-e > d,
                                                                     # Adjust using crown_s
                                                                     mask.exp(tinv_value) * ros_in,
                                                                     # Apply growth adjustment
                                                                     mask.power(d + e, 1 / 0.6)
                                                                     )
                                                          )
                                               ),
                                    # Default to the original ROS value if no conditions are met
                                    ros_in)
            out[name] = ros_growth

        hros, bros = out['hros'], out['bros']

    return hros, bros


def calc_accel_param(*,
                     fuel_type: MaskedArray,
                     ftypes: Sequence[int],
                     open_fuel_types: Sequence[int],
                     cfb: MaskedArray,
                     accel_param: MaskedArray) -> MaskedArray:
    """Calculate the acceleration parameter for a point-ignition fire.

    ``accel_param`` is passed in as the initialized template.
    """
    # Mask for open fuel types that use a fixed acceleration parameter (0.115)
    fixed_accel_mask = mask.where(np.isin(fuel_type, open_fuel_types), True, False)

    # Mask for closed fuel types that require computation
    variable_accel_mask = mask.where(np.isin(fuel_type, ftypes) & ~fixed_accel_mask, True, False)

    # Compute acceleration parameter for open fuel types
    accel_param = mask.where(fixed_accel_mask, 0.115, accel_param)

    # Compute acceleration parameter for closed fuel types (safe calculation)
    accel_param = mask.where(variable_accel_mask,
                             0.115 - 18.8 * np.power(cfb, 2.5) * np.exp(-8 * cfb),
                             accel_param)

    return accel_param
