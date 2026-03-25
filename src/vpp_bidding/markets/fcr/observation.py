"""FCR observation construction.

Builds the FCR-specific portion of the observation vector that the RL
agent receives at each step.
"""

from __future__ import annotations

import logging

import numpy as np

from vpp_bidding.markets.fcr.constants import SLOTS_PER_DAY, STEPS_PER_DAY

logger = logging.getLogger(__name__)


def get_fcr_observation(
    fcr_prices: np.ndarray,
    vpp_capacity: np.ndarray,
    current_step: int,
    price_scaler_min: float,
    price_scaler_max: float,
    size_scaler_min: float,
    size_scaler_max: float,
) -> np.ndarray:
    """Build the FCR-specific observation vector.

    Args:
        fcr_prices: Array of settlement prices per slot (length ``SLOTS_PER_DAY``).
        vpp_capacity: Array of available VPP capacity per quarter-hour
            (length ``STEPS_PER_DAY``).
        current_step: The current quarter-hour step within the day (0-95).
        price_scaler_min: Minimum value for price normalisation.
        price_scaler_max: Maximum value for price normalisation.
        size_scaler_min: Minimum value for size normalisation.
        size_scaler_max: Maximum value for size normalisation.

    Returns:
        A 1-D numpy array containing the normalised FCR observation features.
    """
    price_range = price_scaler_max - price_scaler_min
    size_range = size_scaler_max - size_scaler_min

    # Normalise prices
    if price_range > 0:
        norm_prices = (fcr_prices - price_scaler_min) / price_range
    else:
        norm_prices = np.zeros(SLOTS_PER_DAY, dtype=np.float32)

    # Normalise VPP capacity
    if size_range > 0:
        norm_capacity = (vpp_capacity - size_scaler_min) / size_range
    else:
        norm_capacity = np.zeros(STEPS_PER_DAY, dtype=np.float32)

    # Time encoding
    time_feature = np.array(
        [current_step / STEPS_PER_DAY],
        dtype=np.float32,
    )

    return np.concatenate([norm_prices, norm_capacity, time_feature])
