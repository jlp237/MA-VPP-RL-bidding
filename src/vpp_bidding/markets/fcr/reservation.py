"""FCR reservation check.

Verifies that the VPP can maintain the reserved FCR capacity across all
timesteps within a slot, both in the positive and negative direction.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vpp_bidding.markets.fcr.constants import (
    RESERVATION_STEP_DURATION_HOURS,
    STEPS_PER_SLOT,
)

if TYPE_CHECKING:
    from vpp_bidding.markets.fcr.protocols import FCREnv

logger = logging.getLogger(__name__)


def simulate_reservation(env: FCREnv, slot: int) -> None:
    """Simulate FCR reservation for all timesteps in a slot.

    Checks negative and positive reservation feasibility.
    Mutates ``env.delivery_results`` in-place.

    Symmetric FCR requires the VPP to hold capacity for both directions:
    - Positive FCR (increase output / inject power): bid_size <= available_capacity
    - Negative FCR (decrease output / absorb power): bid_size <= available_capacity
    Both must hold simultaneously for all 16 timesteps (15-min intervals) in the slot.
    If either check fails at any timestep, the slot reservation fails.
    """
    vpp_total_slot = env.delivery_results["vpp_total"][
        slot * STEPS_PER_SLOT : (slot + 1) * STEPS_PER_SLOT
    ]
    bid_sizes_per_slot = env.delivery_results["bid_sizes_all_slots"][
        slot * STEPS_PER_SLOT : (slot + 1) * STEPS_PER_SLOT
    ]

    reservation_possible_list: list[bool] = []
    total_not_reserved_energy: float = 0.0

    for time_step in range(STEPS_PER_SLOT):
        agent_bid_size = bid_sizes_per_slot[time_step]
        available_vpp_capacity = vpp_total_slot[time_step]

        # Symmetric FCR: VPP must hold capacity for both directions
        reservation_possible = (available_vpp_capacity - agent_bid_size) > 0

        if not reservation_possible:
            not_reserved_capacity = abs(available_vpp_capacity - agent_bid_size)
            total_not_reserved_energy += not_reserved_capacity * RESERVATION_STEP_DURATION_HOURS

        reservation_possible_list.append(reservation_possible)

    if all(reservation_possible_list):
        env.delivery_results["reserved_slots"][slot] = 1
    else:
        env.delivery_results["reserved_slots"][slot] = -1

    env.delivery_results["total_not_reserved_energy"][slot] = total_not_reserved_energy
