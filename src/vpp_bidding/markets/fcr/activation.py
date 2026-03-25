"""FCR activation simulation.

Determines whether the VPP can deliver contracted FCR capacity when activated.
Activation share is sampled from a weighted distribution; delivery is checked
against available VPP capacity at each 15-min timestep.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

import numpy as np

from vpp_bidding.markets.fcr.constants import (
    ACTIVATION_POPULATION,
    ACTIVATION_STEP_DURATION_HOURS,
    ACTIVATION_SUCCESS_PROBABILITY,
    ACTIVATION_WEIGHTS,
    SLOTS_PER_DAY,
    STEPS_PER_SLOT,
)

if TYPE_CHECKING:
    from vpp_bidding.markets.fcr.protocols import FCREnv

logger = logging.getLogger(__name__)


def prepare_activation(env: FCREnv) -> None:
    """Expand bid sizes from 6 slots to 96 timesteps and initialise activation tracking.

    Mutates ``env.delivery_results`` in-place.
    """
    bid_sizes_list: list[float] = []
    for slot in range(SLOTS_PER_DAY):
        for _step in range(STEPS_PER_SLOT):
            bid_sizes_list.append(env.delivery_results["agents_bid_sizes_round"][slot])
    bid_sizes_all_slots = np.array(bid_sizes_list)
    env.delivery_results["agents_bid_sizes_round_all_slots"] = bid_sizes_all_slots
    env.delivery_results["bid_sizes_all_slots"] = bid_sizes_all_slots

    env.delivery_results["reserved_slots"] = [0] * SLOTS_PER_DAY
    env.delivery_results["activated_slots"] = [0] * SLOTS_PER_DAY
    env.delivery_results["positive_activation_possible_list"] = [None] * SLOTS_PER_DAY
    env.delivery_results["negative_activation_possible_list"] = [None] * SLOTS_PER_DAY


def check_activation_possible(
    env: Any,
    agent_bid_size: float,
    vpp_total_step: float,
) -> tuple[bool | None, float]:
    """Check whether a single activation timestep can be delivered.

    Activation share is sampled from empirical distribution (Seidel & Haase VPP
    reliability model). Delivery success probability accounts for stochastic
    VPP component availability.

    Returns:
        (activation_possible, not_delivered_energy)
    """
    not_delivered_capacity: float = 0.0
    activation_possible: bool | None = None

    max_activation_share: float = random.choices(
        population=ACTIVATION_POPULATION,
        weights=ACTIVATION_WEIGHTS,
        k=1,
    )[0]
    capacity_to_activate = max_activation_share * agent_bid_size

    if abs(capacity_to_activate) > vpp_total_step:
        activation_possible = False
        not_delivered_capacity = abs(vpp_total_step - abs(capacity_to_activate))
        available_capacity = vpp_total_step
        not_delivered_capacity += available_capacity * (1 - ACTIVATION_SUCCESS_PROBABILITY)
    else:
        activation_possible = True
        not_delivered_capacity = abs(capacity_to_activate) * (1 - ACTIVATION_SUCCESS_PROBABILITY)

    not_delivered_energy = not_delivered_capacity * ACTIVATION_STEP_DURATION_HOURS

    return activation_possible, not_delivered_energy


def simulate_activation(env: FCREnv, slot: int) -> None:
    """Simulate FCR activation for all timesteps in a slot.

    Checks both positive and negative FCR for each timestep.
    Mutates ``env.delivery_results`` in-place.
    """
    vpp_total_slot = env.delivery_results["vpp_total"][
        slot * STEPS_PER_SLOT : (slot + 1) * STEPS_PER_SLOT
    ]
    bid_sizes_per_slot = env.delivery_results["bid_sizes_all_slots"][
        slot * STEPS_PER_SLOT : (slot + 1) * STEPS_PER_SLOT
    ]

    positive_activation_possible_list: list[bool | None] = []
    negative_activation_possible_list: list[bool | None] = []
    total_not_delivered_energy: float = 0.0

    for time_step in range(STEPS_PER_SLOT):
        agent_bid_size = bid_sizes_per_slot[time_step]

        # Positive FCR
        activation_possible, not_delivered_energy = check_activation_possible(
            env, agent_bid_size, vpp_total_slot[time_step]
        )
        positive_activation_possible_list.append(activation_possible)
        total_not_delivered_energy += not_delivered_energy

        # Negative FCR
        activation_possible, not_delivered_energy = check_activation_possible(
            env, -agent_bid_size, vpp_total_slot[time_step]
        )
        negative_activation_possible_list.append(activation_possible)
        total_not_delivered_energy += not_delivered_energy

    if all(positive_activation_possible_list) and all(negative_activation_possible_list):
        env.delivery_results["activated_slots"][slot] = 1
    else:
        env.delivery_results["activated_slots"][slot] = -1

    env.delivery_results["total_not_delivered_energy"][slot] = total_not_delivered_energy
    env.delivery_results["positive_activation_possible_list"][slot] = (
        positive_activation_possible_list
    )
    env.delivery_results["negative_activation_possible_list"][slot] = (
        negative_activation_possible_list
    )
