"""FCR reward computation and financial settlement.

Computes the shaped reward signal and actual financial P&L for a day of
FCR market participation. Calls reservation and activation simulations
internally for won slots.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vpp_bidding.markets.fcr.activation import simulate_activation
from vpp_bidding.markets.fcr.constants import (
    PENALTY_SURCHARGE_EUR,
    PENALTY_SURCHARGE_FACTOR,
    REWARD_DISTANCE_EXPONENT,
    SLOTS_PER_DAY,
    STEPS_PER_SLOT,
)
from vpp_bidding.markets.fcr.reservation import simulate_reservation

if TYPE_CHECKING:
    from vpp_bidding.markets.fcr.protocols import FCRRewardEnv

logger = logging.getLogger(__name__)


def calculate_reward_and_financials(env: FCRRewardEnv) -> tuple[float, float, float, float]:
    """Compute reward and financials for a full day of FCR participation.

    Mutates ``env.delivery_results`` in-place (reservation/activation).
    As a side effect, reinitialises ``env.delivery_results["total_not_reserved_energy"]``
    and ``env.delivery_results["total_not_delivered_energy"]`` to zero-filled lists of
    length ``SLOTS_PER_DAY``.

    Returns:
        (weighted_day_reward, day_revenue, day_penalties, day_profit)

    Note: Penalty model is a simplified approximation from the original thesis.
    Actual German TSO penalty rules (Geschaeftsbedingungen) vary by TSO and year:
    - Tennet: max(spot price, activation price) * 1.5
    - 50Hertz: similar structure with different multipliers
    The simplified model uses: max(market_price * 1.25, market_price + 10 EUR, settlement_price)
    This captures the general structure but not exact regulatory values.
    """
    env.delivery_results["total_not_reserved_energy"] = [0] * SLOTS_PER_DAY
    env.delivery_results["total_not_delivered_energy"] = [0] * SLOTS_PER_DAY

    day_profit: float = 0.0
    day_revenue: float = 0.0
    day_penalties: float = 0.0
    day_reward: float = 0.0
    day_reward_list: list[float] = []

    for slot in range(len(env.delivery_results["slots_won"])):
        auction_reward: float = 0.0
        reservation_reward: float = 0.0
        activation_reward: float = 0.0
        slot_profit: float = 0.0
        slot_revenue: float = 0.0
        slot_penalty: float = 0.0

        # --- LOST ---
        if env.delivery_results["slots_won"][slot] == -1:
            slot_settlement_price = env.delivery_results["slot_settlement_prices_DE"][slot]
            agents_bid_price = env.delivery_results["agents_bid_prices"][slot]
            distance_to_settlement_price = agents_bid_price - slot_settlement_price
            auction_reward = (
                1
                - (distance_to_settlement_price / env.price_scaler.data_max_[0])
                ** REWARD_DISTANCE_EXPONENT
            )

        # --- NOT PARTICIPATED ---
        if env.delivery_results["slots_won"][slot] == 0:
            vpp_total_slot_min = min(
                env.delivery_results["vpp_total"][
                    slot * STEPS_PER_SLOT : (slot + 1) * STEPS_PER_SLOT
                ]
            )
            if vpp_total_slot_min >= 1:
                distance_to_vpp_capacity = vpp_total_slot_min
                auction_reward = (
                    1
                    - (distance_to_vpp_capacity / env.size_scaler.data_max_[0])
                    ** REWARD_DISTANCE_EXPONENT
                )
            else:
                auction_reward = 1.0

        # --- WON ---
        if env.delivery_results["slots_won"][slot] == 1:
            auction_reward = 1.0

            agents_bid_size = env.delivery_results["agents_bid_sizes_round"][slot]
            basic_compensation = (
                agents_bid_size * env.delivery_results["slot_settlement_prices_DE"][slot]
            )
            slot_revenue = basic_compensation

            # Simulate reservation
            simulate_reservation(env, slot)

            # Reservation FAILED
            if env.delivery_results["reserved_slots"][slot] == -1:
                penalty_fee_1 = env.current_daily_mean_market_price * PENALTY_SURCHARGE_FACTOR
                penalty_fee_2 = env.current_daily_mean_market_price + PENALTY_SURCHARGE_EUR
                penalty_fee_3 = env.delivery_results["slot_settlement_prices_DE"][slot]
                penalty_list = [penalty_fee_1, penalty_fee_2, penalty_fee_3]
                penalty_fee_reservation = env.delivery_results["total_not_reserved_energy"][
                    slot
                ] * max(penalty_list)
                slot_penalty -= penalty_fee_reservation

                vpp_total_slot_min = min(
                    env.delivery_results["vpp_total"][
                        slot * STEPS_PER_SLOT : (slot + 1) * STEPS_PER_SLOT
                    ]
                )
                distance_to_vpp_capacity = agents_bid_size - vpp_total_slot_min
                reservation_reward = (
                    1
                    - (distance_to_vpp_capacity / env.size_scaler.data_max_[0])
                    ** REWARD_DISTANCE_EXPONENT
                )

            # Reservation SUCCEEDED
            if env.delivery_results["reserved_slots"][slot] == 1:
                reservation_reward = 1.0

                # Simulate activation
                simulate_activation(env, slot)

                # Activation FAILED
                if env.delivery_results["activated_slots"][slot] == -1:
                    penalty_fee_1 = env.current_daily_mean_market_price * PENALTY_SURCHARGE_FACTOR
                    penalty_fee_2 = env.current_daily_mean_market_price + PENALTY_SURCHARGE_EUR
                    penalty_fee_3 = env.delivery_results["slot_settlement_prices_DE"][slot]
                    penalty_list = [penalty_fee_1, penalty_fee_2, penalty_fee_3]
                    penalty_fee_activation = env.delivery_results["total_not_delivered_energy"][
                        slot
                    ] * max(penalty_list)
                    slot_penalty -= penalty_fee_activation

                    positive_list = env.delivery_results["positive_activation_possible_list"][slot]
                    negative_list = env.delivery_results["negative_activation_possible_list"][slot]
                    joined = positive_list + negative_list
                    activation_possible_count = sum(x is True for x in joined)
                    activation_reward = (
                        activation_possible_count / STEPS_PER_SLOT
                    ) ** REWARD_DISTANCE_EXPONENT

                # Activation SUCCEEDED
                if env.delivery_results["activated_slots"][slot] == 1:
                    activation_reward = 1.0

        slot_reward = auction_reward + reservation_reward + activation_reward
        weighted_slot_reward = slot_reward / 3
        day_reward += weighted_slot_reward

        slot_profit = 0.0 if abs(slot_penalty) >= slot_revenue else slot_revenue - abs(slot_penalty)

        day_reward_list.append(weighted_slot_reward)
        day_revenue += slot_revenue
        day_penalties += slot_penalty
        day_profit += slot_profit

    weighted_day_reward = day_reward / SLOTS_PER_DAY
    env.delivery_results["day_reward_list"] = day_reward_list

    return weighted_day_reward, day_revenue, day_penalties, day_profit
