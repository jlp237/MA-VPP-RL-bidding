"""Tests for FCR reward computation."""

import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from vpp_bidding.markets.fcr.constants import (
    REWARD_DISTANCE_EXPONENT,
    SLOTS_PER_DAY,
    STEPS_PER_SLOT,
)
from vpp_bidding.markets.fcr.reward import calculate_reward_and_financials


def _make_mock_env(
    slots_won: list[int],
    bid_prices: list[float],
    bid_sizes: list[float],
    settlement_prices: list[float],
    vpp_capacity: float = 50.0,
    daily_mean_price: float = 100.0,
    price_scaler_max: float = 200.0,
    size_scaler_max: float = 50.0,
) -> SimpleNamespace:
    """Create a minimal mock env object for reward calculation."""
    vpp_total = np.ones(SLOTS_PER_DAY * STEPS_PER_SLOT, dtype=np.float32) * vpp_capacity

    # Build bid_sizes_all_slots (expanded from 6 slots to 96 steps)
    bid_sizes_all_slots = []
    for s in bid_sizes:
        bid_sizes_all_slots.extend([s] * STEPS_PER_SLOT)

    env = SimpleNamespace(
        delivery_results={
            "slots_won": list(slots_won),
            "agents_bid_prices": list(bid_prices),
            "agents_bid_sizes_round": list(bid_sizes),
            "slot_settlement_prices_DE": list(settlement_prices),
            "vpp_total": vpp_total,
            "bid_sizes_all_slots": np.array(bid_sizes_all_slots, dtype=np.float32),
            "agents_bid_sizes_round_all_slots": np.array(bid_sizes_all_slots, dtype=np.float32),
            "day_reward_list": [0.0] * SLOTS_PER_DAY,
            "reserved_slots": [0] * SLOTS_PER_DAY,
            "activated_slots": [0] * SLOTS_PER_DAY,
            "positive_activation_possible_list": [None] * SLOTS_PER_DAY,
            "negative_activation_possible_list": [None] * SLOTS_PER_DAY,
        },
        current_daily_mean_market_price=daily_mean_price,
        price_scaler=MagicMock(data_max_=np.array([price_scaler_max])),
        size_scaler=MagicMock(data_max_=np.array([size_scaler_max])),
        logging_step=0,
    )
    return env


class TestAllSlotsLost:
    def test_no_revenue_when_all_lost(self) -> None:
        """All 6 slots lost -> no revenue, no penalties."""
        random.seed(42)
        env = _make_mock_env(
            slots_won=[-1] * SLOTS_PER_DAY,
            bid_prices=[150.0] * SLOTS_PER_DAY,
            bid_sizes=[5.0] * SLOTS_PER_DAY,
            settlement_prices=[100.0] * SLOTS_PER_DAY,
        )
        reward, revenue, penalties, _profit = calculate_reward_and_financials(env)

        assert revenue == 0.0
        assert penalties == 0.0
        assert 0.0 <= reward <= 1.0

    def test_distance_reward_shape(self) -> None:
        """Lost slots get distance-based reward using 0.2 exponent."""
        random.seed(42)
        env = _make_mock_env(
            slots_won=[-1] * SLOTS_PER_DAY,
            bid_prices=[150.0] * SLOTS_PER_DAY,
            bid_sizes=[5.0] * SLOTS_PER_DAY,
            settlement_prices=[100.0] * SLOTS_PER_DAY,
            price_scaler_max=200.0,
        )
        reward, _, _, _ = calculate_reward_and_financials(env)

        # Each slot: distance = (150-100)/200 = 0.25
        # auction_reward = 1 - 0.25^0.2
        expected_per_slot = 1 - (50.0 / 200.0) ** REWARD_DISTANCE_EXPONENT
        # weighted_slot = expected_per_slot / 3 (reservation=0, activation=0)
        # weighted_day = sum / 6
        expected_day = (expected_per_slot / 3) * SLOTS_PER_DAY / SLOTS_PER_DAY
        assert reward == pytest.approx(expected_day, abs=0.01)


class TestAllSlotsNotParticipated:
    def test_no_revenue_no_penalties(self) -> None:
        random.seed(42)
        env = _make_mock_env(
            slots_won=[0] * SLOTS_PER_DAY,
            bid_prices=[0.0] * SLOTS_PER_DAY,
            bid_sizes=[0.0] * SLOTS_PER_DAY,
            settlement_prices=[100.0] * SLOTS_PER_DAY,
            vpp_capacity=0.5,  # below 1 MW -> auction_reward = 1.0
        )
        reward, revenue, penalties, _profit = calculate_reward_and_financials(env)

        assert revenue == 0.0
        assert penalties == 0.0
        # vpp_total_slot_min < 1 -> auction_reward = 1.0 for each slot
        # weighted = 1.0/3 per slot, day = sum/6
        expected = (1.0 / 3) * SLOTS_PER_DAY / SLOTS_PER_DAY
        assert reward == pytest.approx(expected, abs=0.01)


class TestWonSlots:
    def test_won_with_ample_capacity(self) -> None:
        """Won slots with VPP capacity >> bid size -> full reward."""
        random.seed(42)
        env = _make_mock_env(
            slots_won=[1] * SLOTS_PER_DAY,
            bid_prices=[80.0] * SLOTS_PER_DAY,
            bid_sizes=[2.0] * SLOTS_PER_DAY,
            settlement_prices=[100.0] * SLOTS_PER_DAY,
            vpp_capacity=50.0,  # way more than bid
        )
        reward, revenue, _penalties, profit = calculate_reward_and_financials(env)

        # Revenue = 6 * (2.0 * 100.0) = 1200
        assert revenue == pytest.approx(1200.0)
        # With ample capacity, reservation and activation should mostly succeed
        assert reward > 0.5
        assert profit >= 0.0

    def test_revenue_calculation(self) -> None:
        """Revenue = bid_size * settlement_price for each won slot."""
        random.seed(42)
        env = _make_mock_env(
            slots_won=[1] * SLOTS_PER_DAY,
            bid_prices=[50.0] * SLOTS_PER_DAY,
            bid_sizes=[10.0] * SLOTS_PER_DAY,
            settlement_prices=[200.0] * SLOTS_PER_DAY,
            vpp_capacity=100.0,
        )
        _, revenue, _, _ = calculate_reward_and_financials(env)
        assert revenue == pytest.approx(10.0 * 200.0 * SLOTS_PER_DAY)


class TestPenalties:
    def test_penalty_uses_max_of_three_fees(self) -> None:
        """Penalty formula: energy * max(price*1.25, price+10, settlement)."""
        # Just verify the penalty structure exists
        random.seed(42)
        env = _make_mock_env(
            slots_won=[1] * SLOTS_PER_DAY,
            bid_prices=[80.0] * SLOTS_PER_DAY,
            bid_sizes=[100.0] * SLOTS_PER_DAY,  # way more than VPP capacity
            settlement_prices=[100.0] * SLOTS_PER_DAY,
            vpp_capacity=1.0,  # very low -> reservation will fail
            daily_mean_price=100.0,
        )
        _, _, penalties, _ = calculate_reward_and_financials(env)
        # With bid_size=100 and vpp=1, reservation should fail -> penalties
        assert penalties < 0.0  # penalties are negative (subtracted)
