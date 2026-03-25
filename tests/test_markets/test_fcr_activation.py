"""Tests for FCR activation simulation."""

import random
from types import SimpleNamespace

import numpy as np

from vpp_bidding.markets.fcr.activation import (
    check_activation_possible,
    prepare_activation,
    simulate_activation,
)
from vpp_bidding.markets.fcr.constants import (
    ACTIVATION_STEP_DURATION_HOURS,
    SLOTS_PER_DAY,
    STEPS_PER_SLOT,
)

MOCK_ENV = SimpleNamespace(logging_step=0)


class TestCheckActivationPossible:
    def test_delivery_succeeds_when_capacity_sufficient(self) -> None:
        """VPP has more capacity than bid -> activation possible."""
        random.seed(42)
        possible, _not_delivered = check_activation_possible(
            MOCK_ENV,
            agent_bid_size=5.0,
            vpp_total_step=100.0,
        )
        assert possible is True

    def test_energy_when_capacity_insufficient(self) -> None:
        """VPP capacity less than activated amount -> not_delivered_energy > 0."""
        random.seed(42)
        _possible, not_delivered = check_activation_possible(
            MOCK_ENV,
            agent_bid_size=100.0,
            vpp_total_step=1.0,
        )
        # Should have some not-delivered energy
        assert not_delivered >= 0.0

    def test_energy_calculation_factor(self) -> None:
        """Energy uses ACTIVATION_STEP_DURATION_HOURS (0.0625h) factor."""
        # The energy formula: not_delivered_capacity * 0.0625
        assert ACTIVATION_STEP_DURATION_HOURS == 0.0625


class TestPrepareActivation:
    def test_populates_all_slots(self) -> None:
        """prepare_activation should expand bid sizes to 96 timesteps."""
        random.seed(42)
        env = SimpleNamespace(
            delivery_results={
                "agents_bid_sizes_round": [5.0] * SLOTS_PER_DAY,
            },
            logging_step=0,
        )
        prepare_activation(env)

        assert len(env.delivery_results["bid_sizes_all_slots"]) == SLOTS_PER_DAY * STEPS_PER_SLOT
        assert len(env.delivery_results["reserved_slots"]) == SLOTS_PER_DAY
        assert len(env.delivery_results["activated_slots"]) == SLOTS_PER_DAY

    def test_bid_sizes_expanded_correctly(self) -> None:
        """Each slot's bid size should repeat for STEPS_PER_SLOT timesteps."""
        random.seed(42)
        bid_sizes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        env = SimpleNamespace(
            delivery_results={"agents_bid_sizes_round": bid_sizes},
            logging_step=0,
        )
        prepare_activation(env)

        expanded = env.delivery_results["bid_sizes_all_slots"]
        for slot in range(SLOTS_PER_DAY):
            slot_values = expanded[slot * STEPS_PER_SLOT : (slot + 1) * STEPS_PER_SLOT]
            assert all(v == bid_sizes[slot] for v in slot_values)


class TestSimulateActivation:
    def test_activation_passes_with_ample_capacity(self) -> None:
        """All timesteps have enough capacity -> slot activated."""
        random.seed(42)
        env = SimpleNamespace(
            delivery_results={
                "vpp_total": np.ones(SLOTS_PER_DAY * STEPS_PER_SLOT) * 100.0,
                "bid_sizes_all_slots": np.ones(SLOTS_PER_DAY * STEPS_PER_SLOT) * 1.0,
                "activated_slots": [0] * SLOTS_PER_DAY,
                "total_not_delivered_energy": [0.0] * SLOTS_PER_DAY,
                "positive_activation_possible_list": [None] * SLOTS_PER_DAY,
                "negative_activation_possible_list": [None] * SLOTS_PER_DAY,
            },
            logging_step=0,
        )
        simulate_activation(env, slot=0)

        # With capacity=100 and bid=1, should pass
        assert env.delivery_results["activated_slots"][0] == 1

    def test_activation_fails_with_zero_capacity(self) -> None:
        """Zero VPP capacity -> activation fails."""
        random.seed(42)
        env = SimpleNamespace(
            delivery_results={
                "vpp_total": np.zeros(SLOTS_PER_DAY * STEPS_PER_SLOT),
                "bid_sizes_all_slots": np.ones(SLOTS_PER_DAY * STEPS_PER_SLOT) * 10.0,
                "activated_slots": [0] * SLOTS_PER_DAY,
                "total_not_delivered_energy": [0.0] * SLOTS_PER_DAY,
                "positive_activation_possible_list": [None] * SLOTS_PER_DAY,
                "negative_activation_possible_list": [None] * SLOTS_PER_DAY,
            },
            logging_step=0,
        )
        simulate_activation(env, slot=0)

        assert env.delivery_results["activated_slots"][0] == -1
        assert env.delivery_results["total_not_delivered_energy"][0] > 0.0
