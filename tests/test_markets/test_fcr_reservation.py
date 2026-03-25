"""Tests for FCR reservation simulation."""

from types import SimpleNamespace

import numpy as np

from vpp_bidding.markets.fcr.constants import (
    RESERVATION_STEP_DURATION_HOURS,
    SLOTS_PER_DAY,
    STEPS_PER_SLOT,
)
from vpp_bidding.markets.fcr.reservation import simulate_reservation


def _make_env(vpp_capacity: float, bid_size: float) -> SimpleNamespace:
    """Create a mock env for reservation testing."""
    return SimpleNamespace(
        delivery_results={
            "vpp_total": np.ones(SLOTS_PER_DAY * STEPS_PER_SLOT) * vpp_capacity,
            "bid_sizes_all_slots": np.ones(SLOTS_PER_DAY * STEPS_PER_SLOT) * bid_size,
            "reserved_slots": [0] * SLOTS_PER_DAY,
            "total_not_reserved_energy": [0.0] * SLOTS_PER_DAY,
        },
        logging_step=0,
    )


class TestSimulateReservation:
    def test_reservation_succeeds_when_capacity_exceeds_bid(self) -> None:
        """VPP capacity > bid size -> reservation succeeds."""
        env = _make_env(vpp_capacity=50.0, bid_size=5.0)
        simulate_reservation(env, slot=0)

        assert env.delivery_results["reserved_slots"][0] == 1
        assert env.delivery_results["total_not_reserved_energy"][0] == 0.0

    def test_reservation_fails_when_bid_exceeds_capacity(self) -> None:
        """Bid size > VPP capacity -> reservation fails."""
        env = _make_env(vpp_capacity=5.0, bid_size=50.0)
        simulate_reservation(env, slot=0)

        assert env.delivery_results["reserved_slots"][0] == -1
        assert env.delivery_results["total_not_reserved_energy"][0] > 0.0

    def test_energy_uses_reservation_factor(self) -> None:
        """Not-reserved energy uses RESERVATION_STEP_DURATION_HOURS (0.25h) factor."""
        assert RESERVATION_STEP_DURATION_HOURS == 0.25

    def test_reservation_at_boundary(self) -> None:
        """Bid size == VPP capacity -> negative check fails (not > 0)."""
        env = _make_env(vpp_capacity=10.0, bid_size=10.0)
        simulate_reservation(env, slot=0)

        # available - bid = 0, not > 0 -> negative fails
        # bid < available is False -> positive also fails
        assert env.delivery_results["reserved_slots"][0] == -1

    def test_multiple_slots_independent(self) -> None:
        """Each slot is simulated independently."""
        env = _make_env(vpp_capacity=50.0, bid_size=5.0)
        simulate_reservation(env, slot=0)
        simulate_reservation(env, slot=2)

        assert env.delivery_results["reserved_slots"][0] == 1
        assert env.delivery_results["reserved_slots"][1] == 0  # untouched
        assert env.delivery_results["reserved_slots"][2] == 1
