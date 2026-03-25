"""Tests for the abstract Market base class."""

import numpy as np
import pytest

from vpp_bidding.domain.enums import MarketType
from vpp_bidding.markets.base import (
    Financials,
    Market,
    MarketState,
    RewardBreakdown,
)


class TestMarketABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            Market()  # type: ignore[abstract]

    def test_subclass_must_implement_all_abstract_methods(self) -> None:
        """A partial implementation should still raise TypeError."""

        class PartialMarket(Market):
            @property
            def name(self) -> str:
                return "partial"

        with pytest.raises(TypeError):
            PartialMarket()  # type: ignore[abstract]

    def test_complete_subclass_can_be_instantiated(self) -> None:
        """A fully-implemented subclass should be instantiable."""

        class FullMarket(Market):
            @property
            def name(self) -> str:
                return "full"

            @property
            def market_type(self) -> MarketType:
                return MarketType.FCR

            @property
            def action_size(self) -> int:
                return 12

            @property
            def observation_size(self) -> int:
                return 103

            def get_observation(self, data: dict, step: int) -> np.ndarray:
                return np.zeros(self.observation_size)

            def simulate(self, actions: np.ndarray, data: dict, step: int) -> MarketState:
                return MarketState()

            def calculate_reward(self, state: MarketState) -> RewardBreakdown:
                return RewardBreakdown(0.0, [], [], [], [])

            def calculate_financials(self, state: MarketState) -> Financials:
                return Financials(0.0, 0.0, 0.0, [])

            def get_capacity_commitment(self, state: MarketState) -> np.ndarray:
                return np.zeros(96)

        market = FullMarket()
        assert market.name == "full"
        assert market.action_size == 12


class TestMarketState:
    def test_default_empty(self) -> None:
        state = MarketState()
        assert state.slot_statuses == []
        assert state.bids == []
        assert state.settlement_prices == []
        assert state.delivery_results == {}


class TestRewardBreakdown:
    def test_frozen(self) -> None:
        rb = RewardBreakdown(
            total_reward=1.0,
            slot_rewards=[0.5, 0.5],
            auction_rewards=[1.0, 1.0],
            reservation_rewards=[1.0, 1.0],
            activation_rewards=[1.0, 1.0],
        )
        assert rb.total_reward == 1.0
        assert len(rb.slot_rewards) == 2


class TestFinancials:
    def test_creation(self) -> None:
        fin = Financials(
            revenue=500.0,
            penalties=50.0,
            profit=450.0,
            settlement_prices=[100.0, 120.0],
        )
        assert fin.profit == 450.0
