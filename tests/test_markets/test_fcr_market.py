"""Tests for the FCR market implementation."""

import random

import numpy as np
import pytest

from vpp_bidding.domain.enums import MarketType, SlotStatus
from vpp_bidding.domain.models import Bid
from vpp_bidding.markets.base import MarketState
from vpp_bidding.markets.fcr.constants import SLOTS_PER_DAY, STEPS_PER_DAY
from vpp_bidding.markets.fcr.market import AuctionBid, FCRMarket


@pytest.fixture
def fcr_market() -> FCRMarket:
    return FCRMarket(
        price_scaler_min=0.0,
        price_scaler_max=500.0,
        size_scaler_min=0.0,
        size_scaler_max=50.0,
    )


@pytest.fixture
def sample_data(sample_fcr_prices: np.ndarray, sample_vpp_capacity: np.ndarray) -> dict:
    return {
        "fcr_prices": sample_fcr_prices,
        "vpp_capacity": sample_vpp_capacity,
        "order_book": [],
    }


class TestFCRMarketProperties:
    def test_name(self, fcr_market: FCRMarket) -> None:
        assert fcr_market.name == "FCR"

    def test_market_type(self, fcr_market: FCRMarket) -> None:
        assert fcr_market.market_type == MarketType.FCR

    def test_action_size(self, fcr_market: FCRMarket) -> None:
        assert fcr_market.action_size == SLOTS_PER_DAY * 2  # 12

    def test_observation_size(self, fcr_market: FCRMarket) -> None:
        expected = SLOTS_PER_DAY + STEPS_PER_DAY + 1  # 6 + 96 + 1 = 103
        assert fcr_market.observation_size == expected


class TestFCRMarketSimulate:
    def test_bid_size_zero_gives_not_participated(
        self,
        fcr_market: FCRMarket,
        sample_data: dict,
    ) -> None:
        """When all bid sizes are 0 -> all slots NOT_PARTICIPATED."""
        random.seed(42)
        actions = np.zeros(12, dtype=np.float32)
        # Prices can be anything
        actions[6:] = 50.0

        state = fcr_market.simulate(actions, sample_data, step=0)
        for status in state.slot_statuses:
            assert status == SlotStatus.NOT_PARTICIPATED

    def test_high_price_gives_lost(
        self,
        fcr_market: FCRMarket,
        sample_data: dict,
    ) -> None:
        """Bid price > settlement price -> slot LOST."""
        random.seed(42)
        actions = np.zeros(12, dtype=np.float32)
        # Set bid sizes to something > 0
        actions[:6] = 5.0
        # Set prices way above settlement prices (max in fixture is 150)
        actions[6:] = 9999.0

        state = fcr_market.simulate(actions, sample_data, step=0)
        for status in state.slot_statuses:
            assert status == SlotStatus.LOST

    def test_low_price_gives_won(
        self,
        fcr_market: FCRMarket,
        sample_data: dict,
    ) -> None:
        """Bid price <= settlement price with no order book -> slot WON."""
        random.seed(42)
        actions = np.zeros(12, dtype=np.float32)
        actions[:6] = 2.0  # small bid size
        actions[6:] = 1.0  # very low price, below all settlement prices

        state = fcr_market.simulate(actions, sample_data, step=0)
        for status in state.slot_statuses:
            assert status == SlotStatus.WON

    def test_bids_are_recorded(
        self,
        fcr_market: FCRMarket,
        sample_data: dict,
    ) -> None:
        random.seed(42)
        actions = np.zeros(12, dtype=np.float32)
        actions[:6] = 3.0
        actions[6:] = 50.0

        state = fcr_market.simulate(actions, sample_data, step=0)
        assert len(state.bids) == SLOTS_PER_DAY
        for bid in state.bids:
            assert isinstance(bid, Bid)
            assert bid.capacity_mw == pytest.approx(3.0)
            assert bid.price_eur_per_mw == pytest.approx(50.0)

    def test_settlement_prices_populated(
        self,
        fcr_market: FCRMarket,
        sample_data: dict,
    ) -> None:
        random.seed(42)
        actions = np.zeros(12, dtype=np.float32)
        state = fcr_market.simulate(actions, sample_data, step=0)
        assert len(state.settlement_prices) == SLOTS_PER_DAY

    def test_delivery_results_populated(
        self,
        fcr_market: FCRMarket,
        sample_data: dict,
    ) -> None:
        random.seed(42)
        actions = np.zeros(12, dtype=np.float32)
        actions[:6] = 2.0
        actions[6:] = 1.0  # cheap -> WON

        state = fcr_market.simulate(actions, sample_data, step=0)
        for slot in range(SLOTS_PER_DAY):
            assert slot in state.delivery_results
            assert "agent_bid_size" in state.delivery_results[slot]
            assert "slot_status" in state.delivery_results[slot]


class TestFCRMarketCapacityCommitment:
    def test_commitment_for_won_slots(self, fcr_market: FCRMarket) -> None:
        """Won slots should commit their capacity across the slot's timesteps."""
        bids = [Bid(slot=i, capacity_mw=5.0, price_eur_per_mw=100.0) for i in range(SLOTS_PER_DAY)]
        statuses = [
            SlotStatus.WON,
            SlotStatus.LOST,
            SlotStatus.WON,
            SlotStatus.NOT_PARTICIPATED,
            SlotStatus.WON,
            SlotStatus.LOST,
        ]

        state = MarketState(
            slot_statuses=statuses,
            bids=bids,
            settlement_prices=[100.0] * SLOTS_PER_DAY,
            delivery_results={},
        )

        commitment = fcr_market.get_capacity_commitment(state)
        assert commitment.shape == (STEPS_PER_DAY,)

        # Slot 0 (WON): steps 0-15 should be 5.0
        assert np.all(commitment[0:16] == pytest.approx(5.0))
        # Slot 1 (LOST): steps 16-31 should be 0.0
        assert np.all(commitment[16:32] == pytest.approx(0.0))
        # Slot 2 (WON): steps 32-47 should be 5.0
        assert np.all(commitment[32:48] == pytest.approx(5.0))
        # Slot 3 (NOT_PARTICIPATED): 0.0
        assert np.all(commitment[48:64] == pytest.approx(0.0))


class TestFCRMarketAuction:
    def test_auction_with_order_book(self) -> None:
        """Agent bid replaces expensive bids in the order book."""
        random.seed(42)
        market = FCRMarket()

        order_book_slot = [
            AuctionBid(price=200.0, size=10.0, country="DE"),
            AuctionBid(price=150.0, size=10.0, country="DE"),
            AuctionBid(price=100.0, size=10.0, country="DE"),
        ]

        bid = Bid(slot=0, capacity_mw=5.0, price_eur_per_mw=80.0)
        vpp_steps = np.ones(16, dtype=np.float32) * 20.0

        status, settlement, _delivery = market._simulate_auction(
            bid=bid,
            order_book=order_book_slot,
            historical_settlement_price=200.0,
            vpp_total_steps=vpp_steps,
        )

        assert status == SlotStatus.WON
        # Agent replaced 5MW of the 200 EUR bid, so remaining highest is still 200 (5MW left)
        assert settlement > 0

    def test_auction_with_dict_order_book(self) -> None:
        """Order book entries can be dicts instead of AuctionBid objects."""
        random.seed(42)
        market = FCRMarket()

        order_book_slot = [
            {"price": 200.0, "size": 10.0, "country": "DE"},
            {"price": 100.0, "size": 10.0, "country": "DE"},
        ]

        bid = Bid(slot=0, capacity_mw=3.0, price_eur_per_mw=80.0)
        vpp_steps = np.ones(16, dtype=np.float32) * 20.0

        status, _settlement, _delivery = market._simulate_auction(
            bid=bid,
            order_book=order_book_slot,
            historical_settlement_price=200.0,
            vpp_total_steps=vpp_steps,
        )

        assert status == SlotStatus.WON
