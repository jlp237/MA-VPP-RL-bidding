"""Tests for domain models and enums."""

import dataclasses

import pytest

from vpp_bidding.domain.enums import (
    Algorithm,
    EnvMode,
    MarketType,
    RenderMode,
    SlotStatus,
)
from vpp_bidding.domain.models import AssetConfig, Bid, VPPConfig

# ---------------------------------------------------------------------------
# Bid
# ---------------------------------------------------------------------------


class TestBid:
    def test_creation(self) -> None:
        bid = Bid(slot=0, capacity_mw=5.0, price_eur_per_mw=100.0)
        assert bid.slot == 0
        assert bid.capacity_mw == 5.0
        assert bid.price_eur_per_mw == 100.0

    def test_frozen_immutability(self) -> None:
        bid = Bid(slot=0, capacity_mw=5.0, price_eur_per_mw=100.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            bid.capacity_mw = 10.0  # type: ignore[misc]

    def test_equality(self) -> None:
        bid1 = Bid(slot=0, capacity_mw=5.0, price_eur_per_mw=100.0)
        bid2 = Bid(slot=0, capacity_mw=5.0, price_eur_per_mw=100.0)
        assert bid1 == bid2

    def test_inequality(self) -> None:
        bid1 = Bid(slot=0, capacity_mw=5.0, price_eur_per_mw=100.0)
        bid2 = Bid(slot=1, capacity_mw=5.0, price_eur_per_mw=100.0)
        assert bid1 != bid2


# ---------------------------------------------------------------------------
# AssetConfig
# ---------------------------------------------------------------------------


class TestAssetConfig:
    def test_creation(self) -> None:
        ac = AssetConfig(
            asset_type="wind",
            plant_type="onshore",
            max_capacity_mw=10.0,
            quantity=3,
            max_fcr_share=0.5,
            asset_column_names=["wind_1", "wind_2", "wind_3"],
        )
        assert ac.asset_type == "wind"
        assert ac.quantity == 3
        assert len(ac.asset_column_names) == 3

    def test_frozen_immutability(self) -> None:
        ac = AssetConfig(
            asset_type="wind",
            plant_type="onshore",
            max_capacity_mw=10.0,
            quantity=1,
            max_fcr_share=0.5,
            asset_column_names=["wind_1"],
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ac.max_capacity_mw = 20.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# VPPConfig
# ---------------------------------------------------------------------------


class TestVPPConfig:
    def test_creation(self, sample_asset_config: AssetConfig) -> None:
        cfg = VPPConfig(assets=[sample_asset_config])
        assert len(cfg.assets) == 1
        assert cfg.assets[0].asset_type == "wind"

    def test_frozen_immutability(self, sample_asset_config: AssetConfig) -> None:
        cfg = VPPConfig(assets=[sample_asset_config])
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.assets = []  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestSlotStatus:
    def test_values(self) -> None:
        assert SlotStatus.LOST == -1
        assert SlotStatus.NOT_PARTICIPATED == 0
        assert SlotStatus.WON == 1

    def test_is_int(self) -> None:
        assert isinstance(SlotStatus.WON, int)


class TestEnvMode:
    def test_values(self) -> None:
        assert EnvMode.TRAINING.value == "training"
        assert EnvMode.EVAL.value == "eval"
        assert EnvMode.TEST.value == "test"


class TestRenderMode:
    def test_values(self) -> None:
        assert RenderMode.HUMAN.value == "human"
        assert RenderMode.FAST_TRAINING.value == "fast_training"


class TestMarketType:
    def test_fcr(self) -> None:
        assert MarketType.FCR.value == "fcr"

    def test_all_values(self) -> None:
        expected = {
            "day_ahead",
            "fcr",
            "afrr_power",
            "afrr_energy",
            "mfrr",
            "intraday",
            "imbalance",
        }
        actual = {mt.value for mt in MarketType}
        assert actual == expected


class TestAlgorithm:
    def test_all_algorithms(self) -> None:
        expected = {"PPO", "A2C", "TRPO", "RecurrentPPO", "SAC", "DDPG", "TD3", "TQC"}
        actual = {a.value for a in Algorithm}
        assert actual == expected
