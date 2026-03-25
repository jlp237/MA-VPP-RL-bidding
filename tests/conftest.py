"""Shared fixtures for the VPP bidding test suite."""

import numpy as np
import pandas as pd
import pytest

from vpp_bidding.domain.models import AssetConfig, Bid, VPPConfig


@pytest.fixture
def sample_bid() -> Bid:
    return Bid(slot=0, capacity_mw=5.0, price_eur_per_mw=100.0)


@pytest.fixture
def sample_asset_config() -> AssetConfig:
    return AssetConfig(
        asset_type="wind",
        plant_type="onshore",
        max_capacity_mw=10.0,
        quantity=2,
        max_fcr_share=0.5,
        asset_column_names=["wind_1", "wind_2"],
    )


@pytest.fixture
def sample_vpp_config(sample_asset_config: AssetConfig) -> VPPConfig:
    return VPPConfig(assets=[sample_asset_config])


@pytest.fixture
def sample_renewables_df() -> pd.DataFrame:
    """Minimal renewables DataFrame with 96 quarter-hour rows (one day)."""
    index = pd.date_range("2021-01-01", periods=96, freq="15min", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "wind_1": rng.uniform(0.3, 0.9, size=96),
            "wind_2": rng.uniform(0.3, 0.9, size=96),
        },
        index=index,
    )


@pytest.fixture
def sample_vpp_capacity() -> np.ndarray:
    """96-element VPP capacity array (MW) for one day."""
    rng = np.random.default_rng(42)
    return rng.uniform(5.0, 15.0, size=96).astype(np.float32)


@pytest.fixture
def sample_fcr_prices() -> np.ndarray:
    """6-element settlement price array, one per FCR slot."""
    return np.array([100.0, 120.0, 80.0, 150.0, 90.0, 110.0], dtype=np.float32)


@pytest.fixture
def sample_delivery_results_won(sample_vpp_capacity: np.ndarray) -> dict:
    """Delivery results dict for 6 WON slots with ample VPP capacity."""
    from vpp_bidding.domain.enums import SlotStatus

    results: dict = {}
    for slot in range(6):
        start = slot * 16
        end = start + 16
        results[slot] = {
            "agent_bid_size": 3.0,
            "agent_bid_price": 80.0,
            "vpp_total_steps": sample_vpp_capacity[start:end],
            "slot_status": SlotStatus.WON,
            "settlement_price": 100.0,
        }
    return results


@pytest.fixture
def sample_delivery_results_lost() -> dict:
    """Delivery results dict for 6 LOST slots."""
    from vpp_bidding.domain.enums import SlotStatus

    results: dict = {}
    for slot in range(6):
        results[slot] = {
            "agent_bid_size": 5.0,
            "agent_bid_price": 200.0,
            "vpp_total_steps": np.ones(16, dtype=np.float32) * 10.0,
            "slot_status": SlotStatus.LOST,
            "settlement_price": 100.0,
        }
    return results
