"""Configuration management with TOML loading."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class MarketConfig:
    enabled: list[str] = field(default_factory=lambda: ["fcr"])


@dataclass(frozen=True)
class FCRConfig:
    slots_per_day: int = 6
    steps_per_slot: int = 16


@dataclass(frozen=True)
class TimeConfig:
    hist_window_size: int = 1
    forecast_window_size: int = 1
    first_slot_date_start: str = "2020-07-02 22:00:00+00:00"
    last_slot_date_end: str = "2022-05-31 21:45:00+00:00"


@dataclass(frozen=True)
class DataConfig:
    renewables: str = "data/clean/renewables.csv"
    tenders: str = "data/clean/tenders_all.csv"
    market_results: str = "data/clean/market_results.csv"
    bids: str = "data/clean/bids_all.csv"
    time_features: str = "data/clean/time_features.csv"
    test_set: str = "data/clean/test_set_70days.csv"
    market_prices: str = "data/clean/wholesale_market_prices.csv"


@dataclass(frozen=True)
class AssetEntry:
    type: str
    max_capacity_mw: float
    quantity: int
    max_fcr_capacity_share: float
    asset_column_names: list[str]


@dataclass(frozen=True)
class WandbConfig:
    project: str = "vpp-bidding"
    entity: str | None = None
    mode: str = "online"


@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int = 11140
    seed: int = 42


@dataclass(frozen=True)
class AppConfig:
    markets: MarketConfig = field(default_factory=MarketConfig)
    fcr: FCRConfig = field(default_factory=FCRConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    assets: dict[str, list[AssetEntry]] = field(default_factory=dict)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _parse_assets(raw_assets: dict[str, Any]) -> dict[str, list[AssetEntry]]:
    """Parse the assets section from raw TOML data."""
    result: dict[str, list[AssetEntry]] = {}
    for group_name, entries in raw_assets.items():
        if isinstance(entries, list):
            result[group_name] = [
                AssetEntry(
                    type=e["type"],
                    max_capacity_mw=e["max_capacity_mw"],
                    quantity=e["quantity"],
                    max_fcr_capacity_share=e["max_fcr_capacity_share"],
                    asset_column_names=e.get("asset_column_names", []),
                )
                for e in entries
            ]
    return result


def load_config(path: Path) -> AppConfig:
    """Load configuration from TOML file.

    Args:
        path: Path to the TOML configuration file.

    Returns:
        Parsed AppConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        tomllib.TOMLDecodeError: If the file is not valid TOML.
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    markets = MarketConfig(**raw.get("markets", {}))
    fcr = FCRConfig(**raw.get("fcr", {}))
    time = TimeConfig(**raw.get("time", {}))
    data = DataConfig(**raw.get("data", {}))
    assets = _parse_assets(raw.get("assets", {}))
    wandb_cfg = WandbConfig(**raw.get("wandb", {}))
    training = TrainingConfig(**raw.get("training", {}))

    return AppConfig(
        markets=markets,
        fcr=fcr,
        time=time,
        data=data,
        assets=assets,
        wandb=wandb_cfg,
        training=training,
    )
