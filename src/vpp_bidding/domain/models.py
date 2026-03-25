"""Domain models for VPP bidding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Bid:
    """A single market bid.

    Attributes:
        slot: Time slot index.
        capacity_mw: Bid capacity in MW.
        price_eur_per_mw: Price in EUR/MW. For FCR: capacity price per 4h block.
            For energy markets: use EUR/MWh instead.
    """

    slot: int
    capacity_mw: float
    price_eur_per_mw: float


@dataclass(frozen=True)
class AssetConfig:
    asset_type: str
    plant_type: str
    max_capacity_mw: float
    quantity: int
    max_fcr_share: float
    asset_column_names: list[str]


@dataclass(frozen=True)
class VPPConfig:
    assets: list[AssetConfig]
