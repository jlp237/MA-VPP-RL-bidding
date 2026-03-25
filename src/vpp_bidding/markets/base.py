"""Abstract base class for market implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vpp_bidding.domain.enums import MarketType, SlotStatus
from vpp_bidding.domain.models import Bid


@dataclass
class MarketState:
    """Mutable state tracked during a market simulation day."""

    slot_statuses: list[SlotStatus] = field(default_factory=list)
    bids: list[Bid] = field(default_factory=list)
    settlement_prices: list[float] = field(default_factory=list)
    delivery_results: dict[Any, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardBreakdown:
    """Decomposition of the reward signal for a single day."""

    total_reward: float
    slot_rewards: list[float]
    auction_rewards: list[float]
    reservation_rewards: list[float]
    activation_rewards: list[float]


@dataclass(frozen=True)
class Financials:
    """Financial outcome for a single day."""

    revenue: float
    penalties: float
    profit: float
    settlement_prices: list[float]


class Market(ABC):
    """Abstract base class that all market implementations must satisfy."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def market_type(self) -> MarketType: ...

    @property
    @abstractmethod
    def action_size(self) -> int: ...

    @property
    @abstractmethod
    def observation_size(self) -> int: ...

    @abstractmethod
    def get_observation(self, data: dict[str, Any], step: int) -> np.ndarray:
        """Build the observation vector for the current step."""
        ...

    @abstractmethod
    def simulate(
        self,
        actions: np.ndarray,
        data: dict[str, Any],
        step: int,
    ) -> MarketState:
        """Run market clearing and delivery simulation for one day."""
        ...

    @abstractmethod
    def calculate_reward(self, state: MarketState) -> RewardBreakdown:
        """Compute the shaped reward from the market state."""
        ...

    @abstractmethod
    def calculate_financials(self, state: MarketState) -> Financials:
        """Compute actual financial P&L from the market state."""
        ...

    @abstractmethod
    def get_capacity_commitment(self, state: MarketState) -> np.ndarray:
        """Return the capacity committed per timestep (for cross-market constraints)."""
        ...
