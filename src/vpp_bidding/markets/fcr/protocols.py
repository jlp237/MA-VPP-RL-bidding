"""Structural typing protocols for FCR module dependencies.

These protocols capture the env interface that the FCR reservation,
activation, and reward modules rely on, so that concrete env classes
need not be imported (avoiding circular dependencies) while still
providing meaningful type checking.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class FCREnv(Protocol):
    """Minimal env interface used by reservation and activation modules."""

    delivery_results: dict[str, Any]
    logging_step: int


@runtime_checkable
class FCRRewardEnv(FCREnv, Protocol):
    """Extended env interface used by the reward module.

    Adds attributes needed for penalty/reward calculation that are not
    required by the lower-level reservation/activation checks.
    """

    current_daily_mean_market_price: float
    price_scaler: Any
    size_scaler: Any
