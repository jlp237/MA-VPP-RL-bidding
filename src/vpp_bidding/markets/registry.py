"""Registry that maps MarketType enums to concrete Market subclasses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from vpp_bidding.domain.enums import MarketType

if TYPE_CHECKING:
    from vpp_bidding.markets.base import Market


class MarketRegistry:
    """Central registry for market implementations.

    Uses a class-level ``_registry`` dict as a singleton so that all callers
    share the same mapping.  Default markets are registered at import time
    via :func:`_register_defaults`.
    """

    # Intentionally a class-level mutable: acts as a process-wide singleton
    # so that ``_register_defaults()`` (called at import time) and later
    # ``register()`` calls all write to the same mapping.
    _registry: ClassVar[dict[MarketType, type[Market]]] = {}

    @classmethod
    def register(cls, market_type: MarketType, market_class: type[Market]) -> None:
        cls._registry[market_type] = market_class

    @classmethod
    def get(cls, market_type: MarketType) -> type[Market]:
        if market_type not in cls._registry:
            raise KeyError(f"No market registered for {market_type!r}")
        return cls._registry[market_type]

    @classmethod
    def clear(cls) -> None:
        """Remove all registered markets.  Intended for test isolation."""
        cls._registry.clear()

    @classmethod
    def build(
        cls,
        enabled_markets: list[MarketType],
        configs: dict[MarketType, dict[str, Any]],
    ) -> list[Market]:
        """Instantiate all enabled markets with their configs."""
        markets: list[Market] = []
        for mt in enabled_markets:
            market_class = cls.get(mt)
            config = configs.get(mt, {})
            markets.append(market_class(**config))
        return markets


def _register_defaults() -> None:
    """Register built-in market implementations. Called at import time."""
    from vpp_bidding.markets.fcr.market import FCRMarket

    MarketRegistry.register(MarketType.FCR, FCRMarket)


_register_defaults()
