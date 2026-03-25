"""Shared constants used across multiple market implementations."""

from __future__ import annotations

# General time constants
HOURS_PER_DAY: int = 24
QUARTER_HOURS_PER_DAY: int = 96

# Default scaler bounds
DEFAULT_PRICE_MIN: float = 0.0
DEFAULT_PRICE_MAX: float = 1000.0
DEFAULT_SIZE_MIN: float = 0.0
DEFAULT_SIZE_MAX: float = 100.0
