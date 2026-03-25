"""mFRR (manual Frequency Restoration Reserve) market implementation."""

from typing import Any

import numpy as np

from vpp_bidding.domain.enums import MarketType
from vpp_bidding.markets.base import Financials, Market, MarketState, RewardBreakdown


class MFRRMarket(Market):
    """Manual Frequency Restoration Reserve (mFRR) market
    (Minutenreserve / Minute Reserve).

    mFRR is a balancing product for manual frequency restoration, activated by
    TSOs when automatic reserves (FCR and aFRR) are insufficient or need to be
    relieved. It is the slowest-responding balancing product in the activation
    cascade: FCR (30s) -> aFRR (5min) -> mFRR (12.5-15min).

    Gate closure: D-1 11:30 CET.

    **Product structure:**
        - Capacity is procured in 4-hour blocks (6 blocks per day), similar
          to FCR and aFRR capacity auctions.
        - Separate auctions for positive (upward) and negative (downward) mFRR.
        - Positive mFRR: provider must increase generation or reduce consumption.
        - Negative mFRR: provider must decrease generation or increase consumption.

    **Activation mechanism:**
        - 12.5-15 minute full activation requirement.
        - Activation is manual (phone call or electronic signal from TSO).
        - Activated energy is settled at the marginal activation price (merit-order).
        - Activation duration is typically 15 minutes per call, but can be extended.

    **Revenue streams:**
        - Prices: capacity in EUR/MW, activation energy in EUR/MWh.
        - Capacity reservation payment (EUR/MW per 4h block) for holding capacity
          available, regardless of whether activation occurs.
        - Energy activation payment (EUR/MWh) for energy actually delivered
          when the TSO calls for mFRR activation.

    **VPP relevance:**
        - mFRR has the lowest capacity prices among balancing products but also
          the lowest activation probability, making it a lower-risk, lower-reward
          option for the VPP's capacity allocation.
        - The slower ramp rate (12.5-15 min) is easier for most VPP assets to
          deliver compared to FCR's 30-second requirement.
        - Often used as a "residual" market: allocate leftover capacity after
          committing to higher-value FCR and aFRR products.

    **Cross-market constraints:**
        - Capacity reserved for mFRR cannot be simultaneously committed to
          FCR, aFRR, or day-ahead market.
        - The VPP must ensure total commitments across all markets do not
          exceed available generation capacity at any point in time.
    """

    @property
    def name(self) -> str:
        return "mfrr"

    @property
    def market_type(self) -> MarketType:
        return MarketType.MFRR

    @property
    def action_size(self) -> int:
        raise NotImplementedError(
            "MFRRMarket.action_size: action space not yet defined. "
            "Expected: 6 capacity bid sizes + 6 capacity bid prices + "
            "6 energy bid prices = 18 actions (for 4h block structure)."
        )

    @property
    def observation_size(self) -> int:
        raise NotImplementedError(
            "MFRRMarket.observation_size: observation vector design not yet finalized. "
            "Expected components: VPP capacity forecast, historical mFRR prices, "
            "historical activation frequency, system imbalance forecasts."
        )

    def get_observation(self, data: dict[str, Any], step: int) -> np.ndarray:
        raise NotImplementedError(
            "MFRRMarket.get_observation: mFRR observation builder not yet implemented. "
            "Should include capacity forecast, lagged mFRR capacity and energy prices, "
            "and activation probability indicators."
        )

    def simulate(
        self,
        actions: np.ndarray,
        data: dict[str, Any],
        step: int,
    ) -> MarketState:
        raise NotImplementedError(
            "MFRRMarket.simulate: mFRR market simulation not yet implemented. "
            "Should simulate capacity auction clearing and stochastic activation "
            "using historical mFRR activation patterns and durations."
        )

    def calculate_reward(self, state: MarketState) -> RewardBreakdown:
        raise NotImplementedError(
            "MFRRMarket.calculate_reward: reward shaping not yet implemented. "
            "Should combine capacity reservation reward, energy activation reward, "
            "and penalties for non-delivery."
        )

    def calculate_financials(self, state: MarketState) -> Financials:
        raise NotImplementedError(
            "MFRRMarket.calculate_financials: P&L calculation not yet implemented. "
            "Revenue = capacity_price * reserved_MW + activation_price * activated_MWh. "
            "Penalties for failure to deliver within 12.5-15 minute ramp requirement."
        )

    def get_capacity_commitment(self, state: MarketState) -> np.ndarray:
        raise NotImplementedError(
            "MFRRMarket.get_capacity_commitment: capacity tracking not yet implemented. "
            "Should return a 96-element array (15-min resolution) of committed MW "
            "based on won 4h capacity reservation blocks."
        )
