"""aFRR (automatic Frequency Restoration Reserve) market implementation."""

from typing import Any

import numpy as np

from vpp_bidding.domain.enums import MarketType
from vpp_bidding.markets.base import Financials, Market, MarketState, RewardBreakdown


class AFRRMarket(Market):
    """Automatic Frequency Restoration Reserve (aFRR) market
    (Sekundärregelleistung / Secondary Control Reserve).

    aFRR is a balancing product used by TSOs to restore system frequency after
    disturbances. It consists of two distinct revenue streams that the VPP can
    participate in. Single joint tender for all four German TSO areas
    (since 2018 harmonization).

    **aFRR Capacity (Power) -- reservation market:**
        - The TSO procures reserved balancing capacity through auctions.
        - Product structure: 4-hour blocks (similar to FCR), 6 products per day.
        - The VPP commits to holding a specified MW of capacity available for
          activation during each block.
        - Prices: capacity in EUR/MW per 4h block, activation energy in EUR/MWh.
        - Gate closure: D-1 10:45 CET.
        - Pay-as-cleared since 2020; pay-as-bid (discriminatory) before 2020.

    **aFRR Energy -- activation market:**
        - When the TSO activates aFRR, the reserved capacity must ramp within
          5 minutes (full activation within 5 min, compared to 30s for FCR).
          5-minute full activation requirement (vs. 30s for FCR) -- more suitable
          for hydro VPPs with slower ramp rates.
        - Activated energy is settled at the merit-order activation price.
        - The activation price is determined by a merit-order list of energy bids
          submitted alongside capacity bids.
        - Revenue = activated_energy_MWh * activation_price_EUR_per_MWh.

    **Cross-market interaction:**
        - Capacity reserved for aFRR cannot simultaneously be committed to FCR
          or other balancing products.
        - The VPP agent must allocate its total capacity budget across markets.
        - aFRR capacity prices are typically lower than FCR but activation
          revenue can be significant during high-imbalance periods.

    **VPP strategy considerations:**
        - Bid both capacity reservation price and energy activation price.
        - Balance guaranteed capacity revenue vs. uncertain activation revenue.
        - Consider correlation between aFRR activation probability and
          renewable generation forecast errors.
    """

    @property
    def name(self) -> str:
        return "afrr"

    @property
    def market_type(self) -> MarketType:
        return MarketType.AFRR_POWER

    @property
    def action_size(self) -> int:
        raise NotImplementedError(
            "AFRRMarket.action_size: action space not yet defined. "
            "Expected: 6 capacity bid sizes + 6 capacity bid prices + "
            "6 energy bid prices = 18 actions (for 4h block structure)."
        )

    @property
    def observation_size(self) -> int:
        raise NotImplementedError(
            "AFRRMarket.observation_size: observation vector design not yet finalized. "
            "Expected components: VPP capacity forecast, historical aFRR capacity prices, "
            "historical activation volumes, system imbalance forecasts."
        )

    def get_observation(self, data: dict[str, Any], step: int) -> np.ndarray:
        raise NotImplementedError(
            "AFRRMarket.get_observation: aFRR observation builder not yet implemented. "
            "Should include capacity forecast, lagged aFRR prices, activation probability "
            "estimates, and system balance indicators."
        )

    def simulate(
        self,
        actions: np.ndarray,
        data: dict[str, Any],
        step: int,
    ) -> MarketState:
        raise NotImplementedError(
            "AFRRMarket.simulate: aFRR market simulation not yet implemented. "
            "Should simulate both capacity auction clearing (pay-as-cleared) and "
            "stochastic energy activation using historical activation patterns."
        )

    def calculate_reward(self, state: MarketState) -> RewardBreakdown:
        raise NotImplementedError(
            "AFRRMarket.calculate_reward: reward shaping not yet implemented. "
            "Should combine capacity reservation reward, energy activation reward, "
            "and penalties for failing to deliver activated energy."
        )

    def calculate_financials(self, state: MarketState) -> Financials:
        raise NotImplementedError(
            "AFRRMarket.calculate_financials: P&L calculation not yet implemented. "
            "Revenue = capacity_price * reserved_MW + activation_price * activated_MWh. "
            "Penalties for non-delivery of activated energy."
        )

    def get_capacity_commitment(self, state: MarketState) -> np.ndarray:
        raise NotImplementedError(
            "AFRRMarket.get_capacity_commitment: capacity tracking not yet implemented. "
            "Should return a 96-element array (15-min resolution) of committed MW "
            "based on won 4h capacity reservation blocks."
        )
