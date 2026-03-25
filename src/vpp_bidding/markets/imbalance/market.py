"""Imbalance settlement market implementation."""

from typing import Any

import numpy as np

from vpp_bidding.domain.enums import MarketType
from vpp_bidding.markets.base import Financials, Market, MarketState, RewardBreakdown


class ImbalanceMarket(Market):
    """Real-time imbalance settlement price exposure
    (Ausgleichsenergiepreis / reBAP since 2021).

    The imbalance market is not a market in the traditional sense -- participants
    do not actively bid into it. Instead, it represents the financial settlement
    of deviations between a Balance Responsible Party's (BRP) scheduled position
    and its actual metered generation/consumption in real time.

    Settlement at the imbalance price (reBAP = regelzonenübergreifender
    einheitlicher Bilanzausgleichsenergiepreis).

    **Settlement mechanism:**
        - After delivery, the TSO calculates each BRP's imbalance as:
          imbalance = actual_metered - scheduled_position (DA + ID trades).
        - Positive imbalance (surplus): BRP injected more than scheduled,
          settled at the imbalance price (may be low or even negative).
        - Negative imbalance (shortage): BRP delivered less than scheduled,
          settled at the imbalance price (may be very high during scarcity).
        - The imbalance price is derived from the marginal cost of balancing
          energy activated by the TSO (aFRR + mFRR activation costs).
        - Prices in EUR/MWh.

    **Price characteristics:**
        - The imbalance price is highly volatile and unpredictable.
        - During normal conditions, it is close to the day-ahead price.
        - During system stress, it can spike to thousands of EUR/MWh
          (or go deeply negative during surplus conditions).
        - Single price for long and short positions since 2021 harmonization
          (previously separate long/short prices). In Germany, the single
          imbalance price (reBAP) applies to both long and short positions.

    **VPP relevance:**
        - The imbalance market represents the residual risk after all
          forward markets (DA, ID) have closed.
        - A VPP with good renewable forecasting can minimize imbalance
          exposure by trading accurately in DA and ID markets.
        - Deliberate imbalance exposure (passive balancing) can be profitable
          if the agent can predict the imbalance price direction, but this
          is risky and may face regulatory scrutiny.
        - The imbalance cost/revenue is the final P&L adjustment applied
          after all other market settlements.

    **Modeling approach:**
        - No active bidding: the imbalance "action" is implicit in the
          deviation between the VPP's total scheduled position and its
          actual output.
        - The environment calculates imbalance automatically from:
          actual_generation - (DA_schedule + ID_net_trades).
        - Imbalance settlement prices come from historical reBAP data.

    **Cross-market interaction:**
        - Imbalance exposure is the residual of all other market positions.
        - Better DA/ID trading reduces imbalance volume and risk.
        - The imbalance price signal provides feedback on the quality of
          the agent's overall portfolio management.
    """

    @property
    def name(self) -> str:
        return "imbalance"

    @property
    def market_type(self) -> MarketType:
        return MarketType.IMBALANCE

    @property
    def action_size(self) -> int:
        """No active bidding in the imbalance market; exposure is implicit."""
        return 0

    @property
    def observation_size(self) -> int:
        raise NotImplementedError(
            "ImbalanceMarket.observation_size: observation design not yet finalized. "
            "Expected components: current system imbalance indicator, historical "
            "imbalance prices, net scheduled position from other markets."
        )

    def get_observation(self, data: dict[str, Any], step: int) -> np.ndarray:
        raise NotImplementedError(
            "ImbalanceMarket.get_observation: imbalance observation builder not yet "
            "implemented. Should include historical imbalance prices (reBAP), system "
            "balance indicator, and the VPP's net scheduled position."
        )

    def simulate(
        self,
        actions: np.ndarray,
        data: dict[str, Any],
        step: int,
    ) -> MarketState:
        raise NotImplementedError(
            "ImbalanceMarket.simulate: imbalance settlement not yet implemented. "
            "Should calculate the deviation between actual VPP output (from VPP "
            "simulation) and the sum of all forward market commitments, then apply "
            "the historical imbalance settlement price to compute costs/revenue."
        )

    def calculate_reward(self, state: MarketState) -> RewardBreakdown:
        raise NotImplementedError(
            "ImbalanceMarket.calculate_reward: reward shaping not yet implemented. "
            "Should penalize large imbalance volumes (encouraging accurate scheduling) "
            "while reflecting actual settlement revenue/cost."
        )

    def calculate_financials(self, state: MarketState) -> Financials:
        raise NotImplementedError(
            "ImbalanceMarket.calculate_financials: P&L calculation not yet implemented. "
            "Settlement = imbalance_volume_MWh * imbalance_price_EUR_per_MWh. "
            "Positive for surplus when imbalance price > 0, negative for shortage."
        )

    def get_capacity_commitment(self, state: MarketState) -> np.ndarray:
        raise NotImplementedError(
            "ImbalanceMarket.get_capacity_commitment: not applicable for imbalance. "
            "The imbalance market does not involve capacity commitment. Returns zeros."
        )
