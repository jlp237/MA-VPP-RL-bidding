"""FCR market implementation.

Wraps auction clearing, activation/reservation simulation, reward
computation and observation construction behind the abstract ``Market``
interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from vpp_bidding.domain.enums import MarketType, SlotStatus
from vpp_bidding.domain.models import Bid
from vpp_bidding.markets.base import (
    Financials,
    Market,
    MarketState,
    RewardBreakdown,
)
from vpp_bidding.markets.fcr.constants import (
    SLOTS_PER_DAY,
    STEPS_PER_DAY,
    STEPS_PER_SLOT,
)
from vpp_bidding.markets.fcr.observation import get_fcr_observation

logger = logging.getLogger(__name__)

# Action layout: 6 sizes + 6 prices
_ACTION_SIZE = SLOTS_PER_DAY * 2  # 12


@dataclass
class AuctionBid:
    """A single bid in the FCR auction order book."""

    price: float
    size: float
    country: str
    indivisible: bool = False


class FCRMarket(Market):
    """Concrete ``Market`` implementation for Frequency Containment Reserve."""

    def __init__(
        self,
        price_scaler_min: float = 0.0,
        price_scaler_max: float = 1000.0,
        size_scaler_min: float = 0.0,
        size_scaler_max: float = 100.0,
        **kwargs: Any,
    ) -> None:
        self._price_scaler_min = price_scaler_min
        self._price_scaler_max = price_scaler_max
        self._size_scaler_min = size_scaler_min
        self._size_scaler_max = size_scaler_max

    # ------------------------------------------------------------------
    # Market interface properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "FCR"

    @property
    def market_type(self) -> MarketType:
        return MarketType.FCR

    @property
    def action_size(self) -> int:
        return _ACTION_SIZE

    @property
    def observation_size(self) -> int:
        # prices (6) + capacity (96) + time (1)
        return SLOTS_PER_DAY + STEPS_PER_DAY + 1

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_observation(self, data: dict[str, Any], step: int) -> np.ndarray:
        return get_fcr_observation(
            fcr_prices=data["fcr_prices"],
            vpp_capacity=data["vpp_capacity"],
            current_step=step,
            price_scaler_min=self._price_scaler_min,
            price_scaler_max=self._price_scaler_max,
            size_scaler_min=self._size_scaler_min,
            size_scaler_max=self._size_scaler_max,
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        actions: np.ndarray,
        data: dict[str, Any],
        step: int,
    ) -> MarketState:
        """Run FCR market clearing and delivery simulation for one day.

        ``actions`` is expected to be a 1-D array of length 12:
        indices 0-5 are bid sizes (MW) and indices 6-11 are bid prices
        (EUR/MW), one per slot.
        """
        sizes = actions[:SLOTS_PER_DAY]
        prices = actions[SLOTS_PER_DAY:]

        bids: list[Bid] = []
        slot_statuses: list[SlotStatus] = []
        settlement_prices: list[float] = []
        delivery_results: dict[int, dict[str, Any]] = {}

        market_order_book: list[list[AuctionBid]] = data.get("order_book", [])

        for slot in range(SLOTS_PER_DAY):
            bid = Bid(
                slot=slot,
                capacity_mw=float(sizes[slot]),
                price_eur_per_mw=float(prices[slot]),
            )
            bids.append(bid)

            slot_order_book = market_order_book[slot] if slot < len(market_order_book) else []
            historical_settlement = data["fcr_prices"][slot]

            status, settlement, slot_delivery = self._simulate_auction(
                bid=bid,
                order_book=slot_order_book,
                historical_settlement_price=historical_settlement,
                vpp_total_steps=data["vpp_capacity"][
                    slot * STEPS_PER_SLOT : (slot + 1) * STEPS_PER_SLOT
                ],
            )

            slot_statuses.append(status)
            settlement_prices.append(settlement)
            delivery_results[slot] = slot_delivery

        state = MarketState(
            slot_statuses=slot_statuses,
            bids=bids,
            settlement_prices=settlement_prices,
            delivery_results=delivery_results,
        )
        return state

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def calculate_reward(self, state: MarketState) -> RewardBreakdown:
        # Reward calculation currently uses the legacy env-based bridge.
        # This method will be implemented when FCRMarket fully owns its pipeline.
        raise NotImplementedError("FCRMarket.calculate_reward: use the legacy env bridge for now")

    def calculate_financials(self, state: MarketState) -> Financials:
        # Financials calculation currently uses the legacy env-based bridge.
        raise NotImplementedError(
            "FCRMarket.calculate_financials: use the legacy env bridge for now"
        )

    # ------------------------------------------------------------------
    # Capacity commitment
    # ------------------------------------------------------------------

    def get_capacity_commitment(self, state: MarketState) -> np.ndarray:
        """Return per-quarter-hour committed FCR capacity (MW)."""
        commitment = np.zeros(STEPS_PER_DAY, dtype=np.float32)
        for slot in range(SLOTS_PER_DAY):
            if state.slot_statuses[slot] == SlotStatus.WON:
                start = slot * STEPS_PER_SLOT
                end = start + STEPS_PER_SLOT
                commitment[start:end] = state.bids[slot].capacity_mw
        return commitment

    # ------------------------------------------------------------------
    # Private: auction clearing
    # ------------------------------------------------------------------

    def _simulate_auction(
        self,
        bid: Bid,
        order_book: list[AuctionBid] | list[dict[str, Any]],
        historical_settlement_price: float,
        vpp_total_steps: np.ndarray,
    ) -> tuple[SlotStatus, float, dict[str, Any]]:
        """Simulate the FCR auction clearing for a single slot.

        Algorithm:
        1. If ``bid.capacity_mw == 0`` -> NOT_PARTICIPATED.
        2. If ``bid.price_eur_per_mw > historical_settlement_price`` -> LOST.
        3. Otherwise: compute CBMP, determine if DE has LMP or CBMP, sort
           bids by price, replace bids from highest price downward until the
           agent's bid fits (skip indivisible bids), set new settlement price.

        Note: This auction clearing algorithm uses a "bid replacement" approach from the
        original thesis, where the agent's bid replaces existing marginal bids from the
        highest-priced end. This is a simplified model that approximates the effect of
        the agent's participation on the FCR settlement price.

        The actual German FCR tender (regelleistung.net) works as follows:
        1. Bids sorted ascending by price (cheapest first)
        2. Accepted until required capacity is reached
        3. Settlement price = highest accepted bid (pay-as-cleared)
        4. CBMP harmonization across control areas (if applicable)

        The bid replacement approach was chosen because the thesis uses historical
        bid data and simulates "what would have happened if the agent participated"
        rather than re-running the full clearing from scratch.

        Returns:
            ``(slot_status, settlement_price, delivery_data_dict)``
        """
        delivery_data: dict[str, Any] = {
            "agent_bid_size": bid.capacity_mw,
            "agent_bid_price": bid.price_eur_per_mw,
            "vpp_total_steps": vpp_total_steps,
            "slot_status": SlotStatus.NOT_PARTICIPATED,
            "settlement_price": historical_settlement_price,
        }

        # --- Not participated ---
        if bid.capacity_mw == 0:
            delivery_data["slot_status"] = SlotStatus.NOT_PARTICIPATED
            return SlotStatus.NOT_PARTICIPATED, historical_settlement_price, delivery_data

        settlement_price = historical_settlement_price

        # --- Lost: bid price above settlement ---
        if bid.price_eur_per_mw > settlement_price:
            delivery_data["slot_status"] = SlotStatus.LOST
            delivery_data["settlement_price"] = settlement_price
            return SlotStatus.LOST, settlement_price, delivery_data

        # --- Won: run the auction bid replacement logic ---
        # Normalise order_book entries to AuctionBid objects
        normalised_book: list[AuctionBid] = []
        for entry in order_book:
            if isinstance(entry, AuctionBid):
                normalised_book.append(entry)
            elif isinstance(entry, dict):
                normalised_book.append(AuctionBid(**entry))
            else:
                normalised_book.append(entry)

        # Sort by price descending (highest first)
        normalised_book.sort(key=lambda b: b.price, reverse=True)

        # Replace bids from the most expensive end until the agent's bid fits
        remaining_capacity = bid.capacity_mw
        new_settlement = settlement_price

        for i, existing_bid in enumerate(normalised_book):
            if remaining_capacity <= 0:
                break
            # Skip indivisible bids
            if existing_bid.indivisible:
                continue
            # Replace this bid (partially or fully)
            replaced = min(existing_bid.size, remaining_capacity)
            remaining_capacity -= replaced
            normalised_book[i] = AuctionBid(
                price=existing_bid.price,
                size=existing_bid.size - replaced,
                country=existing_bid.country,
                indivisible=existing_bid.indivisible,
            )

        # New settlement price: highest price among remaining non-zero bids,
        # or the agent's price if it becomes the marginal bid
        active_prices = [b.price for b in normalised_book if b.size > 0]
        new_settlement = max(active_prices) if active_prices else bid.price_eur_per_mw

        # The agent wins if their price <= new settlement
        if bid.price_eur_per_mw <= new_settlement:
            delivery_data["slot_status"] = SlotStatus.WON
            delivery_data["settlement_price"] = new_settlement
            return SlotStatus.WON, new_settlement, delivery_data
        else:
            delivery_data["slot_status"] = SlotStatus.LOST
            delivery_data["settlement_price"] = new_settlement
            return SlotStatus.LOST, new_settlement, delivery_data


# ---------------------------------------------------------------------------
# Legacy function-based API (used by VPPBiddingEnv._simulate_fcr bridge)
# ---------------------------------------------------------------------------


def simulate_market(env: Any, action_dict: dict[str, Any]) -> None:
    """Simulate FCR market clearing for all slots using the env object.

    This preserves the original function signature for backward compatibility
    with the environment's legacy FCR bridge.

    Note: This auction clearing algorithm uses a "bid replacement" approach from the
    original thesis, where the agent's bid replaces existing marginal bids from the
    highest-priced end. This is a simplified model that approximates the effect of
    the agent's participation on the FCR settlement price.

    The actual German FCR tender (regelleistung.net) works as follows:
    1. Bids sorted ascending by price (cheapest first)
    2. Accepted until required capacity is reached
    3. Settlement price = highest accepted bid (pay-as-cleared)
    4. CBMP harmonization across control areas (if applicable)

    The bid replacement approach was chosen because the thesis uses historical
    bid data and simulates "what would have happened if the agent participated"
    rather than re-running the full clearing from scratch.

    Mutates ``env.delivery_results`` in-place.
    """
    auction_bids = env.bids_df[env.market_start : env.market_end]

    env.delivery_results["agents_bid_prices"] = [None] * SLOTS_PER_DAY
    env.delivery_results["agents_bid_sizes_round"] = [None] * SLOTS_PER_DAY
    env.delivery_results["slots_won"] = [None] * SLOTS_PER_DAY
    env.delivery_results["day_reward_list"] = [0.0] * SLOTS_PER_DAY

    for slot in range(len(env.slot_date_list)):
        slot_date = env.slot_date_list[slot]
        slot_bids = auction_bids[slot_date:slot_date].reset_index(drop=True).reset_index(drop=False)
        slot_bids_list = slot_bids.to_dict("records")

        agents_bid_size = round(action_dict["size"][slot])
        env.delivery_results["agents_bid_sizes_round"][slot] = agents_bid_size

        agents_bid_price = action_dict["price"][slot]
        env.delivery_results["agents_bid_prices"][slot] = agents_bid_price

        # Get settlement price for DE
        settlement_price_DE = next(
            bid["settlement_price"] for bid in slot_bids_list if bid["country"] == "DE"
        )
        env.delivery_results["slot_settlement_prices_DE"][slot] = settlement_price_DE

        # Not participating
        if agents_bid_size == 0.0:
            env.delivery_results["slots_won"][slot] = 0
        else:
            # Lost: bid price above settlement
            if agents_bid_price > settlement_price_DE:
                env.delivery_results["slots_won"][slot] = -1
            else:
                # Compute CBMP
                unique_country_bids = list({v["country"]: v for v in slot_bids_list}.values())
                grouped_prices = [x["settlement_price"] for x in unique_country_bids]
                cbmp = max(set(grouped_prices), key=grouped_prices.count)

                price_filter = cbmp if cbmp == settlement_price_DE else settlement_price_DE

                slot_bids_list_sorted = sorted(slot_bids_list, key=lambda x: x["price"])
                slot_bids_filtered = [
                    bid for bid in slot_bids_list_sorted if bid["settlement_price"] == price_filter
                ]
                accumulated_replaced_capacity = 0

                slot_bids_filtered_size_sum = sum(bid["size"] for bid in slot_bids_filtered)

                if agents_bid_size >= slot_bids_filtered_size_sum:
                    env.delivery_results["slots_won"][slot] = -1
                    env.delivery_results["slot_settlement_prices_DE"][slot] = settlement_price_DE
                else:
                    for bid_idx in range(len(slot_bids_filtered)):
                        bid_capacity = slot_bids_filtered[-(bid_idx + 1)]["size"]
                        accumulated_replaced_capacity += bid_capacity

                        if accumulated_replaced_capacity >= agents_bid_size:
                            if (
                                slot_bids_filtered[-(bid_idx + 1)].get("indivisible", False)
                                is False
                            ):
                                new_settlement_price_DE = slot_bids_filtered[-(bid_idx + 1)][
                                    "price"
                                ]
                                env.delivery_results["slots_won"][slot] = 1
                                env.delivery_results["slot_settlement_prices_DE"][slot] = (
                                    new_settlement_price_DE
                                )
                                break
                            else:
                                accumulated_replaced_capacity -= bid_capacity
                                continue
