import logging
from collections import OrderedDict
import numpy as np
import pandas as pd
from typing import List


def get_observation(self):
    """
    This function returns an observation for the current environment state.
    The observation consists of the forecast of the asset data for the forecast window,
    time features for the market window (weekday and month), lists of whether slots were won,
    reserved, or activated, a list of rewards for each slot in the market window, and a list of
    settlement prices for each slot in the market window. If the environment is still in the
    initial state or has just been reset, some of the lists will be filled with default values.
    If the environment has just taken an action and the auction has finished, the lists will be
    filled with the results of the auction. The returned observation is used by the agent to make
    a decision on the next action to take.
    """
    asset_data_forecast = self.asset_data_FCR_total[
        str(self.forecast_start) : str(self.forecast_end)
    ].to_numpy(dtype=np.float32)
    logging.debug(
        "log_step: "
        + str(self.logging_step)
        + " slot: "
        + "None"
        + " asset_data_forecast = "
        + str(
            self.asset_data_FCR_total[str(self.forecast_start) : str(self.forecast_end)]
        )
    )

    # use perfect foresight forecast
    asset_data_forecast = asset_data_forecast.astype(np.float32)
    logging.debug(
        "log_step: "
        + str(self.logging_step)
        + " slot: "
        + "None"
        + " asset_data_forecast = "
        + str(asset_data_forecast)
    )
    asset_data_forecast_norm = self.size_scaler.transform(
        (asset_data_forecast.reshape(-1, 1))
    )
    asset_data_forecast_norm = asset_data_forecast_norm.flatten()

    time_features = self.time_features_df[str(self.market_start) : str(self.market_end)]
    logging.debug(self.time_features_df[str(self.market_start) : str(self.market_end)])

    weekday = int(time_features["weekday"][0])
    weekday_norm = self.weekday_scaler.transform(np.array(weekday).reshape(-1, 1))
    weekday_norm = weekday_norm.flatten().astype("float32")

    month = int(time_features["month"][0])
    month_norm = self.month_scaler.transform(np.array(month).reshape(-1, 1))
    month_norm = month_norm.flatten().astype("float32")

    slots_won_list: List[int] = []
    slots_reserved_list: List[int] = []
    slots_activated_list: List[int] = []
    day_reward_list: List[float] = []

    slot_settlement_prices_DE_list: List[float] = []

    # beim ersten Trainingstep, wenn noch keine Daten vorhanden
    if self.initial is True:
        slots_won_list = [0] * 6
        slots_reserved_list = [0] * 6
        slots_activated_list = [0] * 6
        day_reward_list = [0.0] * 6

        slot_settlement_prices_DE_list = [0.0] * 6

    # observation after reset, before action is taken, NEW VPP data and dates but old auction results
    if (self.done is False) and (self.initial is False):
        slots_won_list = self.previous_delivery_results["slots_won"]
        slots_reserved_list = self.previous_delivery_results["reserved_slots"]
        slots_activated_list = self.previous_delivery_results["activated_slots"]
        day_reward_list = self.previous_delivery_results["day_reward_list"]

        slot_settlement_prices_DE_list = self.previous_delivery_results[
            "slot_settlement_prices_DE"
        ]
        # replace None with 0
        for i in range(len(slot_settlement_prices_DE_list)):
            if slot_settlement_prices_DE_list[i] == None:
                slot_settlement_prices_DE_list[i] = 0.0

    # observation after action was taken and auction is done, VPP data and auction results of auction day
    if (self.done is True) and (self.initial is False):
        slots_won_list = self.delivery_results["slots_won"].copy()
        slots_reserved_list = self.delivery_results["reserved_slots"].copy()
        slots_activated_list = self.delivery_results["activated_slots"].copy()
        day_reward_list = self.delivery_results["day_reward_list"].copy()

        slot_settlement_prices_DE_list = self.delivery_results[
            "slot_settlement_prices_DE"
        ]
        # replace None with 0
        for i in range(len(slot_settlement_prices_DE_list)):
            if slot_settlement_prices_DE_list[i] == None:
                slot_settlement_prices_DE_list[i] = 0.0

    slots_won_array = np.array(slots_won_list, dtype=np.int32)
    slots_won_norm = self.list_scaler.transform(
        np.array(slots_won_array).reshape(-1, 1)
    )
    slots_won_norm = slots_won_norm.flatten().astype("float32")

    slots_reserved_array = np.array(slots_reserved_list, dtype=np.int32)
    slots_reserved_norm = self.list_scaler.transform(
        np.array(slots_reserved_array).reshape(-1, 1)
    )
    slots_reserved_norm = slots_reserved_norm.flatten().astype("float32")

    slots_activated_array = np.array(slots_activated_list, dtype=np.int32)
    slots_activated_norm = self.list_scaler.transform(
        np.array(slots_activated_array).reshape(-1, 1)
    )
    slots_activated_norm = slots_activated_norm.flatten().astype("float32")

    day_reward_array = np.array(day_reward_list, dtype=np.int32)
    day_reward_norm = np.array(day_reward_array).reshape(-1, 1)
    day_reward_norm = day_reward_norm.flatten().astype("float32")

    slot_settlement_prices_DE_array = np.array(
        slot_settlement_prices_DE_list, dtype=np.float32
    )
    slot_settlement_prices_DE_norm = self.slot_settlement_prices_DE_scaler.transform(
        (slot_settlement_prices_DE_array.reshape(-1, 1))
    )
    slot_settlement_prices_DE_norm = slot_settlement_prices_DE_norm.flatten().astype(
        "float32"
    )

    # get min capacity for each slot
    min_slot_vpp_capacity_list = []
    for i in range(0, 96, 16):
        min_slot_vpp_capacity_list.append(min(asset_data_forecast_norm[i : i + 16 :]))

    min_slot_vpp_capacity = np.array(min_slot_vpp_capacity_list, dtype=np.float32)

    # get price for each slot

    predicted_market_prices = self.market_results[
        "DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"
    ][str(self.forecast_start) : str(self.forecast_end)].to_numpy(dtype=np.float32)
    predicted_market_prices_norm = self.price_scaler.transform(
        (np.array(predicted_market_prices).reshape(-1, 1))
    )
    predicted_market_prices_norm = predicted_market_prices_norm.flatten()
    if len(predicted_market_prices_norm) > 6:
        predicted_market_prices_norm = predicted_market_prices_norm[0:6]

    observation = OrderedDict(
        {
            "asset_data_forecast": min_slot_vpp_capacity,
            "predicted_market_prices": predicted_market_prices_norm,
            "weekday": weekday_norm,
            "month": month_norm,
            "slots_won": slots_won_norm,
            "slots_reserved": slots_reserved_norm,
            "day_reward_list": day_reward_norm,
        }
    )

    logging.debug(
        "log_step: "
        + str(self.logging_step)
        + " slot: "
        + "None"
        + " NEW Observation = "
        + str(observation)
    )

    return observation
