# Basics
import os
import random

# Data
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler

# Logging
import logging
import wandb

# RL
from gym import Env
from gym.spaces import Box, Dict, MultiDiscrete

# import other python files
from .render import render
from .reward import calculate_reward_and_financials
from .activation import prepare_activation
from .market import simulate_market
from .vpp import configure_vpp, simulate_vpp
from .observation import get_observation


class VPPBiddingEnv(Env):
    """
    A virtual power plant (VPP) bidding environment for training, evaluating, and testing reinforcement learning agents.

    The VPPBiddingEnv class is a subclass of the Env class from the OpenAI gym library. It simulates the bidding process of a VPP in the electricity market. It loads data from a configuration file and several CSV files containing data about renewables, tenders, market results, bids, time features, and market prices. The environment has three modes: "training", "eval", and "test". In "training" mode, the environment is used to train a reinforcement learning agent. In "eval" mode, the environment is used to evaluate the performance of a trained agent. In "test" mode, the environment is used to test the performance of a trained agent. The environment also has two render modes: "human" and "fast_training". In "human" mode, the environment's state and actions are printed to the console. In "fast_training" mode, the environment's state and actions are not printed to the console and the environment runs faster.

    Attributes:
        metadata (dict): A dictionary containing the environment's render modes.
        config (dict): A dictionary containing the configuration data loaded from the configuration file.
        renewables_df (pd.DataFrame): A DataFrame containing data about renewables.
        tenders_df (pd.DataFrame): A DataFrame containing data about tenders.
        market_results (pd.DataFrame): A DataFrame containing data about market results.
        bids_df (pd.DataFrame): A DataFrame containing data about bids.
        time_features_df (pd.DataFrame): A DataFrame containing data about time features.
        market_prices_df (pd.DataFrame): A DataFrame containing data about market prices.
        test_set_date_list (List[str]): A list of strings containing the dates of the slots in the test set.
        asset_data (pd.DataFrame): A DataFrame containing data about the VPP's assets.
        asset_data_FCR (pd.DataFrame): A DataFrame containing data about the VPP's assets in FCR (Flexibility Capacity Remaining) format.
        asset_data_total (pd.Series): A Series containing the total data for the VPP's assets.
        asset_data_FCR_total (pd.Series): A Series containing the total data for the VPP's assets in FCR format.
        total_slot_FCR_demand (float): The total FCR demand for the current slot.
        mean_bid_price_germany (float): The mean bid price in Germany for the current slot.
        hist_window_size (int): The size of the historical window.
        forecast_window_size (int): The size of the forecasting window.
        first_slot_date_start (pd.Timestamp): The starting date of the first slot.
        last_slot_date_end (pd.Timestamp): The ending date of the last slot.
        env_type (str): The type of the environment ("training", "eval", or "test").
        render_mode (str): The render mode of the environment ("human" or "fast_training").
        training_seed (int): The seed used to initialize the environment's random number generator in training mode.

    """

    metadata = {"render_modes": ["human", "fast_training"]}

    def __init__(
        self,
        config_path,
        log_level,
        env_type,
        seed,
        render_mode=None,
    ):
        logger = logging.getLogger()

        while logger.hasHandlers():
            logger.removeHandler(logger.handlers[0])

        logger.setLevel(log_level)
        fhandler = logging.StreamHandler()

        if env_type == "training":
            self.env_type = "training"
            try:
                os.remove("logs/training.log")
            except OSError:
                pass
            fhandler = logging.FileHandler(filename="logs/training.log", mode="w")
        if env_type == "eval":
            self.env_type = "eval"
            try:
                os.remove("logs/eval.log")
            except OSError:
                pass
            fhandler = logging.FileHandler(filename="logs/eval.log", mode="w")
        if env_type == "test":
            self.env_type = "test"

        logger.addHandler(fhandler)

        logging.debug(
            "log_step: "
            + str("initial")
            + " // slot: "
            + "initial "
            + " log level = debug"
        )
        logging.info(
            "log_step: "
            + str("initial")
            + " // slot: "
            + "initial "
            + " log level = info"
        )
        logging.warning(
            "log_step: "
            + str("initial")
            + " // slot: "
            + "initial "
            + " log level = warning"
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.training_seed = seed
        self._seed_everything(self.training_seed)

        # data
        self.config = self._load_config(config_path)

        self.renewables_df: pd.DataFrame = self._load_data("renewables")
        self.tenders_df = self._load_data("tenders")
        self.market_results = self._load_data("market_results")
        self.bids_df = self._load_data("bids")
        self.time_features_df = self._load_data("time_features")
        self.market_prices_df = self._load_data("market_prices")
        self.test_set_date_list = self._load_test_set()

        self.asset_data, self.asset_data_FCR = configure_vpp(self)
        self.asset_data_total = self.asset_data.loc[:, "Total"]
        self.asset_data_FCR_total = self.asset_data_FCR.loc[:, "Total"]
        # replaced self.asset_data_total with self.asset_data_FCR_total

        self.total_slot_FCR_demand = None
        self.mean_bid_price_germany = 0.0

        # window_size
        self.hist_window_size = self.config["config"]["time"]["hist_window_size"]
        self.forecast_window_size = self.config["config"]["time"][
            "forecast_window_size"
        ]

        # episode
        if self.env_type == "training" or self.env_type == "test":
            self.first_slot_date_start = pd.to_datetime(
                self.config["config"]["time"]["first_slot_date_start"]
            )
            self.last_slot_date_end = pd.to_datetime(
                self.config["config"]["time"]["last_slot_date_end"]
            )
        if self.env_type == "eval":
            self.first_slot_date_start = pd.to_datetime(self.test_set_date_list[0], utc=True) - pd.offsets.DateOffset(hours=2)  # type: ignore
            self.last_slot_date_end = pd.to_datetime(self.test_set_date_list[-1], utc=True) + pd.offsets.DateOffset(hours=21, minutes=45)  # type: ignore

        # Timeselection of Dataframes
        self.renewables_df = self.renewables_df[
            self.first_slot_date_start : self.last_slot_date_end
        ]
        self.tenders_df = self.tenders_df[
            self.first_slot_date_start : self.last_slot_date_end
        ]
        # start prior to first_slot_date_start as data is needed for historic market results
        self.market_results = self.market_results[: self.last_slot_date_end]
        self.bids_df = self.bids_df[
            self.first_slot_date_start : self.last_slot_date_end
        ]
        self.time_features_df = self.time_features_df[
            self.first_slot_date_start : self.last_slot_date_end
        ]

        # slot start , gate closure, auction time
        self.lower_slot_start_boundary = self.first_slot_date_start
        self.gate_closure = pd.to_datetime(
            self.tenders_df[self.lower_slot_start_boundary :]["GATE_CLOSURE_TIME"][0]
        )
        self.slot_start = self.tenders_df[self.lower_slot_start_boundary :].index[0]
        self.bid_submission_time = self.gate_closure - pd.offsets.DateOffset(hours=1)  # type: ignore

        logging.debug(
            "initial self.first_slot_date_start = " + str(self.first_slot_date_start)
        )
        logging.debug(
            "initial self.lower_slot_start_boundary = "
            + str(self.lower_slot_start_boundary)
        )
        logging.debug("initial self.slot_start = " + str(self.slot_start))
        logging.debug(
            "initial self.bid_submission_time = " + str(self.bid_submission_time)
        )

        self.initial = True
        self.done = None
        self.total_reward = 0.0
        self.total_revenue = 0.0
        self.total_penalties = 0.0
        self.total_profit = 0.0
        self.total_won_count = 0
        self.total_lost_count = 0
        self.total_not_part_count = 0
        self.total_res_count = 0
        self.total_not_res_count = 0
        self.total_activ_count = 0
        self.total_not_activ_count = 0
        self.history = None
        self.current_daily_mean_market_price = 0.0
        self.delivery_results = {}
        self.previous_delivery_results = {}
        self.logging_step = -1

        # Spaces

        # Action Space

        # Convert complex action space to flattended space
        # 12 values from  min 0.0
        action_low = np.array([-1.0] * 12, dtype=np.float32)
        # 6 values to max maximum_possible_FCR_capacity = the bid sizes
        # 6 values to max maximum_possible_market_price = the bid prices
        action_high = np.array([1.0] * 12, dtype=np.float32)
        self.action_space = Box(low=action_low, high=action_high, dtype=np.float32)

        ### Normalization of Action Space
        self.size_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.size_scaler.fit(np.array([0.0, self.asset_data_FCR_total.max()]).reshape(-1, 1))  # type: ignore
        logging.debug(
            "Size Normalization: size_scaler min = "
            + str(self.size_scaler.data_min_)
            + " and max = "
            + str(self.size_scaler.data_max_)
        )

        self.price_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.price_scaler.fit(np.array([0.0, self.market_results["DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"].max()]).reshape(-1, 1))  # type: ignore
        logging.debug(
            "Price Normalization: price_scaler min = "
            + str(self.price_scaler.data_min_)
            + " and max = "
            + str(self.price_scaler.data_max_)
        )

        # predicted_market_prices_scaler not needed, as self.price_scaler from action space can be used

        self.weekday_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.weekday_scaler.fit(np.array(self.time_features_df["weekday"].values).reshape(-1, 1))  # type: ignore

        self.month_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.month_scaler.fit(np.array(self.time_features_df["month"].values).reshape(-1, 1))  # type: ignore

        self.bool_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.bool_scaler.fit(np.array([0, 1]).reshape(-1, 1))  # type: ignore

        self.list_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.list_scaler.fit(np.array([-1, 0, 1]).reshape(-1, 1))  # type: ignore

        self.slot_settlement_prices_DE_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.slot_settlement_prices_DE_scaler.fit(np.array([0.0, 4257.07]).reshape(-1, 1))  # type: ignore

        # Observation Space
        self.observation_space = Dict(
            {
                "asset_data_forecast": Box(
                    low=-1.0, high=1.0, shape=(6,), dtype=np.float32
                ),
                "predicted_market_prices": Box(
                    low=-1.0, high=1.0, shape=(6,), dtype=np.float32
                ),  # for each slot, can be prices of same day last week
                "weekday": Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                ),  # Discrete(7), # for the days of the week
                "month": Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                ),  # Discrete(12),
                "slots_won": Box(
                    low=-1.0, high=1.0, shape=(6,), dtype=np.float32
                ),  # MultiBinary(6), #boolean for each slot, 0 if loss , 1 if won
                "slots_reserved": Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
                "day_reward_list": Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32),
            }
        )

    def _load_test_set(self):
        """
        Loads the test set data from a CSV file and extracts a list of the dates of the slots in the test set.

        Returns:
        List[str]: A list of strings containing the dates of the slots in the test set.
        """
        df = self._load_data("test_set")
        df = df.reset_index()
        df["ts"] = df["ts"].astype(str)
        test_set_date_list = df["ts"].tolist()
        return test_set_date_list

    def _load_config(self, config_path):
        """
        Loads the configuration data from the configuration file.

        Args:
        config_path (str): The path to the configuration file.

        Returns:
        dict: The configuration data.
        """
        with open(config_path, "r") as f:
            config = json.load(f)
        return config

    def _load_data(self, data_source):
        """
        Loads data from a CSV file.

        Args:
        data_source (str): The type of data to load. Can be one of "renewables", "tenders", "market_results", "bids", "time_features", "market_prices", or "test_set".

        Returns:
        pd.DataFrame: A DataFrame containing the data.
        """
        df = pd.read_csv(self.config["config"]["csv_paths"][data_source], sep=";", index_col=0)  # type: ignore
        df.index = pd.to_datetime(df.index)
        return df

    def reset(self, seed=None, return_info=False, options=None):
        """
        The reset method will be called to initiate a new episode.
        reset should be called whenever a done signal has been issued

        Args:
            seed (type, optional): The seed for the random number generator. Defaults to None.
            return_info (bool, optional): A boolean indicating whether to return additional information. Defaults to False.
            options (type, optional): Additional options. Defaults to None.

        Returns:
            observation: The state of the environment.
        """
        self._seed_everything(self.training_seed)

        if self.initial is False:

            if self.env_type == "training":
                if self.forecast_end == self.last_slot_date_end:
                    self.lower_slot_start_boundary = self.first_slot_date_start

            if self.env_type == "training" or self.env_type == "test":
                # check if next slot date is in Test set
                next_date_in_test_set = True
                while next_date_in_test_set is True:
                    self.lower_slot_start_boundary = (
                        self.lower_slot_start_boundary + pd.offsets.DateOffset(days=1)
                    )
                    # if date is in test set, skip the date
                    # lower_slot_start_boundary is either 22:00 or 23:00 of the previous day, so add 1 day
                    date_to_check = (
                        self.lower_slot_start_boundary + pd.offsets.DateOffset(days=1)
                    )
                    # convert to string and only take the date digits
                    date_to_check = str(date_to_check)[0:10]
                    if date_to_check in self.test_set_date_list:
                        next_date_in_test_set = True
                    else:
                        next_date_in_test_set = False

            if self.env_type == "eval":
                next_date_in_test_set = False
                while next_date_in_test_set is False:
                    self.lower_slot_start_boundary = (
                        self.lower_slot_start_boundary + pd.offsets.DateOffset(days=1)
                    )
                    # if date is NOT in test set, skip the date
                    # lower_slot_start_boundary is either 22:00 or 23:00 of the previous day, so add 1 day
                    date_to_check = (
                        self.lower_slot_start_boundary + pd.offsets.DateOffset(days=1)
                    )
                    # convert to string and only take the date digits
                    date_to_check = str(date_to_check)[0:10]
                    if date_to_check not in self.test_set_date_list:
                        next_date_in_test_set = False
                        if (pd.to_datetime(date_to_check, utc=True)) > pd.to_datetime(self.test_set_date_list[-1], utc=True):  # type: ignore
                            break
                    else:
                        next_date_in_test_set = True

            self.gate_closure = pd.to_datetime(self.tenders_df[self.lower_slot_start_boundary :]["GATE_CLOSURE_TIME"][0])  # type: ignore
            self.slot_start = self.tenders_df[self.lower_slot_start_boundary :].index[0]
            self.bid_submission_time = self.gate_closure - pd.offsets.DateOffset(
                hours=1
            )

            logging.info(
                "log_step: "
                + str(self.logging_step)
                + " slot: "
                + "None"
                + " new self.lower_slot_start_boundary = "
                + str(self.lower_slot_start_boundary)
            )
            logging.info(
                "log_step: "
                + str(self.logging_step)
                + " slot: "
                + "None"
                + " self.gate_closure = "
                + str(self.gate_closure)
            )
            logging.info(
                "log_step: "
                + str(self.logging_step)
                + " slot: "
                + "None"
                + " self.slot_start = "
                + str(self.slot_start)
            )
            logging.info(
                "log_step: "
                + str(self.logging_step)
                + " slot: "
                + "None"
                + " self.bid_submission_time = "
                + str(self.bid_submission_time)
            )

        self.total_slot_FCR_demand = self.tenders_df[str(self.slot_start) :]["total"][0]  # type: ignore

        self.done = False

        self.previous_delivery_results = self.delivery_results.copy()
        logging.debug(
            "log_step: "
            + str(self.logging_step)
            + " slot: "
            + "None"
            + " self.delivery_results = "
            + str(self.delivery_results)
        )
        self.delivery_results.clear()
        logging.debug(
            "log_step: "
            + str(self.logging_step)
            + " slot: "
            + "None"
            + " self.delivery_results after clearing"
        )
        logging.debug(
            "log_step: "
            + str(self.logging_step)
            + " slot: "
            + "None"
            + " self.delivery_results = "
            + str(self.delivery_results)
        )
        logging.debug(
            "log_step: "
            + str(self.logging_step)
            + " slot: "
            + "None"
            + " self.previous_delivery_results after clearing"
        )
        logging.debug(
            "log_step: "
            + str(self.logging_step)
            + " slot: "
            + "None"
            + " self.previous_delivery_results = "
            + str(self.previous_delivery_results)
        )

        self.delivery_results["slots_won"] = [None] * 6
        self.delivery_results["day_reward_list"] = [None] * 6
        self.delivery_results["slot_settlement_prices_DE"] = [None] * 6
        self.delivery_results["reserved_slots"] = [None] * 6
        self.delivery_results["activated_slots"] = [None] * 6

        # reset for each episode
        self._get_new_timestamps()

        # get time stamp index that is nearest to market end
        market_end_date_index = self.market_prices_df.index.get_indexer(
            [self.market_end], method="nearest"
        )
        # get price df
        market_price_df = self.market_prices_df.iloc[market_end_date_index]
        # get the price value
        self.current_daily_mean_market_price = market_price_df["price_DE"].values[0]  # type: ignore

        # get new observation (either the initial one or the one after reset  )
        observation = get_observation(self)

        # when first Episode is finished, set boolean.
        self.initial = False

        self.logging_step += 1
        logging.error(
            "log_step: "
            + str(self.logging_step)
            + " slot: "
            + "None"
            + " logging_step: "
            + str(self.logging_step)
        )

        return observation

    def _get_new_timestamps(self):
        """
        Determines the timestamps for the historic data, forecast, and market.

        Sets the following attributes:
        historic_data_start: The start timestamp for the historic data.
        historic_data_end: The end timestamp for the historic data.
        forecast_start: The start timestamp for the forecast.
        forecast_end: The end timestamp for the forecast.
        market_start: The start timestamp for the market.
        market_end: The end timestamp for the market.
        slot_date_list: A list of timestamps for the slots in the market.
        """
        self.historic_data_start = self.bid_submission_time - pd.offsets.DateOffset(
            days=self.hist_window_size
        )
        self.historic_data_end = self.bid_submission_time - pd.offsets.DateOffset(
            minutes=15
        )
        logging.debug(
            "log_step: "
            + str(self.logging_step)
            + " slot: "
            + "None"
            + " self.historic_data_start = "
            + str(self.historic_data_start)
        )
        logging.debug(
            "log_step: "
            + str(self.logging_step)
            + " slot: "
            + "None"
            + " self.historic_data_end = "
            + str(self.historic_data_end)
        )

        self.forecast_start = self.slot_start
        self.forecast_end = self.forecast_start + pd.offsets.DateOffset(days=self.forecast_window_size) - pd.offsets.DateOffset(minutes=15)  # type: ignore
        logging.debug(
            "log_step: "
            + str(self.logging_step)
            + " slot: "
            + "None"
            + " self.forecast_start = "
            + str(self.forecast_start)
        )
        logging.debug(
            "log_step: "
            + str(self.logging_step)
            + " slot: "
            + "None"
            + " self.forecast_end = "
            + str(self.forecast_end)
        )

        self.market_start = self.slot_start
        self.market_end = self.market_start + pd.offsets.DateOffset(hours=24) - pd.offsets.DateOffset(minutes=15)  # type: ignore
        logging.debug(
            "log_step: "
            + str(self.logging_step)
            + " slot: "
            + "None"
            + " self.market_start = "
            + str(self.market_start)
        )
        logging.debug(
            "log_step: "
            + str(self.logging_step)
            + " slot: "
            + "None"
            + " self.market_end = "
            + str(self.market_end)
        )

        self.slot_date_list = self.tenders_df[self.market_start :][0:6].index
        logging.debug(
            "log_step: "
            + str(self.logging_step)
            + " slot: "
            + "None"
            + " self.slot_date_list = "
            + str(self.slot_date_list)
        )

    def _add_gaussian_noise(self, data, whole_data):
        """
        Adds Gaussian noise to the given data.

        Args:
        data (np.ndarray): The data to add noise to.
        whole_data (np.ndarray): The whole dataset, used to determine the standard deviation of the noise.

        Returns:
        np.ndarray: The data with added noise.
        """
        mean = 0.0
        standard_deviation = np.std(whole_data)
        standard_deviation_gaussian = standard_deviation * 0.2  # for 20% Gaussian noise
        noise = np.random.normal(mean, standard_deviation_gaussian, size=data.shape)
        data_noisy = data + noise
        # Set negative values to 0
        data_noisy = data_noisy.clip(min=0)

        return data_noisy

    def step(self, action):
        """
        Take a step in the environment, applying the given action.

        Args:
            action: The action to be taken, given as a list of bid sizes and bid prices for each hour. The bid sizes are normalized and the bid prices are given in €/MWh.

        Returns:
            observation: The observation of the environment after the action has been taken. This is a dictionary.
            reward: The reward for taking the given action.
            done: A boolean indicating whether the episode is over.
            info: A dictionary containing additional information about the step.
        """
        # convert action list with shape (12,) into dict
        bid_sizes_normalized = action[0:6]
        bid_prices_normalized = action[6:]
        bid_sizes_converted = self.size_scaler.inverse_transform(
            np.array(bid_sizes_normalized).reshape(-1, 1)
        )
        bid_prices_converted = self.price_scaler.inverse_transform(
            np.array(bid_prices_normalized).reshape(-1, 1)
        )

        # convert from 2d array to list
        bid_sizes_converted = [x for xs in list(bid_sizes_converted) for x in xs]
        bid_prices_converted = [x for xs in list(bid_prices_converted) for x in xs]
        action_dict = {"size": bid_sizes_converted, "price": bid_prices_converted}

        # Simulate VPP
        simulate_vpp(self)

        # Simulate Market
        # take the bid out of the action of the agent and resimulate the market clearing algorithm
        simulate_market(self, action_dict)

        # Prepare the data for the activation simulation and reward calculation
        prepare_activation(self)

        # calculate reward from state and action
        (
            step_reward,
            step_revenue,
            step_penalties,
            step_profit,
        ) = calculate_reward_and_financials(self)

        info, logging_dict = self._calculate_dashboard_metrics(
            step_reward, step_revenue, step_penalties, step_profit
        )

        self._update_history(info)
        self.done = True

        # get the observation after action was taken (still same VPP data, but updated won slots and settlement prices)
        observation = get_observation(self)

        if self.env_type != "test":

            if self.env_type == "training":
                if self.render_mode == "fast_training":
                    # logs need to be committed here as they wont be commited in render()
                    wandb.log(logging_dict, commit=True)

                if self.render_mode == "human":
                    # dont commit the logs to wandb, as logs are committed in render funciton
                    wandb.log(logging_dict, commit=False)
                    self.render(mode="human")

            if self.env_type == "eval":
                if self.render_mode == "human":
                    wandb.log(logging_dict, commit=False)
                    self.render(mode="human")

        return observation, step_reward, self.done, info

    def render(self, mode="human"):
        render(self, mode=mode)

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _convert_number_to_float(self, x):
        return np.float32(x)

    def _seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        self.seed(seed)

    def _calculate_dashboard_metrics(
        self, step_reward, step_revenue, step_penalties, step_profit
    ):
        """
        Calculates various metrics to be used in the dashboard.

        This method calculates several performance metrics based on the inputs provided and the current state of the environment. These metrics include step reward, step revenue, step penalties, and step profit, as well as various counts and ratios related to the number of won, lost, and not participated slots, reserved slots, and activated slots. It also updates the total reward, total revenue, total penalties, and total profit.

        Args:
        step_reward (float): The reward earned in this step.
        step_revenue (float): The revenue earned in this step.
        step_penalties (float): The penalties incurred in this step.
        step_profit (float): The profit earned in this step.

        Returns:
        dict: A dictionary containing the calculated metrics.
        dict: A dictionary containing the logging metrics.
        """

        # Step Metrics
        step_won_count = self.delivery_results["slots_won"].count(1)
        step_lost_count = self.delivery_results["slots_won"].count(-1)
        step_not_part_count = self.delivery_results["slots_won"].count(0)

        step_res_count = self.delivery_results["reserved_slots"].count(1)
        step_not_res_count = self.delivery_results["reserved_slots"].count(-1)

        step_activ_count = self.delivery_results["activated_slots"].count(1)
        step_not_activ_count = self.delivery_results["activated_slots"].count(-1)

        # Calaucalte Ratios

        step_won_ratio = step_won_count / 6

        # zero division error safety net
        if (6 - step_not_part_count) == 0:
            step_res_ratio = 0
        else:
            step_res_ratio = step_res_count / (6 - step_not_part_count)
        # zero division error safety net
        if (6 - step_not_part_count - step_not_res_count) == 0:
            step_activ_ratio = 0
        else:
            step_activ_ratio = step_activ_count / (
                6 - step_not_part_count - step_not_res_count
            )

        # Total Metrics
        # update total reward
        self.total_reward += step_reward
        # Update the total profit
        self.total_revenue += step_revenue
        self.total_penalties += step_penalties
        self.total_profit += step_profit

        self.total_won_count += step_won_count
        self.total_lost_count += step_lost_count
        self.total_not_part_count += step_not_part_count

        self.total_res_count += step_res_count
        self.total_not_res_count += step_not_res_count

        self.total_activ_count += step_activ_count
        self.total_not_activ_count += step_not_activ_count

        info = dict(
            bid_submission_time=str(self.bid_submission_time),  # type: ignore
            step_reward=round(float(step_reward), 2),  # type: ignore
            step_profit=round(step_profit, 2),  # type: ignore
            step_revenue=round(step_revenue, 2),  # type: ignore
            step_penalties=round(step_penalties, 2),  # type: ignore
            step_won_count=step_won_count,
            step_lost_count=step_lost_count,
            step_not_part_count=step_not_part_count,
            step_res_count=step_res_count,
            step_not_res_count=step_not_res_count,
            step_activ_count=step_activ_count,
            step_not_activ_count=step_not_activ_count,
            step_won_ratio=step_won_ratio,
            step_res_ratio=step_res_ratio,
            step_activ_ratio=step_activ_ratio,
            total_reward=round(self.total_reward, 2),  # type: ignore
            total_revenue=round(self.total_revenue, 2),  # type: ignore
            total_penalties=round(self.total_penalties, 2),  # type: ignore
            total_profit=round(self.total_profit, 2),  # type: ignore
            total_won_count=self.total_won_count,
            total_lost_count=self.total_lost_count,
            total_not_part_count=self.total_not_part_count,
            total_res_count=self.total_res_count,
            total_not_res_count=self.total_not_res_count,
            total_activ_count=self.total_activ_count,
            total_not_activ_count=self.total_not_activ_count,
        )

        # define basic logging dict
        logging_dict = {
            "global_step": self.logging_step,
            "step_reward": step_reward,
            "step_revenue": step_revenue,
            "step_penalties": step_penalties,
            "step_profit": step_profit,
            "step_won_count": step_won_count,
            "step_lost_count": step_lost_count,
            "step_not_part_count": step_not_part_count,
            "step_res_count": step_res_count,
            "step_not_res_count": step_not_res_count,
            "step_activ_count": step_activ_count,
            "step_not_activ_count": step_not_activ_count,
            "step_won_ratio": step_won_ratio,
            "step_res_ratio": step_res_ratio,
            "step_activ_ratio": step_activ_ratio,
            "total_reward": self.total_reward,
            "total_revenue": self.total_revenue,
            "total_penalties": self.total_penalties,
            "total_profit": self.total_profit,
            "total_won_count": self.total_won_count,
            "total_lost_count": self.total_lost_count,
            "total_not_part_count": self.total_not_part_count,
            "total_res_count": self.total_res_count,
            "total_not_res_count": self.total_not_res_count,
            "total_activ_count": self.total_activ_count,
            "total_not_activ_count": self.total_not_activ_count,
        }

        return info, logging_dict
