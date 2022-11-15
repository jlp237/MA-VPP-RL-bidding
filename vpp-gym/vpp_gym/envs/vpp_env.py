# Basics
import os

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
from gym.spaces import Box, Dict

# import other python files
from .render import render
from .reward import calculate_reward
from .activation import prepare_activation
from .market import simulate_market
from .vpp import configure_vpp, simulate_vpp
from .observation import get_observation


class VPPBiddingEnv(Env):
    metadata = {"render_modes": ["human", "fast_training"]}

    def __init__(self,
                 config_path,
                 log_level, 
                 env_type,
                 render_mode=None
                ):
        logger = logging.getLogger()

        while logger.hasHandlers():
                logger.removeHandler(logger.handlers[0])
        
        logger.setLevel(log_level)
        fhandler = logging.StreamHandler()
        
        if env_type == "training":
            self.env_type = "training"
            os.remove("logs/training.log")
            fhandler = logging.FileHandler(filename='logs/training.log', mode='w')
        if env_type == "eval":
            self.env_type = "eval"
            os.remove("logs/eval.log")
            fhandler = logging.FileHandler(filename='logs/eval.log', mode='w')
        if env_type == "test":
            self.env_type = "test"
        
        logger.addHandler(fhandler)           
        
        logging.debug("log_step: " + str("initial") + " // slot: " +  "initial " + " log level = debug")
        logging.info("log_step: " + str("initial") + " // slot: " +  "initial " + " log level = info")
        logging.warning("log_step: " + str("initial") + " // slot: " +  "initial " + " log level = warning" )
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # data 
        self.config = self._load_config(config_path)
            
        self.renewables_df = self._load_data("renewables")
        self.tenders_df = self._load_data("tenders")
        self.market_results = self._load_data("market_results") 
        self.bids_df = self._load_data("bids") 
        self.time_features_df = self._load_data("time_features") 
        self.market_prices_df = self._load_data("market_prices") 
        self.test_set_date_list = self._load_test_set()
        
        self.asset_data , self.asset_data_FCR = configure_vpp(self)
        self.asset_data_total = self.asset_data.loc[:,"Total"]
        #self.asset_data_FCR_total = self.asset_data_FCR.loc[:,"Total"]
        self.maximum_possible_VPP_capacity = round(self.asset_data_total.max(),2) + 0.01
        
        self.total_slot_FCR_demand = None
        self.mean_bid_price_germany = 0.
        
        
        # window_size
        self.hist_window_size = self.config["config"]["time"]["hist_window_size"]
        self.forecast_window_size = self.config["config"]["time"]["forecast_window_size"]
        
        # episode
        if self.env_type == "training" or self.env_type == "test":
            self.first_slot_date_start = pd.to_datetime(self.config["config"]["time"]["first_slot_date_start"])
            self.last_slot_date_end = pd.to_datetime(self.config["config"]["time"]["last_slot_date_end"])
        if self.env_type == "eval":
            self.first_slot_date_start = pd.to_datetime(self.test_set_date_list[0], utc=True) - pd.offsets.DateOffset(hours = 2)
            self.last_slot_date_end = pd.to_datetime(self.test_set_date_list[-1], utc=True)  + pd.offsets.DateOffset(hours = 21, minutes = 45)

        # Timeselection of Dataframes
        self.renewables_df = self.renewables_df[self.first_slot_date_start:self.last_slot_date_end]
        self.tenders_df = self.tenders_df[self.first_slot_date_start:self.last_slot_date_end]
        self.market_results = self.market_results[:self.last_slot_date_end] # start prior to first_slot_date_start as data is needed for historic market results
        self.bids_df = self.bids_df[self.first_slot_date_start:self.last_slot_date_end]
        self.time_features_df = self.time_features_df[self.first_slot_date_start:self.last_slot_date_end]
        
        #logging.debug("selection self.renewables_df" + str(self.renewables_df))
        #logging.debug("selection self.tenders_df" + str(self.tenders_df))
        #logging.debug("selection self.market_results" + str(self.market_results))
        #logging.debug("selection self.bids_df" + str(self.bids_df))
        #logging.debug("selection self.time_features_df" + str(self.time_features_df))

        # slot start , gate closure, auction time
        self.lower_slot_start_boundary = self.first_slot_date_start
        self.gate_closure = pd.to_datetime(self.tenders_df[self.lower_slot_start_boundary:]["GATE_CLOSURE_TIME"][0])
        self.slot_start = self.tenders_df[self.lower_slot_start_boundary:].index[0]
        self.bid_submission_time = self.gate_closure - pd.offsets.DateOffset(hours = 1)
        
        logging.debug("initial self.first_slot_date_start = " + str(self.first_slot_date_start))
        logging.debug("initial self.lower_slot_start_boundary = " + str(self.lower_slot_start_boundary) )
        logging.debug("initial self.slot_start = " + str(self.slot_start) )
        logging.debug("initial self.bid_submission_time = " + str(self.bid_submission_time) )
        
        self.initial = True
        self.done = None
        self.total_reward = 0.
        self.total_profit = 0.
        self.total_profit_monthly = 0.
        self.violation_counter = 0
        self.history = None
        self.current_daily_mean_market_price = 0.
        self.real_month = self.first_slot_date_start.month
        self.penalty_month_count = 1
        self.month_change = False
        
        self.monthly_penalty = 0
        self.monthly_revenue = 0
        self.monthly_profit = 0
        
        self.activation_results = {}
        self.previous_activation_results  = {}
        
        self.logging_step = 0
            
        # Spaces
        
        # Action Space
        
        # VERSION 1
        # Convert complex action space to flattended space
                        
        # 12 values from  min 0.0
        action_low = np.float32(np.array([-1.0] * 12)) 
        # 6 values to max maximum_possible_FCR_capacity = the bid sizes 
        # 6 values to max maximum_possible_market_price = the bid prices
        action_high = np.float32(np.array([1.0] * 6 + [1.0] * 6 )) 
        
        convert_array_to_float = np.vectorize(self._convert_number_to_float)
        action_low = convert_array_to_float(action_low)
        action_high = convert_array_to_float(action_high)
        
        self.action_space = Box(low=action_low, high=action_high, shape=(12,), dtype=np.float32)

        # VERSION 2 : Box 
        
        '''# Convert complex action space to flattended space
        # bid sizes =  6 DISCRETE slots from 0 to 25  = [ 2s5, 25, 25, 25, 25 , 25]  = in flattened = 150 values [0,1]
        # bid prizes = 6 CONTINUOUS slots from 0 to 100  = [ 100., 100., 100., 100., 100. , 100.]  = in flattened = 150 values [0,1]

        # 156 values from  min 0.0
        action_low = np.float32(np.array([0.0] * 156)) 
        #150 values to max 1.0 = the bid sizes 
        # +6 values to max 100. = the bid prices
        action_high = np.float32(np.array([1.0] * 150 + [100.0]*6)) 
        self.action_space = Box(low=action_low, high=action_high, shape=(156,), dtype=np.float32)'''

        # VERSION 3
        
        
        '''
        self.complex_action_space = Tuple((
            # INFO: TSOs allow divisible and indivisible bids. Biggest divisible bid was 188 MW , maximum price was 4257.07 
            #MultiDiscrete([ 188, 188, 188, 188, 188 , 188]),
            MultiDiscrete([ 25, 25, 25, 25, 25 , 25]),
            #Box(low=0.0, high=np.float32(4257.07), shape=(6,), dtype=np.float32)))
            Box(low=0.0, high=np.float32(100.), shape=(6,), dtype=np.float32)))
        
        #flatten_action_space_64 = flatten_space(self.complex_action_space)
        #self.action_space = flatten_action_space_64

        
        #logging.debug(flatten_action_space_64)
        #logging.debug(type(flatten_action_space_64))
        #logging.debug("#" *42)
        
        #flattened_action = flatten(self.complex_action_space, self.complex_action_space.sample())
        #logging.debug(flattened_action)

        #unflattened_action = unflatten(self.complex_action_space, flattened_action)
        #logging.debug(unflattened_action)'''
        
        # VERSION 4: MultiDiscrete Action Space
        
        '''
        # bid sizes =  6 DISCRETE slots from 0 to 131  = [ 131, 131, 131, 131, 131 , 131]  
        # bid prizes = 6 DISCRETE slots from 0 to 4257  = [ 4257, 4257, 4257, 4257, 4257 , 4257]
        action_space = MultiDiscrete(np.array([[ 131, 131, 131, 131, 131 , 131]  , [ 4257, 4257, 4257, 4257, 4257 , 4257]]))
        '''
        
        ### Normalization of Action Space 
        self.size_scaler = MinMaxScaler(feature_range=(-1,1))
        self.size_scaler.fit(np.array(self.asset_data_total.values).reshape(-1, 1))
        logging.debug("Action Space Normalization: size_scaler min = " + str(self.size_scaler.data_min_) + " and max = " + str(self.size_scaler.data_max_))
        
        self.price_scaler = MinMaxScaler(feature_range=(-1,1))
        self.price_scaler.fit(np.array(self.bids_df["settlement_price"].values).reshape(-1, 1))
        logging.debug("Action Space Normalization: price_scaler min = " + str(self.price_scaler.data_min_) + " and max = " + str(self.price_scaler.data_max_))
        
        ### Normalization of Observation Space
        self.asset_data_historic_scaler = MinMaxScaler(feature_range=(-1,1))
        self.asset_data_historic_scaler.fit(np.array(self.asset_data_total.values).reshape(-1, 1))
        
        noisy_asset_data_forecast_for_fit = self._add_gaussian_noise(self.asset_data_total.to_numpy(dtype=np.float32), self.asset_data_total.to_numpy(dtype=np.float32))
        self.noisy_asset_data_forecast_scaler = MinMaxScaler(feature_range=(-1,1))
        self.noisy_asset_data_forecast_scaler.fit(noisy_asset_data_forecast_for_fit.reshape(-1, 1))
        
        # predicted_market_prices_scaler not needed, as self.price_scaler from action space can be used 
        
        self.weekday_scaler = MinMaxScaler(feature_range=(-1,1))
        self.weekday_scaler.fit(np.array(self.time_features_df["weekday"].values).reshape(-1, 1))
        
        self.week_scaler = MinMaxScaler(feature_range=(-1,1))
        self.week_scaler.fit(np.array(self.time_features_df["week"].values).reshape(-1, 1))

        self.month_scaler = MinMaxScaler(feature_range=(-1,1))
        self.month_scaler.fit(np.array(self.time_features_df["month"].values).reshape(-1, 1))

        self.bool_scaler = MinMaxScaler(feature_range=(-1,1))
        self.bool_scaler.fit(np.array([0,1]).reshape(-1, 1))
        
        self.list_scaler = MinMaxScaler(feature_range=(-1,1))
        self.list_scaler.fit(np.array([-1, 0, 1]).reshape(-1, 1))
        
        self.slot_settlement_prices_DE_scaler = MinMaxScaler(feature_range=(-1,1))
        self.slot_settlement_prices_DE_scaler.fit(np.array([0.0, 4257.07]).reshape(-1, 1))
        
        # Observation Space

        # Create a observation space with all observations inside
        self.observation_space = Dict({
            "asset_data_historic": Box(low=-1.0, high=1.0, shape=(96,), dtype=np.float32),
            "asset_data_forecast": Box(low=-1.0, high=1.0,  shape=(96,), dtype=np.float32),
            "predicted_market_prices":  Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32), # for each slot, can be prices of same day last week 
            "weekday":Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32), #Discrete(7), # for the days of the week
            "week": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32), # Discrete(53),  # for week of the year
            "month": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32), #Discrete(12),
            "isHoliday": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32), #Discrete(2), # holiday = 1, no holiday = 0
            "followsHoliday": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32), #Discrete(2), # followsHoliday = 1, no followsHoliday = 0
            "priorHoliday": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),  #Discrete(2), # priorHoliday = 1, no priorHoliday = 0
            "slots_won": Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32), #MultiBinary(6), #boolean for each slot, 0 if loss , 1 if won 
            "slot_settlement_prices_DE": Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
            })

        
    def _load_test_set(self):
        df = self._load_data("test_set")
        df = df.reset_index()
        df['ts']=df['ts'].astype(str)
        test_set_date_list = df['ts'].tolist()
        return test_set_date_list
        
    
    def _load_config(self, config_path):        
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
            
        
    def _load_data(self, data_source):
        df = pd.read_csv(self.config["config"]["csv_paths"][data_source], sep = ";", index_col = 0)
        df.index = pd.to_datetime(df.index)
        return df
    
    
    def reset(self, seed=None, return_info=False, options=None):
        """ 
        The reset method will be called to initiate a new episode.
        reset should be called whenever a done signal has been issued

        Args:
            seed (_type_, optional): _description_. Defaults to None.
            return_info (bool, optional): _description_. Defaults to False.
            options (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """        
        
        
        #print("in reset() and logging_step = " + str(self.logging_step))

        if self.initial is False:
            
            if self.env_type == "training": 
                if self.forecast_end == self.last_slot_date_end:
                    self.lower_slot_start_boundary = self.first_slot_date_start
                
            if self.env_type == "training" or self.env_type == "test":
                # check if next slot date is in Test set 
                next_date_in_test_set = True
                while next_date_in_test_set is True: 
                    self.lower_slot_start_boundary = self.lower_slot_start_boundary  + pd.offsets.DateOffset(days=1)
                    # if date is in test set, skip the date
                    # lower_slot_start_boundary is either 22:00 or 23:00 of the previous day, so add 1 day
                    date_to_check =  self.lower_slot_start_boundary + pd.offsets.DateOffset(days=1)
                    # convert to string and only take the date digits
                    date_to_check = str(date_to_check)[0:10]                
                    if date_to_check in self.test_set_date_list:
                        next_date_in_test_set = True 
                    else:
                        next_date_in_test_set = False 
            
            if self.env_type == "eval":
                next_date_in_test_set = False
                while next_date_in_test_set is False:
                    self.lower_slot_start_boundary = self.lower_slot_start_boundary  + pd.offsets.DateOffset(days=1)
                    # if date is NOT in test set, skip the date
                    # lower_slot_start_boundary is either 22:00 or 23:00 of the previous day, so add 1 day
                    date_to_check =  self.lower_slot_start_boundary + pd.offsets.DateOffset(days=1)
                    # convert to string and only take the date digits
                    date_to_check = str(date_to_check)[0:10]  
                    if date_to_check not in self.test_set_date_list:
                        next_date_in_test_set = False
                        if (pd.to_datetime(date_to_check, utc=True)) > pd.to_datetime(self.test_set_date_list[-1], utc=True):
                            break
                    else:
                        next_date_in_test_set = True 
            
            self.gate_closure = pd.to_datetime(self.tenders_df[self.lower_slot_start_boundary:]["GATE_CLOSURE_TIME"][0])
            self.slot_start = self.tenders_df[self.lower_slot_start_boundary:].index[0]
            self.bid_submission_time = self.gate_closure - pd.offsets.DateOffset(hours = 1)
            
            logging.info("log_step: " + str(self.logging_step) + " slot: " +  "None" + " new self.lower_slot_start_boundary = " + str(self.lower_slot_start_boundary))
            logging.info("log_step: " + str(self.logging_step) + " slot: " +  "None" + " self.gate_closure = " + str(self.gate_closure))
            logging.info("log_step: " + str(self.logging_step) + " slot: " +  "None" + " self.slot_start = " + str(self.slot_start))
            logging.info("log_step: " + str(self.logging_step) + " slot: " +  "None" + " self.bid_submission_time = " + str(self.bid_submission_time))

        self.total_slot_FCR_demand = self.tenders_df[str(self.slot_start):]["total"][0] 
        self.done = False

        self.previous_activation_results = self.activation_results.copy()
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  'None'  + " self.activation_results = " + str(self.activation_results))
        self.activation_results.clear()
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  'None'  + " self.activation_results after clearing")
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  'None'  + " self.activation_results = " + str(self.activation_results))
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  'None'  + " self.previous_activation_results after clearing")
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  'None'  + " self.previous_activation_results = " + str(self.previous_activation_results))
        
        #if self.initial == False: 
            #print('self.previous_activation_results["slots_won"] = ' + str(self.previous_activation_results["slots_won"]))
            #print('self.previous_activation_results["slot_settlement_prices_DE"] = ' + str(self.previous_activation_results["slot_settlement_prices_DE"]))

        self.activation_results["slots_won"] = [None, None, None, None, None, None]
        self.activation_results["slot_settlement_prices_DE"] = [None, None, None, None, None, None]
        self.activation_results["reserved_slots"] = [None, None, None, None, None, None]
        self.activation_results["delivered_slots"] = [None, None, None, None, None, None]
        
        if self.month_change: 
            self.monthly_penalty = 0
            self.monthly_revenue = 0
            self.monthly_profit = 0
        self.month_change = False
        # reset for each episode 
        self._get_new_timestamps()
        
        # get time stamp index that is nearest to market end 
        market_end_date_index = self.market_prices_df.index.get_indexer([self.market_end], method='nearest')
        # get price df
        market_price_df =  self.market_prices_df.iloc[market_end_date_index]
        # get the price value
        self.current_daily_mean_market_price = market_price_df['price_DE'].values[0]  # type: ignore
                
        # get new observation
        observation = get_observation(self)
        
        # TRY : NO RENDERING IN RESET()
        #if self.render_mode == "human":
        #    self.render()
        
        # when first Episode is finished, set boolean.  
        self.initial = False
        
        self.logging_step += 1
        logging.error("log_step: " + str(self.logging_step) + " slot: " +  'None'  + " logging_step: " + str(self.logging_step))
        
        return observation
    
    
    def _get_info(self):
        """
        method for the auxiliary information that is returned by step and reset
        info will also contain some data that is only available inside the step method (e.g. individual reward terms)

        """        
        info = {"test_info": "info"}
        
        return info 
                
    
    def _get_new_timestamps(self):
                        
        self.historic_data_start = self.bid_submission_time - pd.offsets.DateOffset(days=self.hist_window_size)
        self.historic_data_end =  self.bid_submission_time - pd.offsets.DateOffset(minutes = 15)
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  'None'  + " self.historic_data_start = " + str(self.historic_data_start))
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  'None'  + " self.historic_data_end = " + str(self.historic_data_end))
        
        self.forecast_start = self.slot_start
        self.forecast_end = self.forecast_start + pd.offsets.DateOffset(days=self.forecast_window_size) - pd.offsets.DateOffset(minutes=15)   # type: ignore
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  'None'  + " self.forecast_start = " + str(self.forecast_start))
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  'None'  + " self.forecast_end = " + str(self.forecast_end))

        self.market_start = self.slot_start
        self.market_end = self.market_start + pd.offsets.DateOffset(hours=24) - pd.offsets.DateOffset(minutes = 15)  # type: ignore
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  'None'  + " self.market_start = " + str(self.market_start))
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  'None'  + " self.market_end = " + str(self.market_end))

        self.slot_date_list = self.tenders_df[self.market_start:][0:6].index
        
        logging.debug("self.real_month" + str(self.real_month))
        logging.debug("self.slot_start.month" + str(self.slot_start.month))  # type: ignore
       
        if self.real_month != self.slot_start.month:  # type: ignore
            self.month_change = True
            self.real_month =  self.slot_start.month # type: ignore
            self.penalty_month_count += 1
          
            logging.debug("self.penalty_month_count" + str(self.penalty_month_count))

        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  'None'  + " self.slot_date_list = " + str( self.slot_date_list))
    
    
    def _add_gaussian_noise(self, data, whole_data):
        mean = 0.0
        standard_deviation = np.std(whole_data)
        standard_deviation_gaussian = standard_deviation *  0.2 # for 20% Gaussian noise
        noise = np.random.normal(mean, standard_deviation_gaussian, size = data.shape)
        data_noisy = data + noise
        # Set negative values to 0 
        data_noisy = data_noisy.clip(min=0)

        return data_noisy 
    
    
    def step(self, action):
        """
        The step method usually contains most of the logic of your environment. 
        It accepts an action, computes the state of the environment after applying that
        action and returns the 4-tuple (observation, reward, done, info).
        Once the new state of the environment has been computed, we can check whether 
        it is a terminal state and we set done accordingly. 
        step method will not be called before reset has been called

        Args:
            action (_type_): _description_

        Returns:
            _type_: _description_
        """        
        #print("in step() and logging_step = " + str(self.logging_step))

        
        # convert action list with shape (12,) into dict
        
        bid_sizes_normalized = action[0:6]
        bid_prices_normalized = action[6:]

        bid_sizes_converted = self.size_scaler.inverse_transform(np.array(bid_sizes_normalized).reshape(-1, 1))
        bid_prices_converted = self.price_scaler.inverse_transform(np.array(bid_prices_normalized).reshape(-1, 1))

        # convert from 2d array to list 
        bid_sizes_converted = [x for xs in list(bid_sizes_converted) for x in xs]
        bid_prices_converted = [x for xs in list(bid_prices_converted) for x in xs]

        action_dict = {
            "size": bid_sizes_converted, 
            "price": bid_prices_converted
        }
        
        # Simulate VPP 
        simulate_vpp(self)
        
        # Simulate Market 
        # take the bid out of the action of the agent and resimulate the market clearing algorithm
        simulate_market(self, action_dict)
        
        # Prepare the data for the activation simulation and reward calculation
        prepare_activation(self)
        
        # calculate reward from state and action 
        step_reward, step_profit = calculate_reward(self)
        
        # update total reward
        self.total_reward += step_reward
        # Update the total profit
        self.total_profit += step_profit    
        
        info = dict(
            bid_submission_time = str(self.bid_submission_time),
            step_reward = round(float(step_reward),2),  # type: ignore
            step_profit = round(step_profit,2), # type: ignore
            total_reward = round(self.total_reward,2), # type: ignore
            total_profit = round(self.total_profit,2), # type: ignore
            total_profit_monthly = round(self.total_profit_monthly,2) # type: ignore
        )
        
        self._update_history(info)
        self.done = True
        
        observation = get_observation(self)
        info_NEW = self._get_info()
        
        if self.env_type != "test": 
            # define basic logging dict
            logging_dict = {
                "global_step": self.logging_step,
                "total_reward": self.total_reward,
                "total_profit": self.total_profit,
                "step_reward": step_reward,
                "step_profit": step_profit}
            
            if self.env_type == "training":
               
                if self.render_mode == "fast_training":
                    
                    if self.month_change:
                        logging_dict = self._log_value_on_month_change(logging_dict)

                    # logs need to be committed here as they wont be commited in render()
                    wandb.log(logging_dict,
                        #step=self.logging_step,
                        commit=True)
                
                if self.render_mode == "human":
                    if self.month_change:
                        logging_dict = self._log_value_on_month_change(logging_dict)
                    # dont commit the logs to wandb, as logs are committed in render funciton
                    wandb.log(logging_dict,
                        #step=self.logging_step,
                        commit=False)
                
            if self.render_mode == "human":
                self.render(mode="human")
            
            if self.env_type == "eval":
                
                if self.render_mode == "human":
                    if self.month_change:
                        logging_dict = self._log_value_on_month_change(logging_dict)

                    wandb.log(logging_dict,
                        #step=self.logging_step,
                        commit=False
                    )
                        
        return observation, step_reward, self.done, info
    
    
    def render(self, mode="human"): 
        render(self, mode=mode)        
        
        
    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)
        
    def _log_value_on_month_change(self, logging_dict):
        
        logging_dict["monthly_penalty"] = self.monthly_penalty
        logging_dict["monthly_revenue"] = self.monthly_revenue
        if abs(self.monthly_penalty) > self.monthly_revenue:
            self.monthly_penalty = - (self.monthly_revenue)
        self.monthly_profit = self.monthly_revenue - abs(self.monthly_penalty)
        self.total_profit_monthly += self.monthly_profit
        logging_dict["monthly_profit"] = self.monthly_profit
        logging_dict["penalty_month_count"] = self.penalty_month_count
        logging_dict["total_profit_monthly"] = self.total_profit_monthly

        logging.debug("self.month_change = True")
        logging.debug("self.monthly_penalty = " + str(self.monthly_penalty))
        logging.debug("self.monthly_revenue = " + str(self.monthly_revenue))
        logging.debug("self.monthly_profit = " + str(self.monthly_profit))
        logging.debug("self.total_profit_monthly = " + str(self.total_profit_monthly))
        logging.debug("self.penalty_month_count = " + str(self.penalty_month_count))
        
        return logging_dict
    
    
    def _convert_number_to_float(self, x): 
        return np.float32(x)