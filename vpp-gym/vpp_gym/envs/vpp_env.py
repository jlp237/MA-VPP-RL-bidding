# Basics
import os
import random
import warnings

# Data 
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import numpy as np
from collections import OrderedDict
from functools import reduce
import json
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

# Logging
import logging
import wandb

# Plotting
import plotly.express as px
import plotly.graph_objects as go

# RL
from gym import Env
from gym.spaces import Box, Dict

# Auxiliary 
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

class VPPBiddingEnv(Env):
    
    def __init__(self,
                 config_path,
                 log_level, 
                 env_type,
                 render_mode="human"
                ):
        
        logger = logging.getLogger()
        while logger.hasHandlers():
                logger.removeHandler(logger.handlers[0])
        
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
            fhandler = logging.StreamHandler()
            
        logger.addHandler(fhandler)           
        logger.setLevel(log_level)
        
        logging.debug("log level = debug")
        logging.info("log level = info")
        logging.warning("log level = warning")
        
        # data 
        self.config = self._load_config(config_path)
            
        self.renewables_df = self._load_data("renewables")
        self.tenders_df = self._load_data("tenders")
        self.market_results = self._load_data("market_results") 
        self.bids_df = self._load_data("bids") 
        self.time_features_df = self._load_data("time_features") 
        self.market_prices_df = self._load_data("market_prices") 
        self.test_set_date_list = self._load_test_set()
        
        self.asset_data , self.asset_data_FCR = self._configure_vpp()
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
        
        logging.debug("selection self.renewables_df" + str(self.renewables_df))
        logging.debug("selection self.tenders_df" + str(self.tenders_df))
        logging.debug("selection self.market_results" + str(self.market_results))
        logging.debug("selection self.bids_df" + str(self.bids_df))
        logging.debug("selection self.time_features_df" + str(self.time_features_df))

        # slot start , gate closure, auction time
        self.lower_slot_start_boundary = self.first_slot_date_start
        self.gate_closure = pd.to_datetime(self.tenders_df[self.lower_slot_start_boundary:]["GATE_CLOSURE_TIME"][0])
        self.slot_start = self.tenders_df[self.lower_slot_start_boundary:].index[0]
        self.bid_submission_time = self.gate_closure - pd.offsets.DateOffset(hours = 1)
        
        logging.debug("initial self.first_slot_date_start = " + str(self.first_slot_date_start))
        logging.debug("initial self.lower_slot_start_boundary = " + str(self.lower_slot_start_boundary))
        logging.debug("initial self.slot_start = " + str(self.slot_start))
        logging.debug("initial self.bid_submission_time = " + str(self.bid_submission_time))
        
        self.initial = True
        self.done = None
        self.total_reward = 0.
        self.total_profit = 0.
        self.violation_counter = 0
        self.history = None
        self.current_daily_mean_market_price = 0.
        
        # Slots 
        #self.slots_won = [0, 0, 0, 0, 0, 0]
        #self.slot_settlement_prices_DE = [0., 0., 0., 0., 0., 0.]
        
        self.activation_results = {}
        self.previous_activation_results  = {}
        
        self.logging_step = -1
                        
            
        # Spaces
        
        # Observation Space
        #obs_low = np.float32(np.array([0.0] * 96)) #96 timesteps to min 0.0
        #obs_high = np.float32(np.array([1.0] * 96)) #96 timesteps to max 1.0
       
        #obs_high = np.float32(np.array([self.maximum_possible_VPP_capacity] * 96)) #96 timesteps to max 1.0

        
        # Action Space
        
        # VERSION 3
        # Convert complex action space to flattended space
        
        # maximum possible FCR capacity 
        #maximum_possible_FCR_capacity = round(self.asset_data_FCR_total.max(),2)
        #maximum_possible_market_price = self.bids_df["settlement_price"].max()
        
        #TODO: DELETE NEXT LINE 
        #maximum_possible_market_price = 100.0
        
        # 12 values from  min 0.0
        action_low = np.float32(np.array([-1.0] * 12)) 
        # 6 values to max maximum_possible_FCR_capacity = the bid sizes 
        # 6 values to max maximum_possible_market_price = the bid prices
        #action_high = np.float32(np.array([maximum_possible_FCR_capacity] * 6 + [maximum_possible_market_price] *6 )) 
        #action_high = np.float32(np.array([self.maximum_possible_VPP_capacity] * 6 + [maximum_possible_market_price] *6 )) 
        action_high = np.float32(np.array([1.0] * 6 + [1.0] * 6 )) 

        self.action_space = Box(low=action_low, high=action_high, shape=(12,), dtype=np.float32)
        
        # VERSION 2 : Box 
        
        '''# Convert complex action space to flattended space
        # bid sizes =  6 DISCRETE slots from 0 to 25  = [ 25, 25, 25, 25, 25 , 25]  = in flattened = 150 values [0,1]
        # bid prizes = 6 CONTINUOUS slots from 0 to 100  = [ 100., 100., 100., 100., 100. , 100.]  = in flattened = 150 values [0,1]

        # 156 values from  min 0.0
        action_low = np.float32(np.array([0.0] * 156)) 
        #150 values to max 1.0 = the bid sizes 
        # +6 values to max 100. = the bid prices
        action_high = np.float32(np.array([1.0] * 150 + [100.0]*6)) 
        self.action_space = Box(low=action_low, high=action_high, shape=(156,), dtype=np.float32)'''

        # VERSION 1
        
        
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
        self.size_scaler.fit(self.asset_data_total.values.reshape(-1, 1))
        logging.debug("Action Space Normalization: size_scaler min = " + str(self.size_scaler.data_min_) + " and max = " + str(self.size_scaler.data_max_))
        
        self.price_scaler = MinMaxScaler(feature_range=(-1,1))
        self.price_scaler.fit(self.bids_df["settlement_price"].values.reshape(-1, 1))
        logging.debug("Action Space Normalization: price_scaler min = " + str(self.price_scaler.data_min_) + " and max = " + str(self.price_scaler.data_max_))
        
        ### Normalization of Observation Space
        self.asset_data_historic_scaler = MinMaxScaler(feature_range=(-1,1))
        self.asset_data_historic_scaler.fit(self.asset_data_total.values.reshape(-1, 1))
        
        noisy_asset_data_forecast_for_fit = self._add_gaussian_noise(self.asset_data_total.to_numpy(dtype=np.float32), self.asset_data_total.to_numpy(dtype=np.float32))
        self.noisy_asset_data_forecast_scaler = MinMaxScaler(feature_range=(-1,1))
        self.noisy_asset_data_forecast_scaler.fit(noisy_asset_data_forecast_for_fit.reshape(-1, 1))
        
        # predicted_market_prices_scaler not needed, as self.price_scaler from action space can be used 
        
        self.weekday_scaler = MinMaxScaler(feature_range=(-1,1))
        self.weekday_scaler.fit(self.time_features_df["weekday"].values.reshape(-1, 1))
        
        self.week_scaler = MinMaxScaler(feature_range=(-1,1))
        self.week_scaler.fit(self.time_features_df["week"].values.reshape(-1, 1))

        self.month_scaler = MinMaxScaler(feature_range=(-1,1))
        self.month_scaler.fit(self.time_features_df["month"].values.reshape(-1, 1))

        self.bool_scaler = MinMaxScaler(feature_range=(-1,1))
        self.bool_scaler.fit(np.array([0,1]).reshape(-1, 1))
        
        self.slot_settlement_prices_DE_scaler = MinMaxScaler(feature_range=(-1,1))
        self.slot_settlement_prices_DE_scaler.fit(np.array([0.0, 4257.07]).reshape(-1, 1))
        
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

        self.observation = None
        
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


    def _configure_vpp(self):
        # list to concat all dfs later on
        asset_frames_total = []
        asset_frames_FCR = []
        # for each asset type defined in the config (e.g.: "hydro", "wind")
        for asset_type in self.config["assets"].keys():
            # for every plant configuration there is per asset type
            for plant_config in range(len(self.config["assets"][asset_type])):
                # get the qunatity of plants
                quantity = self.config["assets"][asset_type][plant_config]["quantity"]
                # get the maximum capacity of these plants 
                max_capacity_MW = self.config["assets"][asset_type][plant_config]["max_capacity_MW"]
                max_FCR_capacity_share = self.config["assets"][asset_type][plant_config]["max_FCR_capacity_share"]
                # get the name of the column in the renewables csv
                asset_column_names = self.config["assets"][asset_type][plant_config]["asset_column_names"]
                # initialize a array with zeros and the length of the renewables dataframe
                total_asset_capacity = np.array([0.0] * len(self.renewables_df))
                total_asset_FCR_capacity = np.array([0.0] * len(self.renewables_df))

                i = 1
                while i < quantity: 
                    for asset_column_name in (asset_column_names):
                        asset_data = self.renewables_df[[asset_column_name]].values.flatten()
                        asset_data *= max_capacity_MW
                        asset_FCR_capacity = asset_data * max_FCR_capacity_share
                        
                        total_asset_FCR_capacity += asset_FCR_capacity 
                        total_asset_capacity += asset_data
                        
                        i += 1
                        if i < quantity:
                            continue
                        else: 
                            break                        
                    
                total_df = pd.DataFrame(index=self.renewables_df.index)
                FCR_df = pd.DataFrame(index=self.renewables_df.index)
                
                total_df[asset_type + "_class_" + str(plant_config)] = total_asset_capacity
                FCR_df[asset_type + "_class_" + str(plant_config)] = total_asset_FCR_capacity
                asset_frames_total.append(total_df)
                asset_frames_FCR.append(FCR_df)

        if not asset_frames_total: 
            logging.error("No asset data found")
        all_asset_data = reduce(lambda x, y: pd.merge(x, y, on = "time"), asset_frames_total)
        all_asset_data_FCR = reduce(lambda x, y: pd.merge(x, y, on = "time"), asset_frames_FCR)

        all_asset_data['Total'] = all_asset_data.iloc[:,:].sum(axis=1)
        all_asset_data_FCR['Total'] = all_asset_data_FCR.iloc[:,:].sum(axis=1)
        
        return all_asset_data, all_asset_data_FCR
    
    
    
    def reset(self, seed=None, return_info=False, options=None):
        
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
            
            logging.info("new self.lower_slot_start_boundary = " + str(self.lower_slot_start_boundary))
            logging.info("self.gate_closure = " + str(self.gate_closure))
            logging.info("self.slot_start = " + str(self.slot_start))
            logging.info("self.bid_submission_time = " + str(self.bid_submission_time))

        self.total_slot_FCR_demand = self.tenders_df[str(self.slot_start):]["total"][0] 
        self.done = False

        self.previous_activation_results = self.activation_results.copy()
        logging.debug("self.activation_results = " + str(self.activation_results))
        self.activation_results.clear()
        logging.debug("self.activation_results after clearing")
        logging.debug("self.activation_results = " + str(self.activation_results))
        logging.debug("self.previous_activation_results after clearing")
        logging.debug("self.previous_activation_results = " + str(self.previous_activation_results))
        
        self.activation_results["slots_won"] = [0, 0, 0, 0, 0, 0]
        self.activation_results["slot_settlement_prices_DE"] = [0., 0., 0., 0., 0., 0.]
        
        # reset for each episode 
        self._get_new_timestamps()
        
        self.current_daily_mean_market_price = self.market_prices_df.iloc[self.market_prices_df.index.get_indexer([self.market_end], method='nearest')]['price_DE'].values[0]

        # get new observation
        self._get_observation()
        
        # when first Episode is finished, set boolean.  
        self.initial = False
        
        self.logging_step += 1
        logging.debug("logging_step: " + str(self.logging_step))
        
        return self.observation
                
    
    def _get_new_timestamps(self):
                
        self.historic_data_start = self.bid_submission_time - pd.offsets.DateOffset(days=self.hist_window_size)
        self.historic_data_end =  self.bid_submission_time - pd.offsets.DateOffset(minutes = 15)
        logging.debug("self.historic_data_start = " + str(self.historic_data_start))
        logging.debug("self.historic_data_end = " + str(self.historic_data_end))
        
        self.forecast_start = self.slot_start
        self.forecast_end = self.forecast_start + pd.offsets.DateOffset(days=self.forecast_window_size) - pd.offsets.DateOffset(minutes=15) 
        logging.debug("self.forecast_start = " + str(self.forecast_start))
        logging.debug("self.forecast_end = " + str(self.forecast_end))

        self.market_start = self.slot_start
        self.market_end = self.market_start + pd.offsets.DateOffset(hours=24) - pd.offsets.DateOffset(minutes = 15)
        logging.debug("self.market_start = " + str(self.market_start))
        logging.debug("self.market_end = " + str(self.market_end))

        self.slot_date_list = self.tenders_df[self.market_start:][0:6].index
        
        '''self.slot_date_list = []
        slot_date = self.market_start 
        for i in range(0,6):
            self.slot_date_list.append(str(slot_date))
            slot_date = slot_date + pd.offsets.DateOffset(hours=4)  '''
            
        logging.debug("self.slot_date_list = " + str( self.slot_date_list))
    
    
    def _add_gaussian_noise(self, data, whole_data):
        mean = 0.0
        standard_deviation = np.std(whole_data)
        standard_deviation_gaussian = standard_deviation *  0.2 # for 20% Gaussian noise
        noise = np.random.normal(mean, standard_deviation_gaussian, size = data.shape)
        data_noisy = data + noise
        # Set negative values to 0 
        data_noisy = data_noisy.clip(min=0)

        return data_noisy 
    
    
    def _get_observation(self):
        
        '''if (self.done is False) and (self.initial is False):
            print("if schleife 1 ")
            print("done = " + str(self.done))
            print("initial = " + str(self.initial))
            
            self.observation["slots_won"] = np.array(self.activation_results["slots_won"], dtype=np.int32)
            self.observation["slot_settlement_prices_DE"] = np.array(self.activation_results["slot_settlement_prices_DE"], dtype=np.float32)
            
            
        if (self.done is True) or (self.initial is True):
            print("if schleife 2 ")
            print("done = " + str(self.done))
            print("initial = " + str(self.initial))'''
    
        asset_data_historic = self.asset_data_total[str(self.historic_data_start) : str(self.historic_data_end)].to_numpy(dtype=np.float32)
        logging.debug("asset_data_historic = " + str(self.asset_data_total[str(self.historic_data_start) : str(self.historic_data_end)]) )
        # normalize the data 
        asset_data_historic_norm = self.asset_data_historic_scaler.transform((asset_data_historic.reshape(-1, 1)))
        # convert from 2D to 1D array 
        asset_data_historic_norm = asset_data_historic_norm.flatten()
        #asset_data_historic_norm = [x for xs in list(asset_data_historic_norm) for x in xs]

        asset_data_forecast = self.asset_data_total[str(self.forecast_start) : str(self.forecast_end)].to_numpy(dtype=np.float32)
        logging.debug("asset_data_forecast = "  + str(self.asset_data_total[str(self.forecast_start) : str(self.forecast_end)]))
        # add gaussian noise to data
        noisy_asset_data_forecast = self._add_gaussian_noise(asset_data_forecast, self.asset_data_total)
        noisy_asset_data_forecast = noisy_asset_data_forecast.astype(np.float32)
        logging.debug("noisy_asset_data_forecast = "  + str(noisy_asset_data_forecast))
        noisy_asset_data_forecast_norm = self.noisy_asset_data_forecast_scaler.transform((noisy_asset_data_forecast.reshape(-1, 1)))
        #noisy_asset_data_forecast_norm = [x for xs in list(noisy_asset_data_forecast_norm) for x in xs]
        noisy_asset_data_forecast_norm = noisy_asset_data_forecast_norm.flatten()

        # for predicted market Prices try naive prediction: retrieve price of same day last week 
        market_start_last_week = self.market_start - pd.offsets.DateOffset(days=7) 
        market_end_last_week = self.market_end - pd.offsets.DateOffset(days=7)
        logging.debug("market_start_last_week = "  + str(market_start_last_week))
        logging.debug("market_end_last_week = "  + str(market_end_last_week))
        predicted_market_prices = self.market_results["DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"][str(market_start_last_week) : str(market_end_last_week)].to_numpy(dtype=np.float32)
        logging.debug("predicted_market_prices = "  + str(predicted_market_prices))
        if len(predicted_market_prices) < 6:
            # predicted_market_prices list is smaller than 6 so fake is generated mean of first week
            predicted_market_prices = np.array([ 17.48, 17.48, 17.48, 17.48, 17.48, 17.48], dtype=np.float32) 
            logging.debug("predicted_market_prices list is smaller than 6 so fake is generated: "  + str(predicted_market_prices))
        predicted_market_prices_norm = self.price_scaler.transform((np.array(predicted_market_prices).reshape(-1, 1)))
        #predicted_market_prices_norm = [x for xs in list(predicted_market_prices_norm) for x in xs]
        predicted_market_prices_norm = predicted_market_prices_norm.flatten()
        
        time_features = self.time_features_df[str(self.market_start) : str(self.market_end)]
        logging.debug(self.time_features_df[str(self.market_start) : str(self.market_end)])

        weekday = int(time_features["weekday"][0])
        weekday_norm = self.weekday_scaler.transform(np.array(weekday).reshape(-1, 1))
        weekday_norm = weekday_norm.flatten().astype('float32')
        
        week = int(time_features["week"][0])
        week_norm = self.week_scaler.transform(np.array(week).reshape(-1, 1))
        week_norm = week_norm.flatten().astype('float32')
        
        month = int(time_features["month"][0])
        month_norm = self.month_scaler.transform(np.array(month).reshape(-1, 1))
        month_norm = month_norm.flatten().astype('float32')
        
        isHoliday = int(time_features["is_holiday"][0])
        isHoliday_norm = self.bool_scaler.transform(np.array(isHoliday).reshape(-1, 1))
        isHoliday_norm = isHoliday_norm.flatten().astype('float32')
        
        followsHoliday = int(time_features["followsHoliday"][0])
        followsHoliday_norm = self.bool_scaler.transform(np.array(followsHoliday).reshape(-1, 1))
        followsHoliday_norm = followsHoliday_norm.flatten().astype('float32')
        
        priorHoliday = int(time_features["priorHoliday"][0])
        priorHoliday_norm = self.bool_scaler.transform(np.array(priorHoliday).reshape(-1, 1))
        priorHoliday_norm = priorHoliday_norm.flatten().astype('float32')
        
        slots_won =  np.array(self.activation_results["slots_won"], dtype=np.int32)
        slots_won_norm = self.bool_scaler.transform(np.array(slots_won).reshape(-1, 1))
        slots_won_norm = slots_won_norm.flatten().astype('float32')
        
        slot_settlement_prices_DE = np.array(self.activation_results["slot_settlement_prices_DE"], dtype=np.float32)
        slot_settlement_prices_DE_norm = self.slot_settlement_prices_DE_scaler.transform((slot_settlement_prices_DE.reshape(-1, 1)))
        #slot_settlement_prices_DE_norm = [x for xs in list(slot_settlement_prices_DE_norm) for x in xs]
        slot_settlement_prices_DE_norm = slot_settlement_prices_DE_norm.flatten().astype('float32')

        self.observation = OrderedDict({
            "asset_data_historic": asset_data_historic_norm,
            "asset_data_forecast": noisy_asset_data_forecast_norm,
            "predicted_market_prices": predicted_market_prices_norm,
            "weekday": weekday_norm, 
            "week": week_norm, 
            "month": month_norm,
            "isHoliday": isHoliday_norm, 
            "followsHoliday": followsHoliday_norm,
            "priorHoliday": priorHoliday_norm,
            "slots_won": slots_won_norm,
            "slot_settlement_prices_DE": slot_settlement_prices_DE_norm
            })
        
        logging.debug("NEW Observation = "  + str(self.observation))
            
    
    
    def step(self, action):
        
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
        self._simulate_vpp()
        
        # Simulate Market 
        # take the bid out of the action of the agent and resimulate the market clearing algorithm
        self._simulate_market(action_dict)
        
        # Prepare the data for the activation simulation and reward calculation
        self._prepare_activation()
        
        # calculate reward from state and action 
        step_reward, step_profit = self._calculate_reward()
        
        self.total_reward += step_reward
            
        info = dict(
            bid_submission_time = str(self.bid_submission_time),
            step_reward = round(step_reward,2),
            step_profit = round(step_profit,2),
            total_reward = round(self.total_reward,2),
            total_profit = round(self.total_profit,2)
        )
        
        
        self._update_history(info)
                
        self.done = True
        self._get_observation()
        
        
        if self.env_type != "test":
            
            if self.env_type == "training":
                wandb.log({
                    "global_step": self.logging_step,
                    "total_reward": self.total_reward,
                    "total_profit": self.total_profit,
                    "step_reward": step_reward,
                    "step_profit": step_profit},
                    #step=self.logging_step,
                    commit=False)
                
                self.render()
            
            if self.env_type == "eval":
                wandb.log({
                    "global_step": self.logging_step,
                    "total_reward": self.total_reward,
                    "total_profit": self.total_profit,
                    "step_reward": step_reward,
                    "step_profit": step_profit},
                    #step=self.logging_step,
                    commit=False
                )
        
        return self.observation, step_reward, self.done, info
    
    def _calculate_reward(self):        
        # Step 1 of Reward Function: The Auction
        # did the agent win the auction? 
        # what was the revenue ?
        
        self.activation_results["total_not_reserved_energy"] = [0, 0, 0, 0, 0, 0]
        self.activation_results["total_not_delivered_energy"] = [0, 0, 0, 0, 0, 0]

        step_profit = 0
        total_step_reward = 0
        
        # per slot won: + 100
        # per slot won: + (bid size *  marginal prize)
        # per slot lost: -100

        logging.info("Reward Overview:")
        logging.info("self.activation_results['slots_won']: " + str(self.activation_results["slots_won"]))
        logging.info("len(self.activation_results['slots_won']) : "  + str(len(self.activation_results["slots_won"])))       
        
        for slot in range(0, len(self.activation_results["slots_won"])):
            
            step_reward = 0
            
            logging.info("slot no. " + str(slot))
            
            if self.activation_results["slots_won"][slot] == 0:
                logging.info("slot no " + str(slot) + " was lost")
                slot_settlement_price = self.activation_results["slot_settlement_prices_DE"][slot]
                
                agents_bid_price = self.activation_results["agents_bid_prices"][slot]

                distance_to_settlement_price = agents_bid_price - slot_settlement_price
                logging.info("distance_to_settlement_price = " + str(distance_to_settlement_price))
                logging.info("self.price_scaler.data_max_[0] = " + str(self.price_scaler.data_max_[0]))

                step_reward = 1 - (distance_to_settlement_price / self.price_scaler.data_max_[0] )**0.4
                logging.info("step_reward = " + str(step_reward))
                
                #step_reward -= 0

            if self.activation_results["slots_won"][slot] == 1:
                logging.info("slot no. " + str(slot)+  " was won")

                # Approach 1 : first reward the won slot, then check if it could be activated and give huge negative reward (-1000)
                # Approach 2 : first check if won slot could be activated and then calculate partial reward (60 minutes - penalty minutes / 60 ) * price * size 
                # we try Approach 1 
    
                # Step 1: award the agent for a won slot
                step_reward += 1
                
                # Step 2: Calculate the Profit of the bid if won 
                
                # extract the bid size of the agent 
                agents_bid_size = self.activation_results["agents_bid_sizes_round"][slot]
                # and calculate the reward by multiplying the bid size with the settlement price of the slot
                basic_compensation = (agents_bid_size * self.activation_results["slot_settlement_prices_DE"][slot])
                logging.info("basic_compensation: " + str(basic_compensation))
                #step_reward += basic_compensation
                step_profit += basic_compensation
                
                # Step 3.1: simulate reservation: validate if the VPP can reserve the traded capacity
                total_reservation_possible = self._simulate_reservation(slot)
                
                if total_reservation_possible == False:
                    # Penalty Calculation from "MfRRA"
                    # NO penalty calculation from Elia PDF: "20200317 TC BSP FCRFINAL ConsultEN.pdf"
                    # ID AEP based on: https://www.regelleistung.net/ext/static/konsultation-aep?lang=de
                    # based on :  Preisindex orientiert sich stark am Intraday-Viertelstundenhandel: https://www.next-kraftwerke.de/wissen/ausgleichsenergie#:~:text=Der%20Ausgleichsenergiepreis%20%E2%80%93%20%22reBAP%22%20(,auf%20die%20Verursacher%20der%20Regelenergie.
                    # Marktdaten von: https://www.smard.de/home/marktdaten?marketDataAttributes=%7B%22resolution%22:%22month%22,%22region%22:%22Amprion%22,%22from%22:1589172592929,%22to%22:1654663792928,%22moduleIds%22:%5B8004169%5D,%22selectedCategory%22:17,%22activeChart%22:true,%22style%22:%22color%22,%22categoriesModuleOrder%22:%7B%7D%7D
                    penalty_fee_1 = self.current_daily_mean_market_price * 1.25
                    penalty_fee_2 = self.current_daily_mean_market_price + 10.0
                    penalty_fee_3 = self.activation_results["slot_settlement_prices_DE"][slot]
                    logging.info("penalty_fee_1: " + str(penalty_fee_1))
                    logging.info("penalty_fee_2: " + str(penalty_fee_2))
                    logging.info("penalty_fee_3: " + str(penalty_fee_3))
                    penalty_list = [penalty_fee_1, penalty_fee_2, penalty_fee_3] 
                
                    penalty_fee_reservation = self.activation_results["total_not_reserved_energy"][slot] * max(penalty_list) 
                    logging.debug("penalty_fee_reservation = " + str(penalty_fee_reservation))
                    step_reward -= 0
                    step_profit -= penalty_fee_reservation
                else: 
                    # give reward when capacity could be reserved
                    step_reward += 1
                                      
                # Step 3.2: simulate activation: validate if the VPP can deliver the traded capacity
                self._simulate_activation(slot)
                logging.info("self.activation_results['delivered_slots']")
                logging.info(self.activation_results["delivered_slots"])

                # Step 4: if the capacity can not be delivered give a high Penalty
                if self.activation_results["delivered_slots"][slot] == False:
                    penalty_fee_1 = self.current_daily_mean_market_price * 1.25
                    penalty_fee_2 = self.current_daily_mean_market_price + 10.0
                    penalty_fee_3 = self.activation_results["slot_settlement_prices_DE"][slot]
                    logging.info("penalty_fee_1: " + str(penalty_fee_1))
                    logging.info("penalty_fee_2: " + str(penalty_fee_2))
                    logging.info("penalty_fee_3: " + str(penalty_fee_3))
                    penalty_list = [penalty_fee_1, penalty_fee_2, penalty_fee_3] 
                
                    penalty_fee_activation = self.activation_results["total_not_delivered_energy"][slot] * max(penalty_list) 
                    logging.info("penalty_fee_activation = " + str(penalty_fee_activation))
                    step_reward -= 0
                    step_profit -= penalty_fee_activation
                    #step_reward -= 5000
                else:
                    # give reward when capacity could be activated
                    step_reward += 1

                
                # Update the total profit and Step Reward. 
                self._update_profit(step_profit)
                
                #step_reward +=  step_profit
                
                # create weighted step reward 
                
                weighted_step_reward = step_reward / 3
                
                total_step_reward += weighted_step_reward
                
                logging.info("agents_bid_size: " + str(agents_bid_size))
                logging.info("self.activation_results['slot_settlement_prices_DE'][slot]: " + str(self.activation_results["slot_settlement_prices_DE"][slot]))
                logging.info("step_profit: " + str(step_profit))
                logging.info("weighted_step_reward Slot " + str(slot) +" = " + str(weighted_step_reward))
                        
        # further rewards? 
        # diff to the settlement price
        # diff to the max. forecasted capacity of the VPP
        # incentive to go nearer to settlement price or forecasted capacity can be: 1- (abs(diff_to_capacity)/max_diff_to_capacity)^0.5
        # idea: reward for positive and negative reward separate. 
        
        # Alternative solution: 
        # A reward function, that combines penalty and delivered FCR: 
        # compensation = (60 minutes - penalty minutes / 60 ) * price * size 
        # penalty  = (penalty minutes / 60 ) * price * size 
        # reputation_damage = reputation_factor *  penalty_min/ 60 * size
            # penalty_min = number of minutes where capacity could not be provided
        # in total: r = compensation − penalty − reputation_damage,
        
        total_weighted_step_reward = total_step_reward / 6
        
        logging.info(" total_weighted_step_reward for all 6 slots : " + str(total_weighted_step_reward))

        return total_weighted_step_reward, step_profit
    
    
    def _calculate_reward_OLD(self):        
        # Step 1 of Reward Function: The Auction
        # did the agent win the auction? 
        # what was the revenue ?
        
        self.activation_results["total_not_reserved_energy"] = [0, 0, 0, 0, 0, 0]
        self.activation_results["total_not_delivered_energy"] = [0, 0, 0, 0, 0, 0]

        step_reward = 0
        step_profit = 0
        
        # per slot won: + 100
        # per slot won: + (bid size *  marginal prize)
        # per slot lost: -100

        logging.info("Reward Overview:")
        logging.info("self.activation_results['slots_won']: " + str(self.activation_results["slots_won"]))
        logging.info("len(self.activation_results['slots_won']) : "  + str(len(self.activation_results["slots_won"])))       
        
        for slot in range(0, len(self.activation_results["slots_won"])):
            
            logging.info("slot no. " + str(slot))
            
            if self.activation_results["slots_won"][slot] == 0:
                logging.info("slot no " + str(slot) + " was lost")
                step_reward -= 1000

            if self.activation_results["slots_won"][slot] == 1:
                logging.info("slot no. " + str(slot)+  " was won")

                # Approach 1 : first reward the won slot, then check if it could be activated and give huge negative reward (-1000)
                # Approach 2 : first check if won slot could be activated and then calculate partial reward (60 minutes - penalty minutes / 60 ) * price * size 
                # we try Approach 1 
    
                # Step 1: award the agent for a won slot
                step_reward += 10000
                
                # Step 2: Calculate the Profit of the bid if won 
                
                # extract the bid size of the agent 
                agents_bid_size = self.activation_results["agents_bid_sizes_round"][slot]
                # and calculate the reward by multiplying the bid size with the settlement price of the slot
                basic_compensation = (agents_bid_size * self.activation_results["slot_settlement_prices_DE"][slot])
                logging.info("basic_compensation: " + str(basic_compensation))
                step_reward += basic_compensation
                step_profit += basic_compensation
                
                # Step 3.1: simulate reservation: validate if the VPP can reserve the traded capacity
                total_reservation_possible = self._simulate_reservation(slot)
                
                if total_reservation_possible == False:
                    # Penalty Calculation from "MfRRA"
                    # NO penalty calculation from Elia PDF: "20200317 TC BSP FCRFINAL ConsultEN.pdf"
                    # ID AEP based on: https://www.regelleistung.net/ext/static/konsultation-aep?lang=de
                    # based on :  Preisindex orientiert sich stark am Intraday-Viertelstundenhandel: https://www.next-kraftwerke.de/wissen/ausgleichsenergie#:~:text=Der%20Ausgleichsenergiepreis%20%E2%80%93%20%22reBAP%22%20(,auf%20die%20Verursacher%20der%20Regelenergie.
                    # Marktdaten von: https://www.smard.de/home/marktdaten?marketDataAttributes=%7B%22resolution%22:%22month%22,%22region%22:%22Amprion%22,%22from%22:1589172592929,%22to%22:1654663792928,%22moduleIds%22:%5B8004169%5D,%22selectedCategory%22:17,%22activeChart%22:true,%22style%22:%22color%22,%22categoriesModuleOrder%22:%7B%7D%7D
                    penalty_fee_1 = self.current_daily_mean_market_price * 1.25
                    penalty_fee_2 = self.current_daily_mean_market_price + 10.0
                    penalty_fee_3 = self.activation_results["slot_settlement_prices_DE"][slot]
                    logging.info("penalty_fee_1: " + str(penalty_fee_1))
                    logging.info("penalty_fee_2: " + str(penalty_fee_2))
                    logging.info("penalty_fee_3: " + str(penalty_fee_3))
                    penalty_list = [penalty_fee_1, penalty_fee_2, penalty_fee_3] 
                
                    penalty_fee_reservation = self.activation_results["total_not_reserved_energy"][slot] * max(penalty_list) 
                    logging.info("penalty_fee_reservation = " + str(penalty_fee_reservation))
                    step_reward -= penalty_fee_reservation
                    step_profit -= penalty_fee_reservation  
                else: 
                    # give reward when capacity could be reserved
                    step_reward += 10000
                                      
                # Step 3.2: simulate activation: validate if the VPP can deliver the traded capacity
                self._simulate_activation(slot)
                logging.info("self.activation_results['delivered_slots']")
                logging.info(self.activation_results["delivered_slots"])

                # Step 4: if the capacity can not be delivered give a high Penalty
                if self.activation_results["delivered_slots"][slot] == False:
                    penalty_fee_1 = self.current_daily_mean_market_price * 1.25
                    penalty_fee_2 = self.current_daily_mean_market_price + 10.0
                    penalty_fee_3 = self.activation_results["slot_settlement_prices_DE"][slot]
                    logging.info("penalty_fee_1: " + str(penalty_fee_1))
                    logging.info("penalty_fee_2: " + str(penalty_fee_2))
                    logging.info("penalty_fee_3: " + str(penalty_fee_3))
                    penalty_list = [penalty_fee_1, penalty_fee_2, penalty_fee_3] 
                
                    penalty_fee_activation = self.activation_results["total_not_delivered_energy"][slot] * max(penalty_list) 
                    logging.info("penalty_fee_activation = " + str(penalty_fee_activation))
                    step_reward -= penalty_fee_activation
                    step_profit -= penalty_fee_activation
                    #step_reward -= 5000
                else:
                    # give reward when capacity could be activated
                    step_reward += 10000

                
                # Update the total profit and Step Reward. 
                self._update_profit(step_profit)
                #step_reward +=  step_profit
                
                logging.info("agents_bid_size: " + str(agents_bid_size))
                logging.info("self.activation_results['slot_settlement_prices_DE'][slot]: " + str(self.activation_results["slot_settlement_prices_DE"][slot]))
                logging.info("step_profit: " + str(step_profit))
                logging.info("step_reward Slot " + str(slot) +" = " + str(step_reward))
                        
        # further rewards? 
        # diff to the settlement price
        # diff to the max. forecasted capacity of the VPP
        # incentive to go nearer to settlement price or forecasted capacity can be: 1- (abs(diff_to_capacity)/max_diff_to_capacity)^0.5
        # idea: reward for positive and negative reward separate. 
        
        # Alternative solution: 
        # A reward function, that combines penalty and delivered FCR: 
        # compensation = (60 minutes - penalty minutes / 60 ) * price * size 
        # penalty  = (penalty minutes / 60 ) * price * size 
        # reputation_damage = reputation_factor *  penalty_min/ 60 * size
            # penalty_min = number of minutes where capacity could not be provided
        # in total: r = compensation − penalty − reputation_damage,
        logging.info("total step_reward for all 6 slots : " + str(step_reward))

        return step_reward, step_profit
    
    
    def _update_profit(self, step_profit):
        self.total_profit += step_profit
        logging.debug("self.total_profit = " + str(self.total_profit))
        
    
    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

            
    def render(self):
        if not self.activation_results:
            logging.debug("self.activation_results is empty, not plotting it ")
        else:
            # only plot to wandb when not in test mode
            if self.env_type != "test":
                if self.logging_step > 0: 
                    logging.debug(" now in render()")        
                    logging.debug(" self.previous_activation_results " + str(self.previous_activation_results))      
                    logging.debug(" self.activation_results['slots_won'] " + str(self.activation_results["slots_won"]))      

                    # Render Won / Lost Slots 
                    slots_won = self.previous_activation_results["slots_won"]
                    logging.debug(" slots_won " + str(slots_won))      
                    slots_lost = [None,None,None,None,None,None]
                    for x in range(len(slots_won)):
                        if slots_won[x] == 1:
                            slots_lost[x] = 0
                        else:
                            slots_lost[x] = 1

                    data = {'Slot Won': slots_won, 'Slot Lost': slots_lost}
                    slots_df = pd.DataFrame(data=data, index=[1, 2, 3, 4, 5, 6])
                    logging.debug(" slots_df " + str(slots_df))
                    slots_won_plot = px.bar(slots_df,  x= slots_df.index, y=['Slot Won', 'Slot Lost'], color_discrete_sequence=[ "green", "gainsboro"] )

                    # Render activation for Capacity 
                    activation_plot = go.Figure()
                    activation_plot.add_trace(go.Scatter(x=list(range(1, 97)), y=self.previous_activation_results["vpp_total"], fill='tozeroy', fillcolor='rgba(0, 85, 255, 0.4)',  line_color="blue", name="VPP Cap."))
                    #activation_plot.add_trace(go.Scatter(x=list(range(1, 97)), y=self.previous_activation_results["vpp_total_FCR"], fill='tozeroy', line_color="green", name="VPP FCR Cap." )) 
                    activation_plot.add_trace(go.Scatter(x=list(range(1, 97)), y=self.previous_activation_results["bid_sizes_all_slots"], fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.5)', line_color="red", name="Agents Bid" )) 

                    # Render activation for each Slot 
                    slots_delivered = [None,None,None,None,None,None]
                    for slot in range(6):
                        if self.previous_activation_results["delivered_slots"][slot] == True:
                            slots_delivered[slot] = 1
                        else: 
                            slots_delivered[slot] = 0
                    slots_not_delivered = [None,None,None,None,None,None]
                    for x in range(len(slots_delivered)):
                        if slots_delivered[x] == 1:
                            slots_not_delivered[x] = 0
                        else:
                            slots_not_delivered[x] = 1
                            
                    data = {'delivered': slots_delivered, 'NOT deliv.': slots_not_delivered}
                    slots_delivered_df = pd.DataFrame(data=data, index=[1, 2, 3, 4, 5, 6])
                    logging.debug(" slots_delivered_df " + str(slots_delivered_df))
                    slots_delivered_plot = px.bar(slots_delivered_df,  x= slots_delivered_df.index, y=['delivered', 'NOT deliv.'], color_discrete_sequence=[ "lawngreen", "red"] )

    
                    # Render Agents Slot Prices and Settlement Prices 

                    price_plot = go.Figure()
                    price_plot.add_trace(go.Scatter(x=list(range(1,7)), y=self.previous_activation_results["slot_settlement_prices_DE"], line_color="blue", name="Market Price"))
                    price_plot.add_trace(go.Scatter(x=list(range(1,7)), y=self.previous_activation_results["agents_bid_prices"] , line_color="red", name="Agents Price" )) 


                    if self.env_type != "test":
                        
                        if self.env_type == "training":
                            wandb.log({
                                "Won / Loss of Slots": slots_won_plot,
                                "Sold and Available Capacity" : activation_plot,
                                "Agents and Settlement Prices per Slot" : price_plot,
                                "Delivery per Slot": slots_delivered_plot},
                                #step=self.logging_step,
                                commit=True
                            )
                            
                        if self.env_type == "eval":
                            wandb.log({
                                "global_step": self.logging_step,
                                "Won / Loss of Slots": slots_won_plot,
                                "Sold and Available Capacity" : activation_plot,
                                "Agents and Settlement Prices per Slot" : price_plot,
                                "Delivery per Slot": slots_delivered_plot},
                                step=self.logging_step,
                                commit=False
                            )

    
    def _simulate_vpp(self):
        
        vpp_total = self.asset_data_total[str(self.market_start) : str(self.market_end)].to_numpy(dtype=np.float32)
        #vpp_total_FCR = self.asset_data_FCR_total[str(self.market_start) : str(self.market_end)].to_numpy(dtype=np.float32)
        
        self.activation_results["vpp_total"] = vpp_total
        #self.activation_results["vpp_total_FCR"] = vpp_total_FCR
        self.activation_results["bid_sizes_all_slots"] = [0] * 96
        

    def _simulate_market(self, action_dict):
        
        auction_bids = self.bids_df[self.market_start : self.market_end]
        logging.debug("auction_bids = ")        
        logging.debug(self.bids_df[self.market_start : self.market_end])
        
        logging.info("Bid Submission time (D-1) = %s" % (self.bid_submission_time))
        logging.info("Gate Closure time (D-1) = %s" % (self.gate_closure))
        logging.info("Historic Data Window: from %s to %s " % (self.historic_data_start, self.historic_data_end))
        logging.info("Forecast Data Window: from %s to %s " % (self.forecast_start, self.forecast_end))

        self.activation_results["agents_bid_prices"] = [None,None,None,None,None,None]
        self.activation_results["agents_bid_sizes_round"] = [None,None,None,None,None,None]
        self.activation_results["slots_won"] = [None,None,None,None,None,None]


        for slot in range(0, len(self.slot_date_list)):
            slot_date = self.slot_date_list[slot]
            logging.info("Current Slot Time: (D) = %s" % (slot_date)) 
            slot_bids = auction_bids[slot_date : slot_date].reset_index(drop=True).reset_index(drop=False)
            #logging.debug("slot_bids = " + str(slot_bids))
            slot_bids_list = slot_bids.to_dict('records')
            #logging.debug("slot_bids_list = " + str(slot_bids_list))
            # extract the bid size out of the agents action
            # ROUND TO FULL INTEGER
            agents_bid_size = round(action_dict["size"][slot])
            self.activation_results["agents_bid_sizes_round"][slot] = agents_bid_size
            # extract the bid price out of the agents action
            agents_bid_price = action_dict["price"][slot]
            self.activation_results["agents_bid_prices"][slot] = agents_bid_price
            logging.info("agents_bid_size = %s" % (agents_bid_size))
            logging.info("agents_bid_price = %s" % (agents_bid_price))            
            # get settlement price
            settlement_price_DE = [bid['settlement_price'] for bid in slot_bids_list if bid['country']== "DE"][0] 
            logging.info( "settlement_price_DE : " + str(settlement_price_DE))
            self.activation_results["slot_settlement_prices_DE"][slot] = settlement_price_DE

            
            # First check if agents bid price is higher than the settlement price of Germany 
            # OR if agents bid size is 0 
            if (agents_bid_price > settlement_price_DE) or (agents_bid_size == 0):
                # if it is higher, the slot is lost. 
                self.activation_results["slots_won"][slot] = 0
                # set settlement price for the current auctioned slot in slot_settlement_prices_DE list
                #self.activation_results["slot_settlement_prices_DE"][slot] = settlement_price_DE
            else: 
                # If agents bid price is lower than settlement price (bid could be in awarded bids)
                # get CBMP of countries without LMP
                unique_country_bids = list({v['country']:v for v in slot_bids_list}.values())
                grouped_prices = [x['settlement_price'] for x in unique_country_bids]
                cbmp = max(set(grouped_prices), key = grouped_prices.count)
                logging.info( "cbmp : " + str(cbmp))
                # check if settlement_price_DE is same as CBMP (no limit constraints where hit)
                if cbmp == settlement_price_DE:
                    price_filter = cbmp
                    logging.debug("DE has CBMP")
                else: 
                    # if Germany has a price based on limit constraints
                    price_filter = settlement_price_DE
                    logging.debug("DE has LMP")
                                
                # as the probability is high that the agents bid moved the last bid out of the list, 
                # we have to check which bids moved out of the list and what is the new settlement price
                
                # sort the bid list based on the price
                slot_bids_list_sorted_by_price = sorted(slot_bids_list, key=lambda x: x['price'])
                # filter the bid list by the settlement price of either the CBMP or the LMP of germany 
                #slot_bids_prices_filtered = [bid['price'] for bid in slot_bids_list_sorted_by_price if bid['settlement_price']== price_filter]
                #logging.debug(slot_bids_prices_filtered)
                slot_bids_filtered = [bid for bid in slot_bids_list_sorted_by_price if bid['settlement_price']== price_filter]
                accumulated_replaced_capacity = 0
                
                slot_bids_filtered_size_sum = sum([bid['size'] for bid in slot_bids_filtered])
                # for the case the action_dict space is not dynamic and agent can choose any bid size,
                 # it needs to be checked here if the agents_bid_size is too big and unrealistic
                if agents_bid_size >= slot_bids_filtered_size_sum:
                    logging.debug("unrealistic bid size")
                    # set auction won to false
                    self.activation_results["slots_won"][slot] = 0
                    # set settlement price to zero as it is an unrealistic auction
                    ##self.activation_results["slot_settlement_prices_DE"][slot] = 0
                    # i would rather set the old settlement_price_DE as the market price instead of blank 0. 
                    self.activation_results["slot_settlement_prices_DE"][slot] = settlement_price_DE
                else:
                    for bid in range(0, len(slot_bids_filtered)): 
                        logging.debug("bid size = " + str(slot_bids_filtered[-(bid+1)]["size"]))
                        logging.debug("bid price = " + str(slot_bids_filtered[-(bid+1)]["price"]))
                        bid_capacity = slot_bids_filtered[-(bid+1)]["size"]
                        accumulated_replaced_capacity += bid_capacity
                        logging.debug("accumulated_replaced_capacity = " + str( accumulated_replaced_capacity))
                            
                        if accumulated_replaced_capacity >= agents_bid_size:
                            logging.debug("realistic bid size")
                            if slot_bids_filtered[-(bid+1)]["indivisible"] is False:
                                logging.debug("bid is divisible, so current bids price is new settlement price")
                                new_settlement_price_DE = slot_bids_filtered[-(bid+1)]["price"]
                                logging.info("new_settlement_price_DE = " + str( new_settlement_price_DE))
                                # set boolean for auction win
                                self.activation_results["slots_won"][slot] = 1
                                # set settlement price for the current auctioned slot in slot_settlement_prices_DE list
                                self.activation_results["slot_settlement_prices_DE"][slot] = new_settlement_price_DE
                                break
                            else:
                                logging.debug("bid is INDIVISIBLE, so move one bids further is new settlement price")
                                accumulated_replaced_capacity -= bid_capacity
                                continue
                        

            logging.info("self.activation_results['slots_won'] = ")
            logging.info("\n".join("slot won: \t{}".format(k) for k in self.activation_results["slots_won"]))
            logging.info("     agents bid_size = ")
            logging.info("\n".join("size: \t{}".format(round(k) )for k in action_dict["size"]))            
            logging.info("self.activation_results['slot_settlement_prices_DE'] = ")
            logging.info("\n".join("price: \t{}".format(k) for k in self.activation_results["slot_settlement_prices_DE"]))
            
            
    def _prepare_activation(self):
        '''
        
        '''
        
        # extend slot bid size format from 6 slots to 96 time steps
        bid_sizes_list = []
        for slot_x in range (0,6): 
            for time_step in range(0,16):
                #bid_sizes_list.append(action_dict["size"][slot_x])
                bid_sizes_list.append(self.activation_results["agents_bid_sizes_round"][slot_x])
        bid_sizes_all_slots = np.array(bid_sizes_list)
        self.activation_results["agents_bid_sizes_round_all_slots"] = bid_sizes_all_slots
        self.activation_results["bid_sizes_all_slots"] = bid_sizes_all_slots
        logging.debug("self.activation_results['bid_sizes_all_slots'] : "  + str(self.activation_results['bid_sizes_all_slots']))
        
        # initialize slots dict
        self.activation_results["delivered_slots"] = {}
        self.activation_results["not_delivered_capacity"] = {}
        # initialize slots in dict 
        for slot in range (0,6):
            self.activation_results["delivered_slots"][slot] = None
            
            
    def _simulate_reservation(self, slot):
        
        logging.debug("reservation Simulation for Slot No. " + str(slot))
        
        vpp_total_slot = self.activation_results["vpp_total"][slot *16 : (slot+1)*16]
        bid_sizes_per_slot = self.activation_results["bid_sizes_all_slots"][slot *16 : (slot+1)*16]  
        
        reservation_possible_list = []
        #not_reserved_capacity_list = []
        total_not_reserved_energy = 0.

        
        for time_step in range(0, 16):
            # for a 15 min time interval: 
            # 1kwh/ 15min = * 4 =  4kw/15min Durchschnittsleistung
            # 4kw/15min = * 0,25 = 1kwh/15min Energie
            
            # 0     = 4 MW * 0,25h = 1 MWh / 15min 
            # 15    = 3 MW * 0,25h = 0,75 MWh / 15min 
            # 30    = 2 MW * 0,25h = 0,5 MWh / 15min 
            # 45    = 1 MW * 0,25h = 0,25 MWh / 15min 
            # Summe = 2,5 MWh
        
            agent_bid_size = bid_sizes_per_slot[time_step] 
            available_vpp_capacity = vpp_total_slot[time_step]
            
            reservation_possible = False
            not_reserved_capacity = 0
            not_reserved_counter = 0
            not_reserved_energy = 0
            
            # 1. check negative reservation
            if (available_vpp_capacity - agent_bid_size) > 0:
                logging.error("Negative reservation possible for slot " + str(slot) + "and timestep " + str(time_step))
                reservation_possible = True

            else:
                logging.error("Negative reservation NOT possible for slot " + str(slot) + "and timestep " + str(time_step))
                not_reserved_counter += 1
                not_reserved_capacity += abs(available_vpp_capacity - agent_bid_size)
                #not_reserved_capacity_list.append(not_reserved_capacity)
                
                not_reserved_energy = not_reserved_capacity * 0.25  # multiply power with time to get energy 
                total_not_reserved_energy += not_reserved_energy

                #not_reserved_energy_list.append(not_reserved_energy)
                
            # 2. check positive reservation
            if agent_bid_size < available_vpp_capacity:
                logging.error("positive reservation possible for slot " + str(slot) + "and timestep " + str(time_step))
                reservation_possible = True
            else:
                # only calculate not_reserved_capacity if not already calculated for negative FCR
                if not_reserved_counter != 1: 
                    logging.error("Positive reservation NOT possible (and negative reservation not given penalty yet) for slot " + str(slot) + "and timestep " + str(time_step))

                    not_reserved_capacity += abs(agent_bid_size - available_vpp_capacity)
                    #not_reserved_capacity_list.append(not_reserved_capacity)
                    
                    not_reserved_energy = not_reserved_capacity * 0.25  # multiply power with time to get energy 
                    #not_reserved_energy_list.append(not_reserved_energy)
                    total_not_reserved_energy += not_reserved_energy


            reservation_possible_list.append(reservation_possible)

        if all(reservation_possible_list): 
            total_reservation_possible = True
        else: 
            total_reservation_possible = False
        
        #not_reserved_capacity_mean = sum(not_reserved_capacity_list) / len(not_reserved_capacity_list)
        logging.error("total_not_reserved_energy for " + str(slot) + " and timestep " + str(time_step) + "is " + str(total_not_reserved_energy))
        self.activation_results["total_not_reserved_energy"][slot] = total_not_reserved_energy

        return total_reservation_possible

                

    def _check_activation_possible(self, agent_bid_size, vpp_total_step):
        '''
        
        '''
        # assumption: mean activation length: 30 Seconds 
        # assumption: mean number of deliveries per hour: every 60 seconds = once every minute
        # assumption: positive and negative : 50 % / 50% = per 15 minutes:  
            #  7,5 times * 30 Seconds positive FCR 
            #  7,5 times * 30 Seconds negative FCR 
            
        not_delivered_capacity = 0
        not_delivered_energy = 0

        # 2. Probability of maximum activation Amount (74 % :  0 - 5 %  Capacity , 18 % : 5 - 10 % , 5% : 10-15% ( 97%: max 15 %)

        max_activation_share = random.choices(
             population=[0.05, 0.1,  0.15, 0.2,  0.25,  0.5,   0.75,   1.0],
             weights=   [0.74, 0.18, 0.05, 0.02, 0.007, 0.001, 0.001, 0.001],
             k=1
         )

        capacity_to_deliver = max_activation_share[0] * agent_bid_size

        logging.debug("check No. 2: agent_bid_size : " + str(agent_bid_size))
        logging.debug("check No. 2: max_activation_share : " + str(max_activation_share[0]))
        logging.debug("check No. 2: capacity_to_deliver : " + str(capacity_to_deliver))
        
        # 3. Probability of successfull activation (100%: 10% of HPP Capacity = Probability Curve)

        mean = 0 # symmetrical normal distribution at 0 
        sd = self.maximum_possible_VPP_capacity/7

        max_at_10_percent = norm.pdf(self.maximum_possible_VPP_capacity*0.1,mean,sd)
        scale_factor = 1 / max_at_10_percent

        logging.debug("check No. 3: max_at_10_percent = " + str(max_at_10_percent))
        logging.debug("check No. 3: scale_factor = " + str(scale_factor))
        
        # Plot between -max_power and max_power with .001 steps.
        #x_axis = np.arange(-max_power, max_power, 0.001)
        #plt.plot(x_axis, (norm.pdf(x_axis, mean, sd)) * scale_factor + shift_to_100)
        #plt.show()

        propab_of_activation = round((norm.pdf(capacity_to_deliver, mean,sd) * scale_factor),3)

        if propab_of_activation > 1.0: 
            propab_of_activation =  1.0

        logging.debug("check No. 3: propab_of_activation = " + str(propab_of_activation))

        activation_possible = random.choices(
             population=[True, False],
             weights=   [propab_of_activation , (1-propab_of_activation)],
             k=1
         )
        activation_possible = activation_possible[0] # as bool is in list 
        logging.debug("check No. 3: activation_possible : " + str(activation_possible))
        
        # 4. Check VPP Boundaries: In case of a very high or low operating point (nearly 100% or 0% power output of the HPP): 
        # then the activation is not possible. 
        logging.debug("Check No. 4")
        # check if probability of activation is high and capacity could be deliverd
        if activation_possible == True:
            # when negative FCR : 
            if capacity_to_deliver < 0:
                # check if capacity_to_deliver is bigger than vpp_total_step
                if (vpp_total_step - abs(capacity_to_deliver)) < 0:
                    logging.error("Check No. 4: Error, vpp_total_step is small and cant do more negative FCR")
                    not_delivered_capacity += abs((vpp_total_step - abs(capacity_to_deliver)))
                    # not_delivered_capacity = MW
                    # 0.5/60 = 30Seconds 
                    # multiplied = MWh
                    # 7.5 = only for negative FCR = half of 15minutes , other half for positive FCR.  
                    not_delivered_energy = 7.5 * (not_delivered_capacity * 0.5/60)  # multiply power with time to get energy 

                    activation_possible = False
            # if positive FCR
            else:
                # positive FCR is already included in vpp_total_step, so it should be possible
                '''# if the plant is already at maximum limit, then it cant produce more
                if (vpp_total_step + capacity_to_deliver) > self.maximum_possible_VPP_capacity: 
                    logging.error("Error, FCR is larger than maximum_possible_VPP_capacity")
                    not_delivered_capacity
                    activation_possible = False'''
                # if the plant cant produce any power then positive FCR is also not possible
                if capacity_to_deliver >= vpp_total_step:
                    logging.error("Check No. 4: Error, FCR is larger than vpp_total_step")
                    not_delivered_capacity += capacity_to_deliver - vpp_total_step
                    logging.error("Check No. 4: not_delivered_capacity = " + str(not_delivered_capacity))
                    # not_delivered_capacity = MW
                    # 0.5/60 = 30Seconds 
                    # multiplied = MWh
                    # 7.5 = only for positive FCR = half of 15minutes , other half for negative FCR.  
                    not_delivered_energy = 7.5 * (not_delivered_capacity * 0.5/60)  # multiply power with time to get energy 

                    activation_possible = False
                    
        logging.debug("check No. 4: activation_possible : " + str(activation_possible))   
         
        return activation_possible, not_delivered_energy
    

    def _simulate_activation(self, slot): 
                
        logging.debug("activation Simulation for Slot No. " + str(slot))
        
        vpp_total_slot = self.activation_results["vpp_total"][slot *16 : (slot+1)*16]
        bid_sizes_per_slot = self.activation_results["bid_sizes_all_slots"][slot *16 : (slot+1)*16]
        
        #logging.debug("vpp_total_FCR_slot " + str(vpp_total_FCR_slot))
        logging.debug("vpp_total_slot " + str(vpp_total_slot))
        logging.debug("bid_sizes_per_slot " + str(bid_sizes_per_slot))

        activation_possible = None
        positive_activation_possible_list = []
        negative_activation_possible_list = []
        total_not_delivered_energy = 0.

        # check for every timestep
        for time_step in range(0, 16):
        
            agent_bid_size = bid_sizes_per_slot[time_step]
           
            logging.debug("vpp_total_slot[time_step] for  time_step =" + str(time_step) + " : " + str(vpp_total_slot[time_step]))
            logging.debug("bid_sizes_per_slot[time_step] for time_step ="  + str(time_step) + " :  " + str(bid_sizes_per_slot[time_step]))

            # check if positive FCR could be provided 
            logging.debug("check positive FCR")
            activation_possible, not_delivered_energy = self._check_activation_possible(agent_bid_size, vpp_total_slot[time_step])
            positive_activation_possible_list.append(activation_possible)
            if activation_possible == False:
                    total_not_delivered_energy += not_delivered_energy

            # check if negative FCR could be provided 
            logging.debug("check negative FCR")
            activation_possible, not_delivered_energy = self._check_activation_possible(-agent_bid_size, vpp_total_slot[time_step])
            negative_activation_possible_list.append(activation_possible)
            if activation_possible == False:
                total_not_delivered_energy += not_delivered_energy

        if all(positive_activation_possible_list) and all(negative_activation_possible_list): 
            total_activation_possible = True 
        else: 
            total_activation_possible = False
            
        logging.debug("total_activation_possible for slot " + str(slot) + " : " + str(total_activation_possible))
        self.activation_results["delivered_slots"][slot] = total_activation_possible
        self.activation_results["total_not_delivered_energy"][slot] = total_not_delivered_energy
           