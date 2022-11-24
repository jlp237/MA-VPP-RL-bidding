import logging
from collections import OrderedDict
import numpy as np
import pandas as pd 
from typing import List


def get_observation(self):
    """_summary_

    Returns:
        _type_: _description_
    """    
    #print("in _get_observation() and logging_step = " + str(self.logging_step))

    asset_data_historic = self.asset_data_total[str(self.historic_data_start) : str(self.historic_data_end)].to_numpy(dtype=np.float32)
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  'None'  + " asset_data_historic = " + str(self.asset_data_total[str(self.historic_data_start) : str(self.historic_data_end)]) )
    # normalize the data 
    asset_data_historic_norm = self.asset_data_historic_scaler.transform((asset_data_historic.reshape(-1, 1)))
    # convert from 2D to 1D array 
    asset_data_historic_norm = asset_data_historic_norm.flatten()

    asset_data_forecast = self.asset_data_total[str(self.forecast_start) : str(self.forecast_end)].to_numpy(dtype=np.float32)
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  'None'  + " asset_data_forecast = "  + str(self.asset_data_total[str(self.forecast_start) : str(self.forecast_end)]))
    
    '''
    # add gaussian noise to data
    noisy_asset_data_forecast = self._add_gaussian_noise(asset_data_forecast, self.asset_data_total)
    noisy_asset_data_forecast = noisy_asset_data_forecast.astype(np.float32)
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  'None'  + " noisy_asset_data_forecast = "  + str(noisy_asset_data_forecast))
    noisy_asset_data_forecast_norm = self.noisy_asset_data_forecast_scaler.transform((noisy_asset_data_forecast.reshape(-1, 1)))
    noisy_asset_data_forecast_norm = noisy_asset_data_forecast_norm.flatten()
    '''
    
    # use perfect foresight forecast
    asset_data_forecast = asset_data_forecast.astype(np.float32)
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  'None'  + " asset_data_forecast = "  + str(asset_data_forecast))
    asset_data_forecast_norm = self.asset_data_historic_scaler.transform((asset_data_forecast.reshape(-1, 1)))
    asset_data_forecast_norm = asset_data_forecast_norm.flatten()
    
    # for predicted market Prices try naive prediction: retrieve price of same day last week 
    market_start_last_week = self.market_start - pd.offsets.DateOffset(days=7) 
    market_end_last_week = self.market_end - pd.offsets.DateOffset(days=7)
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  'None'  + " market_start_last_week = "  + str(market_start_last_week))
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  'None'  + " market_end_last_week = "  + str(market_end_last_week))
    predicted_market_prices = self.market_results["DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"][str(market_start_last_week) : str(market_end_last_week)].to_numpy(dtype=np.float32)
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  'None'  + " predicted_market_prices = "  + str(predicted_market_prices))
    if len(predicted_market_prices) < 6:
        # predicted_market_prices list is smaller than 6 so fake is generated mean of first week
        predicted_market_prices = np.array([ 17.48, 17.48, 17.48, 17.48, 17.48, 17.48], dtype=np.float32) 
        logging.debug("log_step: " + str(self.logging_step) + " slot: " +  'None'  + " predicted_market_prices list is smaller than 6 so fake is generated: "  + str(predicted_market_prices))
    predicted_market_prices_norm = self.price_scaler.transform((np.array(predicted_market_prices).reshape(-1, 1)))
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
    
    slots_won_list : List[int] = []
    slot_settlement_prices_DE_list : List[float] = []
    
    # beim ersten Trainingstep, wenn noch keine Daten vorhanden  
    if (self.initial is True):
        slots_won_list = [0,0,0,0,0,0]
        slot_settlement_prices_DE_list = [0.,0.,0.,0.,0.,0.]
    
    # observation nach reset
    if (self.done is False) and (self.initial is False):
        slots_won_list = self.previous_activation_results["slots_won"]
        slot_settlement_prices_DE_list = self.previous_activation_results["slot_settlement_prices_DE"]
        # replace None with 0 
        for i in range(len(slot_settlement_prices_DE_list)): 
            if slot_settlement_prices_DE_list[i] == None: 
                slot_settlement_prices_DE_list[i] = 0.0
        
    # observation nachdem action angewendet wurde 
    if (self.done is True) and (self.initial is False):
        #print("if schleife 3 = observation nachdem action in step() geataked wurde ")
        slots_won_list = self.activation_results["slots_won"].copy()
        
        slot_settlement_prices_DE_list = self.activation_results["slot_settlement_prices_DE"]
        # replace None with 0 
        for i in range(len(slot_settlement_prices_DE_list)): 
            if slot_settlement_prices_DE_list[i] == None: 
                slot_settlement_prices_DE_list[i] = 0.0
        
    slots_won_array =  np.array(slots_won_list, dtype=np.int32)
    slots_won_norm = self.list_scaler.transform(np.array(slots_won_array).reshape(-1, 1))
    slots_won_norm = slots_won_norm.flatten().astype('float32')

    
    slot_settlement_prices_DE_array = np.array(slot_settlement_prices_DE_list, dtype=np.float32)
    slot_settlement_prices_DE_norm = self.slot_settlement_prices_DE_scaler.transform((slot_settlement_prices_DE_array.reshape(-1, 1)))
    slot_settlement_prices_DE_norm = slot_settlement_prices_DE_norm.flatten().astype('float32')
    
    observation = OrderedDict({
        "asset_data_historic": asset_data_historic_norm,
        "asset_data_forecast": asset_data_forecast_norm,
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
    
    logging.debug("log_step: " + str(self.logging_step) + " slot: " +  'None'  + " NEW Observation = "  + str(observation))
    
    return observation