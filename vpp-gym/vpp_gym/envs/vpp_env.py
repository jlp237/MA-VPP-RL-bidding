from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, flatdim
import numpy as np
import sys

def get_initial_state(): 
    pass 

def get_next_state(episodes): 
    pass 

def calculate_reward(state, action): 
    return 1.0

class DataEnv(object):
  ''' 
   VPPBiddingEnv's data source.
  
  Pulls data from repo, does preprocessing for use by VPPBiddingEnv and then 
  acts as data provider for each new episode.
 

  def __init__(self):
    df = 
    df = df[ ~np.isnan(df['Share Volume (000)'])][['Nominal Price','Share Volume (000)']]
    # we calculate returns and percentiles, then kill nans
    df = df[['Nominal Price','Share Volume (000)']]   
    df['Share Volume (000)'].replace(0,1,inplace=True) # days shouldn't have zero volume..
    df['Return'] = (df['Nominal Price']-df['Nominal Price'].shift())/df['Nominal Price'].shift()
    pctrank = lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    df['ClosePctl'] = df['Nominal Price'].expanding(self.MinPercentileDays).apply(pctrank)
    df['VolumePctl'] = df['Share Volume (000)'].expanding(self.MinPercentileDays).apply(pctrank)
    df.dropna(axis=0,inplace=True)
    R = df.Return
    if scale:
      mean_values = df.mean(axis=0)
      std_values = df.std(axis=0)
      df = (df - np.array(mean_values))/ np.array(std_values)
    df['Return'] = R # we don't want our returns scaled
    self.min_values = df.min(axis=0)
    self.max_values = df.max(axis=0)
    self.data = df
    self.step = 0
     '''

class VPPBiddingEnv(Env):
    
    def __init__(self):   
        low = np.array([0.0] * 96) #96 timesteps to min 0.0
        high = np.array([1.0] * 96) #96 timesteps to max 1.0

        self.nested_observation_space = Dict({
            'historic_data': Dict({
                "hydro_historic": Box(low, high, dtype=np.float32),
                "wind_historic":  Box(low, high, dtype=np.float32)
            }),
            'forecast_data':  Dict({
                "hydro_forecast": Box(low, high, dtype=np.float32),
                "wind_forecast": Box(low, high, dtype=np.float32),
                "soc_forecast": Box(low, high, dtype=np.float32)
                # TODO should I keep the Battery state of charge? 
            }),
            'market_data':  Dict({
                "market_demand": Discrete(3), # for the demands 573, 562 and 555 MW
                # TODO for 2021 its always 562, how to handle differetn years? maybe set it as a global constant? 

                "predicted_market_prices":  Box(low=0.0, high=1634.52, shape=(6, 1), dtype=np.float32), # for each slot, can be prices of same day last week 
            }),
            'time_features':  Dict({
                "weekday": Discrete(7), # for the days of the week
                "holiday": Discrete(2), # holiday = 1, no holiday = 0
                "month": Discrete(12), # for the month
            })
        })
        
        # first approach: 
        # Slots 1,2,3,4,5,6 = integrated in bidsize if 0 or non 0 
        # bid size: MultiDiscrete([ 25, 25, 25, 25, 25 ]),
        # bid prize: Box(low=0.0, high=1634.52, shape=(6,), dtype=np.float32))

        self.action_space = Tuple((
            # returns array([ 0,  2, 13,  6, 23]
            MultiDiscrete([ 25, 25, 25, 25, 25 ]),
            # returns array([1311.5632  ,  665.4643  ,  807.9639  ,  104.337715,  425.967, 205.23262 ]
            Box(low=0.0, high=1634.52, shape=(6,), dtype=np.float32)))
        
        # TODO: Add second approach with shares of plants
        '''
        second approach: 
            Share of hydro
            Share of wind
            Share of battery
        '''
        self.episodes = 365
        
    def step(self, action):
        
        # calculate reward from state and action 
        reward = calculate_reward(self.state, action)
        #TODO: write reward function 
        
        # apply the action, as each episode the action is set completely new, we dont need to calculate anything here
        # the application of the action is already done in the reward function.  
        #self.state = apply_action(self.state, action) 
        
        self.episodes -= 1
        
        # Check if whole year is iterated
        if self.episodes <= 0: 
            done = True
        else:
            done = False
                
        info = {}
        # TODO: info can contain state variables that are hidden from observations
        # or individual reward terms that are combined to produce the total reward

        
        self.state = get_next_state(self.episodes)

        return self.state, reward, done, info
    

    def render(self):
        # Implement visulisation
        pass
    
    def reset(self):
        # for each episode we need to get new state
        self.state = get_initial_state()
        # TODO: write get next state function
        # TODO: how to set the first state? needs to be 01.01.2021 from the DataEnv
        
        return self.state
    