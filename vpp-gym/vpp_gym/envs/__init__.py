from vpp_gym.envs.vpp_env import VPPBiddingEnv
from gym.envs.registration import register

register(
    id='vpp-v0',
    entry_point='vpp_gym.envs:VPPBiddingEnv',
)
