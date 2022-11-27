from gym.envs.registration import register

def register_env(config, seed):
    
    register(
        id="VPPBiddingEnv-TRAIN-v1",
        entry_point='vpp_gym.vpp_gym.envs.vpp_env:VPPBiddingEnv',
        max_episode_steps=1,
        kwargs={'config_path': config,
                'log_level' : "DEBUG", # "DEBUG" , "INFO" or  "WARNING"
                'env_type' :"training",
                'render_mode' :"human", # "human", "fast_training" or None
                'seed': seed
            }
    )

    register(
        id="VPPBiddingEnv-TUNING-v1",
        entry_point='vpp_gym.vpp_gym.envs.vpp_env:VPPBiddingEnv',
        max_episode_steps=1,
        kwargs={'config_path': config,
                'log_level' : "WARNING", # "DEBUG" , "INFO" or  "WARNING"
                'env_type' :"training",
                'render_mode' :"fast_training", # "human", "fast_training" or None
                'seed': seed
            }
    )


    register(
        id="VPPBiddingEnv-EVAL-v1",
        entry_point='vpp_gym.vpp_gym.envs.vpp_env:VPPBiddingEnv',
        max_episode_steps=1,
        kwargs={'config_path': config,
                'log_level' : "DEBUG", # "DEBUG" , "INFO" or  "WARNING"
                'env_type' :"eval",
                'render_mode' :"human", # "human", "fast_training" or None
                'seed': seed
            }
    )

    register(
        id="VPPBiddingEnv-TUNING-EVAL-v1",
        entry_point='vpp_gym.vpp_gym.envs.vpp_env:VPPBiddingEnv',
        max_episode_steps=1,
        kwargs={'config_path': config,
                'log_level' : "WARNING", # "DEBUG" , "INFO" or  "WARNING"
                'env_type' :"eval",
                'render_mode' :"fast_training", # "human", "fast_training" or None
                'seed': seed
            }
    )

    register(
        id="VPPBiddingEnv-TEST-v1",
        entry_point='vpp_gym.vpp_gym.envs.vpp_env:VPPBiddingEnv',
        max_episode_steps=1,
        kwargs={'config_path': config,
                'log_level' : "WARNING", # "DEBUG" , "INFO" or  "WARNING"
                'env_type' :"test",
                'render_mode' :"human", # "human", "fast_training" or None
                'seed': seed
            }
    )