from gym.envs.registration import register


def register_env(config, seed):
    """Registers VPPBiddingEnv with OpenAI Gym.

        This function registers VPPBiddingEnv, a reinforcement learning environment, with OpenAI Gym.
        The environment has several modes: training, evaluation, and testing. The mode is determined by the
        env_type parameter. In addition, the environment has two rendering modes: "human" and "fast_training".
        The rendering mode is determined by the render_mode parameter.

        Args:
        config (str): Path to the configuration file for the environment.
        seed (int): Seed for the random number generator.

    """
    
    register(
        id="VPPBiddingEnv-TRAIN-v1",
        entry_point='vpp_gym.vpp_gym.envs.vpp_env:VPPBiddingEnv',
        max_episode_steps=1,
        kwargs={
            'config_path': config,
            'log_level': "DEBUG",  # "DEBUG" , "INFO" or  "WARNING"
            'env_type': "training",
            'render_mode': "human",  # "human", "fast_training" or None
            'seed': seed,
        },
    )
    
    register(
        id="VPPBiddingEnv-TRAIN-FAST-v1",
        entry_point='vpp_gym.vpp_gym.envs.vpp_env:VPPBiddingEnv',
        max_episode_steps=1,
        kwargs={
            'config_path': config,
            'log_level': "WARNING",  # "DEBUG" , "INFO" or  "WARNING"
            'env_type': "training",
            'render_mode': "human",  # "human", "fast_training" or None
            'seed': seed,
        },
    )

    register(
        id="VPPBiddingEnv-TUNING-v1",
        entry_point='vpp_gym.vpp_gym.envs.vpp_env:VPPBiddingEnv',
        max_episode_steps=1,
        kwargs={
            'config_path': config,
            'log_level': "WARNING",  # "DEBUG" , "INFO" or  "WARNING"
            'env_type': "training",
            'render_mode': "fast_training",  # "human", "fast_training" or None
            'seed': seed,
        },
    )
    
    register(
        id="VPPBiddingEnv-TUNING-logs-v1",
        entry_point='vpp_gym.vpp_gym.envs.vpp_env:VPPBiddingEnv',
        max_episode_steps=1,
        kwargs={
            'config_path': config,
            'log_level': "DEBUG",  # "DEBUG" , "INFO" or  "WARNING"
            'env_type': "training",
            'render_mode': "fast_training",  # "human", "fast_training" or None
            'seed': seed,
        },
    )

    register(
        id="VPPBiddingEnv-EVAL-v1",
        entry_point='vpp_gym.vpp_gym.envs.vpp_env:VPPBiddingEnv',
        max_episode_steps=1,
        kwargs={
            'config_path': config,
            'log_level': "DEBUG",  # "DEBUG" , "INFO" or  "WARNING"
            'env_type': "eval",
            'render_mode': "human",  # "human", "fast_training" or None
            'seed': seed,
        },
    )

    register(
        id="VPPBiddingEnv-TUNING-EVAL-v1",
        entry_point='vpp_gym.vpp_gym.envs.vpp_env:VPPBiddingEnv',
        max_episode_steps=1,
        kwargs={
            'config_path': config,
            'log_level': "WARNING",  # "DEBUG" , "INFO" or  "WARNING"
            'env_type': "eval",
            'render_mode': "fast_training",  # "human", "fast_training" or None
            'seed': seed,
        },
    )

    register(
        id="VPPBiddingEnv-TEST-v1",
        entry_point='vpp_gym.vpp_gym.envs.vpp_env:VPPBiddingEnv',
        max_episode_steps=1,
        kwargs={
            'config_path': config,
            'log_level': "WARNING",  # "DEBUG" , "INFO" or  "WARNING"
            'env_type': "test",
            'render_mode': "human",  # "human", "fast_training" or None
            'seed': seed,
        },
    )
