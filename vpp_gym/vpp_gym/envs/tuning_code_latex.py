def sample_td3_params(trial: optuna.Trial):
    """
    Sampler for TD3 hyperparams.
    :param trial:
    :return:
    """

    trial.using_her_replay_buffer = False

    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical(
        "batch_size", [16, 32, 64, 100, 128, 200])
    buffer_size = trial.suggest_categorical(
        "buffer_size", [int(1e4), int(1e5), int(1e6)])
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    train_freq = trial.suggest_categorical("train_freq", [1, 2, 8, 32])
    # gradient_steps = train_freq
    gradient_steps = trial.suggest_categorical("gradient_steps", [-1, 1, 2, 8, 32])
    learning_starts = trial.suggest_categorical(
        "learning_starts", [0, 1, 10, 20, 100, 200])
    noise_type = trial.suggest_categorical(
        "noise_type", ["ornstein-uhlenbeck", "normal", None])
    noise_std = trial.suggest_uniform("noise_std", 0, 1)
    policy_delay = trial.suggest_categorical("policy_delay", [1, 2, 5])
    target_policy_noise = trial.suggest_categorical(
        "target_policy_noise", [0.1, 0.2, 0.3])
    if trial.using_her_replay_buffer:
        net_arch = trial.suggest_categorical(
            "net_arch", ["small", "medium", "big", "verybig"])
    else:
        net_arch = trial.suggest_categorical(
            "net_arch", ["small", "medium", "big"])
    activation_fn = trial.suggest_categorical(
        "activation_fn", ["tanh", "relu", "elu", "leaky_relu"])
    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
        "verybig": [256, 256, 256],
    }[net_arch]
    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }[activation_fn]
    hyperparams = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "tau": tau,
        "gamma": gamma,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "learning_starts": learning_starts,
        "policy_delay": policy_delay,
        "target_policy_noise": target_policy_noise,
        "policy_kwargs": dict(net_arch=net_arch, activation_fn=activation_fn),
    }
    n_actions = 12
    if noise_type == "normal":
        hyperparams["action_noise"] = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
        )
    elif noise_type == "ornstein-uhlenbeck":
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
        )

    return hyperparams
