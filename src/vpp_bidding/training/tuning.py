"""Optuna hyperparameter tuning for VPP bidding agents."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
from stable_baselines3.common.monitor import Monitor

from vpp_bidding.config import load_config
from vpp_bidding.domain.enums import Algorithm
from vpp_bidding.training.algorithms import create_agent
from vpp_bidding.training.callbacks import WandbMetricsCallback

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def sample_ppo_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample PPO hyperparameters."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [64, 128, 256, 512, 1024]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "n_epochs": trial.suggest_int("n_epochs", 3, 30),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 1.0),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 5.0),
    }


def sample_a2c_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample A2C hyperparameters."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128]),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 1.0),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 5.0),
        "normalize_advantage": trial.suggest_categorical("normalize_advantage", [True, False]),
    }


def sample_sac_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample SAC hyperparameters."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
        "tau": trial.suggest_float("tau", 0.001, 0.1, log=True),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8, 16]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 2, 4, 8]),
        "ent_coef": "auto",
        "learning_starts": trial.suggest_int("learning_starts", 100, 10000, log=True),
        "buffer_size": trial.suggest_categorical("buffer_size", [10000, 50000, 100000, 500000]),
    }


def sample_ddpg_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample DDPG hyperparameters."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
        "tau": trial.suggest_float("tau", 0.001, 0.1, log=True),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8, 16]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 2, 4]),
        "learning_starts": trial.suggest_int("learning_starts", 100, 10000, log=True),
        "buffer_size": trial.suggest_categorical("buffer_size", [10000, 50000, 100000]),
    }


def sample_td3_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample TD3 hyperparameters."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
        "tau": trial.suggest_float("tau", 0.001, 0.1, log=True),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8, 16]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 2, 4]),
        "learning_starts": trial.suggest_int("learning_starts", 100, 10000, log=True),
        "buffer_size": trial.suggest_categorical("buffer_size", [10000, 50000, 100000]),
        "policy_delay": trial.suggest_int("policy_delay", 1, 5),
        "target_policy_noise": trial.suggest_float("target_policy_noise", 0.1, 0.5),
    }


def sample_trpo_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample TRPO hyperparameters."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [64, 128, 256, 512, 1024]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 1.0),
        "target_kl": trial.suggest_float("target_kl", 0.005, 0.1, log=True),
        "cg_max_steps": trial.suggest_int("cg_max_steps", 10, 30),
        "cg_damping": trial.suggest_float("cg_damping", 0.01, 0.2, log=True),
    }


def sample_rppo_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample RecurrentPPO hyperparameters."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [64, 128, 256, 512]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "n_epochs": trial.suggest_int("n_epochs", 3, 20),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 1.0),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 5.0),
    }


def sample_tqc_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample TQC hyperparameters."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
        "tau": trial.suggest_float("tau", 0.001, 0.1, log=True),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8, 16]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 2, 4, 8]),
        "ent_coef": "auto",
        "learning_starts": trial.suggest_int("learning_starts", 100, 10000, log=True),
        "buffer_size": trial.suggest_categorical("buffer_size", [10000, 50000, 100000, 500000]),
        "top_quantiles_to_drop_per_net": trial.suggest_int("top_quantiles_to_drop_per_net", 0, 5),
        "n_quantiles": trial.suggest_int("n_quantiles", 15, 50),
    }


_SAMPLER_REGISTRY: dict[Algorithm, Callable[[optuna.Trial], dict[str, Any]]] = {
    Algorithm.PPO: sample_ppo_params,
    Algorithm.A2C: sample_a2c_params,
    Algorithm.SAC: sample_sac_params,
    Algorithm.DDPG: sample_ddpg_params,
    Algorithm.TD3: sample_td3_params,
    Algorithm.TRPO: sample_trpo_params,
    Algorithm.RECURRENT_PPO: sample_rppo_params,
    Algorithm.TQC: sample_tqc_params,
}


def tune(
    config_path: Path,
    algorithm: Algorithm,
    n_trials: int = 100,
    timeout: int = 21600,
) -> None:
    """Run Optuna hyperparameter optimization.

    Args:
        config_path: Path to the TOML configuration file.
        algorithm: The RL algorithm to tune.
        n_trials: Maximum number of trials.
        timeout: Maximum time in seconds for the optimization.
    """
    config = load_config(config_path)
    sampler_fn = _SAMPLER_REGISTRY.get(algorithm)
    if sampler_fn is None:
        raise ValueError(f"No hyperparameter sampler for algorithm: {algorithm.value}")

    study = optuna.create_study(
        study_name=f"vpp_{algorithm.value}_tuning",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
        ),
    )

    def objective(trial: optuna.Trial) -> float:
        params = sampler_fn(trial)

        from vpp_bidding.env.registration import make_env

        raw_env = make_env(config, mode="train")
        env: Any = Monitor(raw_env)

        try:
            agent = create_agent(algorithm, env, **params, seed=42, verbose=0)
            agent.learn(
                total_timesteps=config.training.total_timesteps,
                callback=[WandbMetricsCallback()],
            )
        except Exception as e:
            logger.warning("Trial %d failed: %s", trial.number, e)
            raise optuna.TrialPruned() from e
        finally:
            env.close()

        # Evaluate on validation set
        from vpp_bidding.env.registration import make_env as _make_env

        raw_eval_env = _make_env(config, mode="validation")
        eval_env: Any = Monitor(raw_eval_env)

        episode_rewards: list[float] = []
        for _ in range(10):
            obs, _info = eval_env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _info = eval_env.step(action)
                done = terminated or truncated
                total_reward += float(reward)
            episode_rewards.append(total_reward)

        eval_env.close()
        return float(np.mean(episode_rewards))

    logger.info(
        "Starting Optuna tuning: algorithm=%s, n_trials=%d, timeout=%ds",
        algorithm.value,
        n_trials,
        timeout,
    )

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    logger.info("Best trial: value=%.4f", study.best_trial.value)
    logger.info("Best params: %s", study.best_trial.params)

    # Save study results
    results_dir = Path("tuning_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(results_dir / f"{algorithm.value}_trials.csv", index=False)
