"""Evaluation pipeline for trained VPP bidding agents."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from stable_baselines3.common.monitor import Monitor

from vpp_bidding.config import load_config
from vpp_bidding.domain.enums import Algorithm

if TYPE_CHECKING:
    from pathlib import Path

    from stable_baselines3.common.base_class import BaseAlgorithm

logger = logging.getLogger(__name__)


def _detect_algorithm(model_path: Path) -> Algorithm:
    """Detect the algorithm from the model filename convention."""
    stem = model_path.stem.lower()
    for algo in Algorithm:
        if algo.value in stem:
            return algo
    raise ValueError(f"Cannot detect algorithm from model path: {model_path}")


def _load_model(model_path: Path, algorithm: Algorithm, env: Any) -> BaseAlgorithm:
    """Load a trained model from disk."""
    from sb3_contrib import TQC, TRPO, RecurrentPPO
    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

    cls_map: dict[Algorithm, type] = {
        Algorithm.PPO: PPO,
        Algorithm.A2C: A2C,
        Algorithm.SAC: SAC,
        Algorithm.DDPG: DDPG,
        Algorithm.TD3: TD3,
        Algorithm.TRPO: TRPO,
        Algorithm.RECURRENT_PPO: RecurrentPPO,
        Algorithm.TQC: TQC,
    }

    cls = cls_map[algorithm]
    return cls.load(str(model_path), env=env)  # type: ignore[attr-defined, no-any-return]


def evaluate(
    model_path: Path,
    config_path: Path,
    episodes: int = 70,
) -> dict[str, float]:
    """Evaluate a trained agent on the test set.

    Args:
        model_path: Path to the saved model file.
        config_path: Path to the TOML configuration file.
        episodes: Number of evaluation episodes to run.

    Returns:
        Dictionary of evaluation metrics (mean_reward, std_reward,
        mean_profit, etc.).
    """
    config = load_config(config_path)
    algorithm = _detect_algorithm(model_path)

    logger.info(
        "Evaluating model=%s, algorithm=%s, episodes=%d",
        model_path,
        algorithm.value,
        episodes,
    )

    # Create evaluation environment
    from vpp_bidding.env.registration import make_env

    raw_env = make_env(config, mode="test")
    env: Any = Monitor(raw_env)

    # Load the trained model
    model = _load_model(model_path, algorithm, env)

    # Run evaluation episodes
    episode_rewards: list[float] = []
    episode_profits: list[float] = []

    for _ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        total_profit = 0.0

        # Handle LSTM state for RecurrentPPO
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)

        while not done:
            if algorithm == Algorithm.RECURRENT_PPO:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=True,
                )
                episode_start = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            total_profit += info.get("profit", 0.0)

        episode_rewards.append(total_reward)
        episode_profits.append(total_profit)

    env.close()

    metrics = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_profit": float(np.mean(episode_profits)),
        "std_profit": float(np.std(episode_profits)),
        "episodes": episodes,
    }

    logger.info(
        "Evaluation complete: mean_reward=%.2f (±%.2f), mean_profit=%.2f",
        metrics["mean_reward"],
        metrics["std_reward"],
        metrics["mean_profit"],
    )

    return metrics
