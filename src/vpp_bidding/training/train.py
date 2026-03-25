"""Main training pipeline for VPP bidding agents."""

import logging
from pathlib import Path
from typing import Any

from stable_baselines3.common.monitor import Monitor

from vpp_bidding.config import load_config
from vpp_bidding.domain.enums import Algorithm
from vpp_bidding.training.algorithms import create_agent
from vpp_bidding.training.callbacks import WandbMetricsCallback
from vpp_bidding.utils.wandb import finish_wandb, init_wandb

logger = logging.getLogger(__name__)


def train(
    config_path: Path,
    algorithm: Algorithm,
    total_timesteps: int,
    seed: int = 42,
) -> None:
    """Train an RL agent on the VPP bidding environment.

    Args:
        config_path: Path to the TOML configuration file.
        algorithm: The RL algorithm to use.
        total_timesteps: Total number of training timesteps.
        seed: Random seed for reproducibility.
    """
    config = load_config(config_path)

    logger.info(
        "Starting training: algorithm=%s, timesteps=%d, seed=%d",
        algorithm.value,
        total_timesteps,
        seed,
    )

    # Register and create the VPP environment
    from vpp_bidding.env.registration import make_env

    raw_env = make_env(config, mode="train")
    env: Any = Monitor(raw_env)

    # Create the agent
    agent = create_agent(algorithm, env, seed=seed, verbose=1)

    # Set up WandB logging
    init_wandb(
        project=config.wandb.project,
        config={
            "algorithm": algorithm.value,
            "total_timesteps": total_timesteps,
            "seed": seed,
        },
        tags=[algorithm.value, "training"],
        mode=config.wandb.mode,
    )

    # Train
    callbacks = [WandbMetricsCallback()]
    try:
        agent.learn(total_timesteps=total_timesteps, callback=callbacks)
    finally:
        finish_wandb()

    # Save model
    model_dir = Path("models") / algorithm.value
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"vpp_{algorithm.value}_{total_timesteps}steps"
    agent.save(str(model_path))
    logger.info("Model saved to %s", model_path)

    env.close()
