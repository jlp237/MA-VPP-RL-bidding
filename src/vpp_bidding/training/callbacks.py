"""Training callbacks for logging and evaluation."""

import logging
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

logger = logging.getLogger(__name__)


class WandbMetricsCallback(BaseCallback):
    """Log training metrics to Weights & Biases.

    Captures rollout and training metrics from the stable-baselines3 logger
    and forwards them to the active WandB run.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            import wandb
        except ImportError:
            return True

        if wandb.run is None:
            return True

        # Log scalar metrics from the SB3 logger
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                for key in ("reward", "profit", "revenue", "cost"):
                    if key in info:
                        wandb.log(
                            {f"rollout/{key}": info[key]},
                            step=self.num_timesteps,
                        )

        return True

    def _on_rollout_end(self) -> None:
        try:
            import wandb
        except ImportError:
            return

        if wandb.run is None:
            return

        # Log episode statistics if available
        if hasattr(self.model, "ep_info_buffer") and self.model.ep_info_buffer:
            ep_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            ep_lengths = [ep["l"] for ep in self.model.ep_info_buffer]
            wandb.log(
                {
                    "rollout/ep_rew_mean": np.mean(ep_rewards),
                    "rollout/ep_len_mean": np.mean(ep_lengths),
                },
                step=self.num_timesteps,
            )


class WandbEvalCallback(EvalCallback):
    """Evaluation callback that also logs results to WandB."""

    def __init__(self, eval_env: Any, **kwargs: Any) -> None:
        super().__init__(eval_env, **kwargs)

    def _on_step(self) -> bool:
        result = super()._on_step()

        try:
            import wandb
        except ImportError:
            return result

        if wandb.run is None:
            return result

        if self.last_mean_reward is not None:
            wandb.log(
                {
                    "eval/mean_reward": self.last_mean_reward,
                },
                step=self.num_timesteps,
            )

        return result
