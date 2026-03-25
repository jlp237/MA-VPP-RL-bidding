"""Weights & Biases helper utilities."""

from __future__ import annotations

import logging
from typing import Any

import wandb

logger = logging.getLogger(__name__)


def init_wandb(
    project: str,
    config: dict[str, Any],
    tags: list[str] | None = None,
    mode: str = "online",
) -> None:
    """Initialize a WandB run.

    Args:
        project: WandB project name.
        config: Configuration dictionary to log.
        tags: Optional list of tags for the run.
        mode: WandB mode ("online", "offline", or "disabled").
    """
    wandb.init(
        project=project,
        config=config,
        tags=tags,
        mode=mode,  # type: ignore[arg-type]
    )
    logger.info("WandB run initialized: project=%s, mode=%s", project, mode)


def finish_wandb() -> None:
    """Finish the current WandB run."""
    if wandb.run is not None:
        wandb.finish()
        logger.info("WandB run finished")
