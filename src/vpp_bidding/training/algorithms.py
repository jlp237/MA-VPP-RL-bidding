"""Algorithm factory for creating RL agents from stable-baselines3."""

from typing import Any

from sb3_contrib import TQC, TRPO, RecurrentPPO
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from vpp_bidding.domain.enums import Algorithm

_ALGORITHM_REGISTRY: dict[Algorithm, type] = {
    Algorithm.PPO: PPO,
    Algorithm.A2C: A2C,
    Algorithm.SAC: SAC,
    Algorithm.DDPG: DDPG,
    Algorithm.TD3: TD3,
    Algorithm.TRPO: TRPO,
    Algorithm.RECURRENT_PPO: RecurrentPPO,
    Algorithm.TQC: TQC,
}


def get_policy_class(algorithm: Algorithm) -> str:
    """Return the appropriate policy class name."""
    if algorithm == Algorithm.RECURRENT_PPO:
        return "MultiInputLstmPolicy"
    return "MultiInputPolicy"


def create_agent(algorithm: Algorithm, env: Any, **kwargs: Any) -> Any:
    """Create an RL agent from the algorithm registry.

    Args:
        algorithm: The RL algorithm to use.
        env: The gymnasium environment.
        **kwargs: Additional keyword arguments passed to the agent constructor.

    Returns:
        An instantiated stable-baselines3 agent.

    Raises:
        ValueError: If the algorithm is not in the registry.
    """
    agent_cls = _ALGORITHM_REGISTRY.get(algorithm)
    if agent_cls is None:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. Supported: {[a.value for a in Algorithm]}"
        )

    policy = get_policy_class(algorithm)
    return agent_cls(policy, env, **kwargs)
