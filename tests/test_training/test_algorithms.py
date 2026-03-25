"""Tests for the algorithm factory."""

import pytest

from vpp_bidding.domain.enums import Algorithm
from vpp_bidding.training.algorithms import (
    _ALGORITHM_REGISTRY,
    create_agent,
    get_policy_class,
)


class TestAlgorithmRegistry:
    def test_all_algorithms_registered(self) -> None:
        """All Algorithm enum members should be in the registry."""
        for algo in Algorithm:
            assert algo in _ALGORITHM_REGISTRY, f"{algo} not registered"

    def test_registry_has_8_algorithms(self) -> None:
        assert len(_ALGORITHM_REGISTRY) == 8

    @pytest.mark.parametrize(
        "algo,expected_class_name",
        [
            (Algorithm.PPO, "PPO"),
            (Algorithm.A2C, "A2C"),
            (Algorithm.SAC, "SAC"),
            (Algorithm.DDPG, "DDPG"),
            (Algorithm.TD3, "TD3"),
            (Algorithm.TRPO, "TRPO"),
            (Algorithm.RECURRENT_PPO, "RecurrentPPO"),
            (Algorithm.TQC, "TQC"),
        ],
    )
    def test_correct_sb3_class(self, algo: Algorithm, expected_class_name: str) -> None:
        cls = _ALGORITHM_REGISTRY[algo]
        assert cls.__name__ == expected_class_name


class TestGetPolicyClass:
    def test_recurrent_ppo_uses_lstm_policy(self) -> None:
        assert get_policy_class(Algorithm.RECURRENT_PPO) == "MultiInputLstmPolicy"

    @pytest.mark.parametrize(
        "algo",
        [
            Algorithm.PPO,
            Algorithm.A2C,
            Algorithm.SAC,
            Algorithm.DDPG,
            Algorithm.TD3,
            Algorithm.TRPO,
            Algorithm.TQC,
        ],
    )
    def test_non_recurrent_uses_multi_input_policy(self, algo: Algorithm) -> None:
        assert get_policy_class(algo) == "MultiInputPolicy"


class TestCreateAgent:
    def test_invalid_algorithm_raises(self) -> None:
        """create_agent with an unregistered algorithm value should raise."""
        # We can't easily pass an invalid Algorithm enum member, but we
        # can test the error path by temporarily removing one from the registry
        from unittest.mock import patch

        with (
            patch.dict(_ALGORITHM_REGISTRY, {}, clear=True),
            pytest.raises(ValueError, match="Unknown algorithm"),
        ):
            create_agent(Algorithm.PPO, env=None)
