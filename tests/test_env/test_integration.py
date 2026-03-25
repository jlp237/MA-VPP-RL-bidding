"""Integration test: spin up the VPP environment and run a short training episode.

This test requires the actual data files in data/clean/ to be present.
It is marked with pytest.mark.integration so it can be skipped in CI
when data is not available.
"""

import json
import os
import tempfile

import pytest


@pytest.fixture
def config_path():
    """Return path to the training config JSON (original format, for env compatibility)."""
    # The env currently reads JSON configs directly
    config = {
        "config": {
            "csv_paths": {
                "renewables": "data/clean/renewables.csv",
                "tenders": "data/clean/tenders_all.csv",
                "market_results": "data/clean/market_results.csv",
                "bids": "data/clean/bids_all.csv",
                "time_features": "data/clean/time_features.csv",
                "test_set": "data/clean/test_set_70days.csv",
                "market_prices": "data/clean/wholesale_market_prices.csv",
            },
            "time": {
                "hist_window_size": 1,
                "forecast_window_size": 1,
                "first_slot_date_start": "2020-07-02 22:00:00+00:00",
                "last_slot_date_end": "2022-05-31 21:45:00+00:00",
            },
        },
        "assets": {
            "hydro": [
                {
                    "type": "run-of-river",
                    "max_capacity_MW": 10,
                    "quantity": 1,
                    "max_FCR_capacity_share": 0.5,
                    "asset_column_names": ["Hydro1"],
                },
                {
                    "type": "run-of-river",
                    "max_capacity_MW": 10,
                    "quantity": 1,
                    "max_FCR_capacity_share": 0.5,
                    "asset_column_names": ["Hydro2"],
                },
                {
                    "type": "run-of-river",
                    "max_capacity_MW": 10,
                    "quantity": 1,
                    "max_FCR_capacity_share": 0.5,
                    "asset_column_names": ["Hydro3"],
                },
            ]
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir=".") as f:
        json.dump(config, f)
        path = f.name
    yield path
    os.unlink(path)


def _data_available() -> bool:
    """Check if the required data files exist."""
    return os.path.exists("data/clean/renewables.csv")


@pytest.mark.skipif(not _data_available(), reason="Data files not present")
class TestIntegration:
    """Integration tests that require real data."""

    def test_env_reset_and_step(self, config_path: str) -> None:
        """Test that the environment can be created, reset, and stepped."""
        import wandb

        # Disable wandb for testing
        os.environ["WANDB_MODE"] = "disabled"
        wandb.init(mode="disabled")

        from vpp_bidding.env.vpp_env import VPPBiddingEnv

        env = VPPBiddingEnv(
            config_path=config_path,
            log_level="WARNING",
            env_type="training",
            seed=42,
            render_mode="fast_training",
        )

        # Test reset
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert "asset_data_forecast" in obs
        assert "predicted_market_prices" in obs
        assert obs["asset_data_forecast"].shape == (6,)
        assert obs["predicted_market_prices"].shape == (6,)

        # Test step with random action
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs2, dict)
        assert isinstance(reward, float)
        assert terminated is True  # 1-step episodes
        assert truncated is False
        assert "step_reward" in info

        wandb.finish()

    def test_multiple_episodes(self, config_path: str) -> None:
        """Test running 3 full episodes (reset + step)."""
        import wandb

        os.environ["WANDB_MODE"] = "disabled"
        wandb.init(mode="disabled")

        from vpp_bidding.env.vpp_env import VPPBiddingEnv

        env = VPPBiddingEnv(
            config_path=config_path,
            log_level="WARNING",
            env_type="training",
            seed=42,
            render_mode="fast_training",
        )

        rewards = []
        for _episode in range(3):
            _obs, _ = env.reset()
            action = env.action_space.sample()
            _obs, reward, terminated, _truncated, _info = env.step(action)
            rewards.append(reward)
            assert terminated is True

        # Rewards should be in valid range [0, 1]
        for r in rewards:
            assert 0.0 <= r <= 1.0, f"Reward {r} out of expected range"

        wandb.finish()

    def test_short_ppo_training(self, config_path: str) -> None:
        """Smoke test: train PPO for 5 steps to verify the full pipeline works."""
        import wandb

        os.environ["WANDB_MODE"] = "disabled"
        wandb.init(mode="disabled")

        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor

        from vpp_bidding.env.vpp_env import VPPBiddingEnv

        env = VPPBiddingEnv(
            config_path=config_path,
            log_level="WARNING",
            env_type="training",
            seed=42,
            render_mode="fast_training",
        )
        env = Monitor(env)

        model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            verbose=0,
        )

        # Train for 4 steps (2 episodes x 1 step each, collected in 2 n_steps)
        model.learn(total_timesteps=4)

        # Verify model can predict
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (12,)

        wandb.finish()
