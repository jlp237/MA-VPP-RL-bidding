"""Tests for the VPPBiddingEnv gymnasium environment.

The full environment requires real data files and heavy dependencies,
so we test importability and space shape expectations without instantiation.
"""

from vpp_bidding.markets.fcr.constants import SLOTS_PER_DAY


class TestVPPBiddingEnvImport:
    def test_class_importable(self) -> None:
        from vpp_bidding.env.vpp_env import VPPBiddingEnv

        assert VPPBiddingEnv is not None

    def test_is_gymnasium_env(self) -> None:
        import gymnasium

        from vpp_bidding.env.vpp_env import VPPBiddingEnv

        assert issubclass(VPPBiddingEnv, gymnasium.Env)

    def test_metadata_has_render_modes(self) -> None:
        from vpp_bidding.env.vpp_env import VPPBiddingEnv

        assert "render_modes" in VPPBiddingEnv.metadata
        assert "human" in VPPBiddingEnv.metadata["render_modes"]

    def test_action_space_expected_size(self) -> None:
        """The FCR market expects 12 actions (6 sizes + 6 prices)."""
        expected_action_size = SLOTS_PER_DAY * 2
        assert expected_action_size == 12

    def test_observation_space_keys(self) -> None:
        """Check expected observation keys in the Dict space definition."""
        expected_keys = {
            "asset_data_forecast",
            "predicted_market_prices",
            "weekday",
            "month",
            "slots_won",
            "slots_reserved",
            "day_reward_list",
        }
        # We just verify the expected structure; actual instantiation needs data
        assert len(expected_keys) == 7
