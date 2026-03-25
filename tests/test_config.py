"""Tests for TOML configuration loading."""

from pathlib import Path

import pytest

from vpp_bidding.config import (
    AppConfig,
    load_config,
)


class TestLoadConfig:
    def test_load_minimal_config(self, tmp_path: Path) -> None:
        """A TOML with no sections should produce defaults."""
        config_file = tmp_path / "empty.toml"
        config_file.write_text("")
        config = load_config(config_file)
        assert isinstance(config, AppConfig)
        assert config.markets.enabled == ["fcr"]
        assert config.training.seed == 42

    def test_load_with_markets_section(self, tmp_path: Path) -> None:
        toml = '[markets]\nenabled = ["fcr", "day_ahead"]\n'
        config_file = tmp_path / "markets.toml"
        config_file.write_text(toml)
        config = load_config(config_file)
        assert config.markets.enabled == ["fcr", "day_ahead"]

    def test_load_with_training_section(self, tmp_path: Path) -> None:
        toml = "[training]\ntotal_timesteps = 5000\nseed = 123\n"
        config_file = tmp_path / "training.toml"
        config_file.write_text(toml)
        config = load_config(config_file)
        assert config.training.total_timesteps == 5000
        assert config.training.seed == 123

    def test_load_with_time_section(self, tmp_path: Path) -> None:
        toml = "[time]\nhist_window_size = 3\nforecast_window_size = 2\n"
        config_file = tmp_path / "time.toml"
        config_file.write_text(toml)
        config = load_config(config_file)
        assert config.time.hist_window_size == 3
        assert config.time.forecast_window_size == 2

    def test_load_with_wandb_section(self, tmp_path: Path) -> None:
        toml = '[wandb]\nproject = "test-project"\nmode = "disabled"\n'
        config_file = tmp_path / "wandb.toml"
        config_file.write_text(toml)
        config = load_config(config_file)
        assert config.wandb.project == "test-project"
        assert config.wandb.mode == "disabled"

    def test_load_with_assets(self, tmp_path: Path) -> None:
        toml = """
[[assets.hydro]]
type = "run-of-river"
max_capacity_mw = 10.0
quantity = 2
max_fcr_capacity_share = 0.5
asset_column_names = ["Hydro1", "Hydro2"]
"""
        config_file = tmp_path / "assets.toml"
        config_file.write_text(toml)
        config = load_config(config_file)
        assert "hydro" in config.assets
        assert len(config.assets["hydro"]) == 1
        assert config.assets["hydro"][0].max_capacity_mw == 10.0

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.toml"))

    def test_default_values(self, tmp_path: Path) -> None:
        config_file = tmp_path / "defaults.toml"
        config_file.write_text("")
        config = load_config(config_file)
        assert config.fcr.slots_per_day == 6
        assert config.fcr.steps_per_slot == 16
        assert config.data.renewables == "data/clean/renewables.csv"
