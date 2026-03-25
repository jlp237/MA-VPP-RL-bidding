"""Tests for VPP asset configuration and capacity simulation."""

import numpy as np
import pandas as pd

from vpp_bidding.domain.models import AssetConfig, VPPConfig
from vpp_bidding.env.vpp import VPP


class TestVPPConfigure:
    def test_basic_configuration(
        self,
        sample_vpp_config: VPPConfig,
        sample_renewables_df: pd.DataFrame,
    ) -> None:
        vpp = VPP(sample_vpp_config, sample_renewables_df)
        assert vpp.asset_data is not None
        assert vpp.asset_data_fcr is not None
        assert len(vpp.asset_data) == 96
        assert len(vpp.asset_data_fcr) == 96

    def test_total_column_is_sum_of_assets(
        self,
        sample_vpp_config: VPPConfig,
        sample_renewables_df: pd.DataFrame,
    ) -> None:
        vpp = VPP(sample_vpp_config, sample_renewables_df)

        # Total column should equal sum of all other columns
        non_total_cols = [c for c in vpp.asset_data.columns if c != "Total"]
        expected_total = vpp.asset_data[non_total_cols].sum(axis=1)
        pd.testing.assert_series_equal(
            vpp.asset_data["Total"],
            expected_total,
            check_names=False,
        )

    def test_fcr_total_column_is_sum(
        self,
        sample_vpp_config: VPPConfig,
        sample_renewables_df: pd.DataFrame,
    ) -> None:
        vpp = VPP(sample_vpp_config, sample_renewables_df)

        non_total_cols = [c for c in vpp.asset_data_fcr.columns if c != "Total"]
        expected = vpp.asset_data_fcr[non_total_cols].sum(axis=1)
        pd.testing.assert_series_equal(
            vpp.asset_data_fcr["Total"],
            expected,
            check_names=False,
        )

    def test_fcr_capacity_is_fraction_of_total(
        self,
        sample_vpp_config: VPPConfig,
        sample_renewables_df: pd.DataFrame,
    ) -> None:
        vpp = VPP(sample_vpp_config, sample_renewables_df)
        # max_fcr_share = 0.5 in the fixture
        ratio = vpp.asset_data_fcr["Total"] / vpp.asset_data["Total"]
        assert np.allclose(ratio.values, 0.5, atol=1e-10)

    def test_capacity_scales_with_quantity(
        self,
        sample_renewables_df: pd.DataFrame,
    ) -> None:
        """Two identical assets with quantity=1 vs one with quantity=2."""
        single = AssetConfig(
            asset_type="wind",
            plant_type="onshore",
            max_capacity_mw=10.0,
            quantity=1,
            max_fcr_share=0.5,
            asset_column_names=["wind_1"],
        )
        double = AssetConfig(
            asset_type="wind",
            plant_type="onshore",
            max_capacity_mw=10.0,
            quantity=2,
            max_fcr_share=0.5,
            asset_column_names=["wind_1", "wind_2"],
        )

        vpp1 = VPP(VPPConfig(assets=[single]), sample_renewables_df)
        vpp2 = VPP(VPPConfig(assets=[double]), sample_renewables_df)

        assert vpp2.total_capacity.sum() > vpp1.total_capacity.sum()


class TestVPPSimulate:
    def test_simulate_returns_correct_length(
        self,
        sample_vpp_config: VPPConfig,
        sample_renewables_df: pd.DataFrame,
    ) -> None:
        vpp = VPP(sample_vpp_config, sample_renewables_df)
        start = sample_renewables_df.index[0]
        end = sample_renewables_df.index[-1]
        result = vpp.simulate(start, end)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 96

    def test_simulate_subset(
        self,
        sample_vpp_config: VPPConfig,
        sample_renewables_df: pd.DataFrame,
    ) -> None:
        vpp = VPP(sample_vpp_config, sample_renewables_df)
        start = sample_renewables_df.index[0]
        end = sample_renewables_df.index[15]  # first 16 steps
        result = vpp.simulate(start, end)
        assert len(result) == 16


class TestVPPEmptyConfig:
    def test_empty_assets(self, sample_renewables_df: pd.DataFrame) -> None:
        cfg = VPPConfig(assets=[])
        vpp = VPP(cfg, sample_renewables_df)
        assert "Total" in vpp.asset_data.columns
        assert np.all(vpp.asset_data["Total"].values == 0.0)
