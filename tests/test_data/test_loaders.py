"""Tests for data loading utilities."""

from pathlib import Path

import pandas as pd
import pytest

from vpp_bidding.data.loaders import load_csv, load_test_set


class TestLoadCsv:
    def test_load_csv_with_semicolon_separator(self, tmp_path: Path) -> None:
        csv_content = (
            "datetime;col_a;col_b\n2021-01-01 00:00:00;1.0;2.0\n2021-01-01 00:15:00;3.0;4.0\n"
        )
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        df = load_csv(csv_file, sep=";")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["col_a", "col_b"]
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_load_csv_with_comma_separator(self, tmp_path: Path) -> None:
        csv_content = "datetime,value\n2021-06-01 12:00:00,42.5\n2021-06-01 12:15:00,43.1\n"
        csv_file = tmp_path / "test_comma.csv"
        csv_file.write_text(csv_content)

        df = load_csv(csv_file, sep=",")
        assert len(df) == 2
        assert "value" in df.columns

    def test_load_csv_parses_dates(self, tmp_path: Path) -> None:
        csv_content = "datetime;val\n2021-03-15 10:30:00;1.0\n"
        csv_file = tmp_path / "dates.csv"
        csv_file.write_text(csv_content)

        df = load_csv(csv_file, sep=";")
        assert pd.api.types.is_datetime64_any_dtype(df.index)

    def test_load_csv_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_csv(Path("/nonexistent/path.csv"))


class TestLoadTestSet:
    def test_load_test_set_returns_date_strings(self, tmp_path: Path) -> None:
        csv_content = "2021-01-01\n2021-01-02\n2021-01-03\n"
        csv_file = tmp_path / "test_set.csv"
        csv_file.write_text(csv_content)

        dates = load_test_set(csv_file)
        assert isinstance(dates, list)
        assert len(dates) == 3
        assert dates[0] == "2021-01-01"
        assert dates[2] == "2021-01-03"

    def test_load_test_set_single_date(self, tmp_path: Path) -> None:
        csv_content = "2022-12-25\n"
        csv_file = tmp_path / "single.csv"
        csv_file.write_text(csv_content)

        dates = load_test_set(csv_file)
        assert len(dates) == 1
        assert dates[0] == "2022-12-25"

    def test_load_test_set_all_strings(self, tmp_path: Path) -> None:
        csv_content = "2021-01-01\n2021-01-02\n"
        csv_file = tmp_path / "strings.csv"
        csv_file.write_text(csv_content)

        dates = load_test_set(csv_file)
        for d in dates:
            assert isinstance(d, str)
