"""Data loading utilities for the VPP bidding environment."""

import logging
from pathlib import Path

import pandas as pd

from vpp_bidding.config import AppConfig

logger = logging.getLogger(__name__)


def load_csv(path: Path, sep: str = ";") -> pd.DataFrame:
    """Load a CSV with datetime index.

    Args:
        path: Path to the CSV file.
        sep: Column separator.

    Returns:
        DataFrame with a DatetimeIndex parsed from the first column.
    """
    df = pd.read_csv(path, sep=sep, index_col=0, parse_dates=True)
    logger.debug("Loaded %s: %d rows, %d columns", path.name, len(df), len(df.columns))
    return df


def load_training_data(config: AppConfig) -> dict[str, pd.DataFrame]:
    """Load all training data files specified in config.

    Args:
        config: Application configuration with data paths.

    Returns:
        Dictionary mapping data names to DataFrames.
    """
    data_config = config.data
    datasets: dict[str, pd.DataFrame] = {}

    file_map = {
        "renewables": data_config.renewables,
        "tenders": data_config.tenders,
        "market_results": data_config.market_results,
        "bids": data_config.bids,
        "time_features": data_config.time_features,
        "market_prices": data_config.market_prices,
    }

    for name, path_str in file_map.items():
        path = Path(path_str)
        if not path.exists():
            logger.warning("Data file not found: %s", path)
            continue
        datasets[name] = load_csv(path)

    logger.info("Loaded %d/%d datasets", len(datasets), len(file_map))
    return datasets


def load_test_set(path: Path) -> list[str]:
    """Load test set dates.

    Args:
        path: Path to a CSV containing test set date strings.

    Returns:
        List of date strings for the test set.
    """
    df = pd.read_csv(path, header=None)
    dates = df.iloc[:, 0].astype(str).tolist()
    logger.info("Loaded %d test set dates from %s", len(dates), path.name)
    return dates
