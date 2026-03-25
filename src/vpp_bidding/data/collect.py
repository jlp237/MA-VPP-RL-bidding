"""Data collection pipeline for VPP bidding data.

Sources:
- regelleistung.net: FCR tenders, results, and anonymous bids
- SMARD (smard.de): Wholesale electricity prices
- SimBench: Renewable generation profiles (wind, hydro, solar)
"""

from pathlib import Path


def fetch_fcr_tenders(output_dir: Path) -> None:
    """Download FCR tender data from regelleistung.net."""
    raise NotImplementedError("Data collection from regelleistung.net API")


def fetch_wholesale_prices(output_dir: Path, start: str, end: str) -> None:
    """Download wholesale prices from SMARD.

    Args:
        output_dir: Directory to write the downloaded data.
        start: Start date string (e.g. "2020-01-01").
        end: End date string (e.g. "2022-12-31").
    """
    raise NotImplementedError("SMARD data download")


def fetch_renewable_profiles(output_dir: Path) -> None:
    """Load SimBench renewable generation profiles."""
    raise NotImplementedError("SimBench profile loading")
