"""Data preprocessing pipeline.

Transforms raw data into clean CSVs ready for the VPP environment.
"""

from pathlib import Path


def preprocess_all(raw_dir: Path, clean_dir: Path) -> None:
    """Run the full preprocessing pipeline.

    Args:
        raw_dir: Directory containing raw downloaded data.
        clean_dir: Directory to write cleaned output CSVs.
    """
    raise NotImplementedError("Full preprocessing pipeline")


def clean_fcr_data(raw_dir: Path, clean_dir: Path) -> None:
    """Clean and merge FCR tender, result, and bid data."""
    raise NotImplementedError


def create_time_features(clean_dir: Path) -> None:
    """Generate time feature DataFrame (weekday, month, etc.)."""
    raise NotImplementedError


def create_train_test_split(clean_dir: Path, test_days: int = 70) -> None:
    """Create train/validation/test date splits.

    Args:
        clean_dir: Directory containing clean data.
        test_days: Number of days to reserve for the test set.
    """
    raise NotImplementedError
