"""Functions for loading raw data from various sources."""

import pandas as pd
from src.config import RAW_DATA_PATH


def load_raw_data(path: str = None) -> pd.DataFrame:
    """
    Load raw data from a CSV file.

    This function loads the telco customer churn dataset from a CSV file.
    If no path is provided, it uses the default RAW_DATA_PATH from config.

    Parameters
    ----------
    path : str, optional
        Path to the CSV file. If None, uses RAW_DATA_PATH from config.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame containing the raw data.

    Notes
    -----
    This function prints the shape of the loaded DataFrame for verification.
    """
    if path is None:
        path = RAW_DATA_PATH

    df = pd.read_csv(path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df
