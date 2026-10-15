"""Unit tests for data loading and validation functions."""

import pandas as pd
import pytest
from src.data.load_data import load_raw_data
from src.data.validate import validate_raw_data


@pytest.fixture
def raw_df():
    """Fixture to load the raw dataframe once and reuse across tests."""
    return load_raw_data()


def test_load_raw_data_returns_dataframe(raw_df):
    """Test that load_raw_data returns a non-empty pandas DataFrame."""
    assert isinstance(raw_df, pd.DataFrame), "load_raw_data should return a pandas DataFrame"
    assert len(raw_df) > 0, "DataFrame should not be empty"


def test_load_raw_data_has_expected_shape(raw_df):
    """Test that the loaded data has the expected shape (7043, 21)."""
    assert raw_df.shape == (7043, 21), f"Expected shape (7043, 21), got {raw_df.shape}"


def test_validate_raw_data_passes_on_clean_data(raw_df):
    """Test that validate_raw_data passes without exceptions on clean real data."""
    # Should not raise any exception
    validation_result = validate_raw_data(raw_df)

    # Check that the returned dict has expected keys
    expected_keys = {
        "shape",
        "duplicates",
        "missing_per_column",
        "unexpected_churn_values",
        "blank_totalcharges_count"
    }
    assert set(validation_result.keys()) == expected_keys, \
        f"Validation result should have keys {expected_keys}, got {set(validation_result.keys())}"


def test_validate_raw_data_raises_on_missing_target_column():
    """Test that validate_raw_data raises AssertionError when Churn column is missing."""
    # Create a fake DataFrame without the Churn column
    fake_df = pd.DataFrame({
        "customerID": ["1", "2", "3"],
        "gender": ["Male", "Female", "Male"],
        "tenure": [1, 2, 3]
    })

    with pytest.raises(AssertionError) as exc_info:
        validate_raw_data(fake_df)

    assert "Churn" in str(exc_info.value), "Error message should mention missing Churn column"


def test_validate_raw_data_raises_on_missing_id_column():
    """Test that validate_raw_data raises AssertionError when customerID column is missing."""
    # Create a fake DataFrame without the customerID column
    fake_df = pd.DataFrame({
        "Churn": ["Yes", "No", "Yes"],
        "gender": ["Male", "Female", "Male"],
        "tenure": [1, 2, 3]
    })

    with pytest.raises(AssertionError) as exc_info:
        validate_raw_data(fake_df)

    assert "customerID" in str(exc_info.value), "Error message should mention missing customerID column"


def test_churn_column_only_has_expected_values(raw_df):
    """Test that the Churn column only contains 'Yes' and 'No' values."""
    unique_churn_values = set(raw_df["Churn"].unique())
    expected_values = {"Yes", "No"}
    assert unique_churn_values == expected_values, \
        f"Churn column should only contain {expected_values}, got {unique_churn_values}"
