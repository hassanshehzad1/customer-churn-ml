"""Unit tests for model training and evaluation functions."""

import os
import numpy as np
import pandas as pd
import pytest
from joblib import load

from src.data.load_data import load_raw_data
from src.data.split import split_data


# Skip decorator for tests requiring the raw dataset
skip_if_no_raw_data = pytest.mark.skipif(
    not os.path.exists("data/raw/telco_customer_churn.csv"),
    reason="raw dataset not available (CI)"
)


@pytest.fixture
def pipeline():
    """Fixture to load the pipeline once and reuse across tests."""
    return load("models/churn_pipeline.joblib")


@pytest.fixture
def X_test():
    """Fixture to load X_test data once and reuse across tests."""
    df = load_raw_data()
    _, X_test, _, _ = split_data(df)
    return X_test


def test_pipeline_file_exists():
    """Test that the pipeline file exists on disk."""
    assert os.path.exists("models/churn_pipeline.joblib"), \
        "Pipeline file models/churn_pipeline.joblib should exist"


def test_pipeline_loads_without_error(pipeline):
    """Test that the pipeline loads without error and is not None."""
    assert pipeline is not None, "Pipeline should load successfully and not be None"


@skip_if_no_raw_data
def test_pipeline_predicts_on_real_test_data(pipeline, X_test):
    """Test that the pipeline can predict on real test data with correct output shape."""
    predictions = pipeline.predict_proba(X_test)

    # Should have shape (n_test_samples, 2) - 2 columns for class 0 and class 1 probabilities
    assert predictions.shape[0] == len(X_test), \
        f"Predictions should have {len(X_test)} rows, got {predictions.shape[0]}"
    assert predictions.shape[1] == 2, \
        f"Predictions should have 2 columns (probabilities for each class), got {predictions.shape[1]}"


@skip_if_no_raw_data
def test_predictions_are_valid_probabilities(pipeline, X_test):
    """Test that all predicted probabilities are between 0.0 and 1.0 inclusive."""
    predictions = pipeline.predict_proba(X_test)

    # Check that all probabilities are in valid range [0, 1]
    assert np.all(predictions >= 0.0), "All probabilities should be >= 0.0"
    assert np.all(predictions <= 1.0), "All probabilities should be <= 1.0"


def test_pipeline_handles_single_row_prediction(pipeline):
    """Test that the pipeline can predict on a single-row DataFrame."""
    # Create a single-row DataFrame with all 19 columns
    single_customer = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 73.35,
        "TotalCharges": "832.45"
    }])

    predictions = pipeline.predict_proba(single_customer)

    # Should return (1, 2) shape for single row
    assert predictions.shape == (1, 2), \
        f"Single-row prediction should have shape (1, 2), got {predictions.shape}"


@skip_if_no_raw_data
def test_pipeline_is_deterministic(pipeline, X_test):
    """Test that the pipeline produces identical predictions for the same input.

    This test protects against accidental randomness in the model or pipeline.
    If the model has stochastic components (e.g., dropout, random forest without fixed seed),
    running predict_proba() twice on the same data would give different results.
    A deterministic model should always give the same output for the same input.
    """
    predictions_1 = pipeline.predict_proba(X_test)
    predictions_2 = pipeline.predict_proba(X_test)

    # Assert that the two predictions are identical
    assert np.array_equal(predictions_1, predictions_2), \
        "Pipeline should produce identical predictions for the same input (deterministic behavior)"
