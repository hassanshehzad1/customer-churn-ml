"""Unit tests for API endpoints and functionality."""

from fastapi.testclient import TestClient
import pytest

from api.main import app


@pytest.fixture
def client():
    """Fixture to create a TestClient instance for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def valid_customer_payload():
    """Fixture with a valid customer payload containing all 19 required fields."""
    return {
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
    }


def test_health_endpoint_returns_200(client):
    """Test that GET /health returns status code 200 and correct body."""
    response = client.get("/health")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert response.json() == {"status": "ok"}, f"Expected {{'status': 'ok'}}, got {response.json()}"


def test_model_info_endpoint_returns_200(client):
    """Test that GET /model-info returns 200 and contains model_type."""
    response = client.get("/model-info")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "model_type" in data, "Response should contain 'model_type' key"


def test_predict_with_valid_data_returns_200(client, valid_customer_payload):
    """Test that POST /predict with valid data returns 200."""
    response = client.post("/predict", json=valid_customer_payload)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"


def test_predict_response_has_expected_fields(client, valid_customer_payload):
    """Test that /predict response has exactly the expected fields."""
    response = client.post("/predict", json=valid_customer_payload)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()

    expected_keys = {"churn_probability", "churn_prediction", "threshold_used"}
    assert set(data.keys()) == expected_keys, \
        f"Expected keys {expected_keys}, got {set(data.keys())}"


def test_predict_response_values_are_valid(client, valid_customer_payload):
    """Test that /predict response values are in valid ranges."""
    response = client.post("/predict", json=valid_customer_payload)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()

    # Check churn_probability is between 0 and 1
    churn_prob = data["churn_probability"]
    assert 0.0 <= churn_prob <= 1.0, \
        f"churn_probability should be between 0 and 1, got {churn_prob}"

    # Check churn_prediction is either "Yes" or "No"
    churn_pred = data["churn_prediction"]
    assert churn_pred in ["Yes", "No"], \
        f"churn_prediction should be 'Yes' or 'No', got {churn_pred}"

    # Check threshold_used equals 0.3
    threshold = data["threshold_used"]
    assert threshold == 0.3, f"threshold_used should be 0.3, got {threshold}"


def test_predict_with_missing_field_returns_422(client, valid_customer_payload):
    """Test that POST /predict with missing required field returns 422."""
    # Remove the 'tenure' field to make it invalid
    invalid_payload = valid_customer_payload.copy()
    del invalid_payload["tenure"]

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422, \
        f"Expected 422 (validation error), got {response.status_code}"


def test_predict_with_wrong_type_returns_422(client, valid_customer_payload):
    """Test that POST /predict with wrong data type returns 422."""
    # Change tenure from int to string
    invalid_payload = valid_customer_payload.copy()
    invalid_payload["tenure"] = "not a number"

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422, \
        f"Expected 422 (validation error), got {response.status_code}"
