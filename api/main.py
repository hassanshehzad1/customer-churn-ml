"""FastAPI main application for serving churn predictions."""

import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from joblib import load

from api.schemas import CustomerData, PredictionResponse

# Decision threshold from Phase 7 (optimized for recall-focused churn prediction)
THRESHOLD = 0.3

# Load the model and metadata at module level.
# This is loaded once when the module is imported, which happens at server startup.
# FastAPI's --reload mode will reload the entire module when code changes, so this is safe.
# For production with multiple workers, you'd use a startup event to load it once per worker.
try:
    model = load("models/churn_pipeline.joblib")
    with open("models/model_metadata.json", "r") as f:
        model_metadata = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model or metadata: {e}")

app = FastAPI()


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer_data: CustomerData):
    """Predict churn probability for a customer.

    Accepts customer data with 19 raw features, applies the preprocessing pipeline,
    and returns churn probability and binary prediction using threshold=0.3.
    """
    try:
        # Convert Pydantic model to DataFrame with correct column names
        # The pipeline expects these exact 19 columns in this order
        data_dict = customer_data.model_dump()
        df = pd.DataFrame([data_dict])

        # Get churn probability (probability of class 1 = churn)
        churn_probability = model.predict_proba(df)[0, 1]

        # Apply threshold to get binary prediction
        churn_prediction = "Yes" if churn_probability >= THRESHOLD else "No"

        return PredictionResponse(
            churn_probability=float(churn_probability),
            churn_prediction=churn_prediction,
            threshold_used=THRESHOLD
        )
    except Exception as e:
        # Catch any errors from the pipeline (e.g., unexpected data values)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/model-info")
def model_info():
    """Return metadata about the currently loaded model.

    This includes model type, training date, decision threshold,
    dataset sizes, and test set performance metrics.
    """
    return model_metadata
