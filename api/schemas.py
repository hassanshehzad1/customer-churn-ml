"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    """Schema for customer data input for churn prediction.

    This matches the 19 raw columns that the model expects before preprocessing.
    Note: TotalCharges is kept as a string because our preprocessing pipeline
    handles cleaning it (converting empty strings to 0.0 and then to float).
    """

    gender: str = Field(..., description="Customer gender: Male or Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether customer is a senior citizen: 0 or 1")
    Partner: str = Field(..., description="Whether customer has a partner: Yes or No")
    Dependents: str = Field(..., description="Whether customer has dependents: Yes or No")
    tenure: int = Field(..., ge=0, description="Number of months the customer has been with the company")
    PhoneService: str = Field(..., description="Whether customer has phone service: Yes or No")
    MultipleLines: str = Field(..., description="Whether customer has multiple lines: Yes, No, or No phone service")
    InternetService: str = Field(..., description="Internet service provider: DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Whether customer has online security: Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Whether customer has online backup: Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Whether customer has device protection: Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Whether customer has tech support: Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Whether customer has streaming TV: Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Whether customer has streaming movies: Yes, No, or No internet service")
    Contract: str = Field(..., description="Contract type: Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., description="Whether customer has paperless billing: Yes or No")
    PaymentMethod: str = Field(..., description="Payment method: Electronic check, Mailed check, Bank transfer (automatic), or Credit card (automatic)")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges amount")
    TotalCharges: str = Field(..., description="Total charges amount (kept as string for preprocessing pipeline to handle empty strings)")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    """Schema for churn prediction response."""

    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn (0.0 to 1.0)")
    churn_prediction: str = Field(..., description="Churn prediction: Yes or No")
    threshold_used: float = Field(..., ge=0, le=1, description="Threshold used for binary classification")

    model_config = {
        "json_schema_extra": {
            "example": {
                "churn_probability": 0.65,
                "churn_prediction": "Yes",
                "threshold_used": 0.3
            }
        }
    }
