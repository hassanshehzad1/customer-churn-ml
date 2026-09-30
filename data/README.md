# Dataset: Telco Customer Churn

## Source
- **Platform:** Kaggle
- **Dataset name:** Telco Customer Churn
- **URL:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Download date:** 2026-09-26
- **License:** Public dataset, for educational/research use

## Basic Info
- **Rows:** 7043
- **Columns:** 21
- **File location:** `data/raw/telco_customer_churn.csv` (not committed to git — see `.gitignore`)

## Unit of Observation
Each row represents **one customer** of the telecom company.

## Target Variable
- **Column:** `Churn`
- **Type:** Binary (categorical: "Yes" / "No")
- **Positive class:** `"Yes"` (customer churned / left the company)
- **Business meaning:** Predicting churn lets the retention team proactively contact at-risk customers.

## Columns to Exclude from Modeling
- `customerID` — unique identifier, no predictive signal, must not be used as a feature.

## Data Dictionary

| Column | Type | Description | Notes |
|---|---|---|---|
| customerID | String | Unique customer identifier | Exclude from features |
| gender | Categorical | Male / Female | |
| SeniorCitizen | Binary (0/1) | Whether customer is a senior citizen | Already numeric |
| Partner | Categorical | Has a partner (Yes/No) | |
| Dependents | Categorical | Has dependents (Yes/No) | |
| tenure | Numeric | Number of months as a customer | |
| PhoneService | Categorical | Has phone service (Yes/No) | |
| MultipleLines | Categorical | Multiple phone lines (Yes/No/No phone service) | |
| InternetService | Categorical | DSL / Fiber optic / No | |
| OnlineSecurity | Categorical | Yes/No/No internet service | |
| OnlineBackup | Categorical | Yes/No/No internet service | |
| DeviceProtection | Categorical | Yes/No/No internet service | |
| TechSupport | Categorical | Yes/No/No internet service | |
| StreamingTV | Categorical | Yes/No/No internet service | |
| StreamingMovies | Categorical | Yes/No/No internet service | |
| Contract | Categorical | Month-to-month / One year / Two year | |
| PaperlessBilling | Categorical | Yes/No | |
| PaymentMethod | Categorical | Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic) | |
| MonthlyCharges | Numeric | Current monthly charge (USD) | |
| TotalCharges | Numeric | Total amount charged to date | Stored as text in raw CSV; contains blank/space values for customers with tenure = 0 — must be cleaned in Phase 2 |
| Churn | Categorical (Target) | Yes/No | Positive class = "Yes" |

## Known Data Quality Issues (to handle in Phase 2)
- `TotalCharges` is stored as a string/object type, not numeric, due to blank values for new customers (tenure = 0). Needs conversion + missing value handling.
- No duplicate customerIDs expected, but must be verified in ingestion.

## Notes
- Correlation-based observations from EDA must not be interpreted as causal (see Phase 3).
- Class imbalance is expected: churn is typically the minority class (~26-27% in this dataset). This affects metric choice (Phase 5+) and threshold selection (Phase 7).
