# Customer Churn Prediction

A machine learning project to predict customer churn using binary classification. This project follows professional ML engineering practices with a clean, modular structure suitable for learning and production deployment.

## Setup Instructions (Windows PowerShell)

1. Create a virtual environment:
   ```powershell
   python -m venv .venv
   ```

2. Activate the virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   - For production/API use only (minimal dependencies):
     ```powershell
     pip install -r requirements.txt
     ```
   - For full local development (includes Jupyter, testing, visualization):
     ```powershell
     pip install -r requirements-dev.txt
     ```

## Project Structure

- `data/`: Raw and processed data files
- `notebooks/`: Jupyter notebooks for exploration and experimentation
- `src/`: Source code for data processing, feature engineering, and modeling
- `api/`: FastAPI application for serving predictions
- `tests/`: Unit tests for project components
- `models/`: Trained model artifacts
- `reports/`: Analysis reports and documentation
- `docs/`: Project documentation and phase documentation

## Dataset

This project uses the Telco Customer Churn dataset from Kaggle. The dataset maintains its own license and is not included in this repository. To run the project, download the dataset from Kaggle and place it in `data/raw/`.
