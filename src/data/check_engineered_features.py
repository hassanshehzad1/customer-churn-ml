"""Script to verify engineered features and their relationship with churn."""

import pandas as pd
from src.data.load_data import load_raw_data
from src.features.preprocessing import engineer_features


def main():
    # Load raw data
    print("=" * 60)
    print("Loading raw data...")
    print("=" * 60)
    df = load_raw_data()

    # Apply feature engineering
    print("\n" + "=" * 60)
    print("Applying feature engineering...")
    print("=" * 60)
    df_engineered = engineer_features(df)
    print(f"Shape after feature engineering: {df_engineered.shape}")

    # Display first 10 rows with key columns
    print("\n" + "=" * 60)
    print("Sample rows (first 10) with engineered features:")
    print("=" * 60)
    display_columns = [
        "tenure",
        "Contract",
        "MonthlyCharges",
        "TotalCharges",
        "is_new_customer",
        "high_risk_contract",
        "avg_monthly_spend_ratio",
        "num_services",
        "Churn"
    ]
    print(df_engineered[display_columns].head(10).to_string())

    # Summary statistics for numeric engineered features
    print("\n" + "=" * 60)
    print("Summary statistics for numeric engineered features:")
    print("=" * 60)
    numeric_features = ["avg_monthly_spend_ratio", "num_services"]
    print(df_engineered[numeric_features].describe().loc[["mean", "min", "max"]].to_string())

    # Value counts for binary features
    print("\n" + "=" * 60)
    print("Value counts for binary features:")
    print("=" * 60)
    print("\nis_new_customer:")
    print(df_engineered["is_new_customer"].value_counts().to_string())
    print("\nhigh_risk_contract:")
    print(df_engineered["high_risk_contract"].value_counts().to_string())

    # Churn rate comparison by is_new_customer
    print("\n" + "=" * 60)
    print("Churn rate by is_new_customer:")
    print("=" * 60)
    churn_by_new = df_engineered.groupby("is_new_customer")["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    )
    print(f"is_new_customer = 0: {churn_by_new[0]:.2f}%")
    print(f"is_new_customer = 1: {churn_by_new[1]:.2f}%")
    print(f"Difference: {churn_by_new[1] - churn_by_new[0]:.2f} percentage points")

    # Churn rate comparison by high_risk_contract
    print("\n" + "=" * 60)
    print("Churn rate by high_risk_contract:")
    print("=" * 60)
    churn_by_risk = df_engineered.groupby("high_risk_contract")["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    )
    print(f"high_risk_contract = 0: {churn_by_risk[0]:.2f}%")
    print(f"high_risk_contract = 1: {churn_by_risk[1]:.2f}%")
    print(f"Difference: {churn_by_risk[1] - churn_by_risk[0]:.2f} percentage points")

    # Churn rate comparison by avg_monthly_spend_ratio (tertiles)
    print("\n" + "=" * 60)
    print("Churn rate by avg_monthly_spend_ratio (tertiles):")
    print("=" * 60)
    df_engineered["spend_ratio_tertile"] = pd.qcut(
        df_engineered["avg_monthly_spend_ratio"],
        q=3,
        labels=["Low", "Medium", "High"]
    )
    churn_by_spend = df_engineered.groupby("spend_ratio_tertile")["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    )
    for tertile in ["Low", "Medium", "High"]:
        count = (df_engineered["spend_ratio_tertile"] == tertile).sum()
        print(f"{tertile}: {churn_by_spend[tertile]:.2f}% (n={count})")
    print(f"Range (High - Low): {churn_by_spend['High'] - churn_by_spend['Low']:.2f} percentage points")

    # Churn rate comparison by num_services
    print("\n" + "=" * 60)
    print("Churn rate by num_services:")
    print("=" * 60)
    churn_by_services = df_engineered.groupby("num_services")["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    )
    for services in sorted(df_engineered["num_services"].unique()):
        count = (df_engineered["num_services"] == services).sum()
        print(f"{services} services: {churn_by_services[services]:.2f}% (n={count})")
    print(f"Range (max - min): {churn_by_services.max() - churn_by_services.min():.2f} percentage points")

    print("\n" + "=" * 60)
    print("Feature engineering verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
