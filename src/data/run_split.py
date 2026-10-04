"""Run train/test split on the raw telco churn dataset."""

import sys
sys.path.append('.')

from src.data.load_data import load_raw_data
from src.data.split import split_data


def main():
    """Load the raw data and perform train/test split."""
    print("Loading raw data...")
    df = load_raw_data()

    print("\nPerforming train/test split...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("\n" + "="*60)
    print("SPLIT COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
