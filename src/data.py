
"""Data loading and cleaning for the IBM Telco dataset."""

import pandas as pd

from src.config import DATA_PATH, TARGET


def load_raw_data(path=DATA_PATH) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean customer data and encode the churn target."""

    # Work on a copy so the original DataFrame stays unchanged.
    clean = df.copy()

    # Convert TotalCharges to numeric values.
    # Blank strings and invalid values become NaN.
    clean["TotalCharges"] = pd.to_numeric(
        clean["TotalCharges"].str.strip(),
        errors="coerce",
    )

    # Keep only one record for each customer ID.
    clean = clean.drop_duplicates(
        subset="customerID"
    ).reset_index(drop=True)

    # Fill missing charges with the median to retain customer records.
    clean["TotalCharges"] = clean["TotalCharges"].fillna(
        clean["TotalCharges"].median()
    )

    # Convert the churn label into a binary target.
    clean[TARGET] = clean[TARGET].map({
        "No": 0,
        "Yes": 1,
    })

    return clean
