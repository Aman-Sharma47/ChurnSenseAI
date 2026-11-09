
"""Feature engineering and preprocessing for ChurnSense AI."""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features from existing customer data."""

    data = df.copy()

    # Estimate average monthly spending over the customer's tenure.
    data["AvgMonthlySpend"] = (
        data["TotalCharges"] / data["tenure"].clip(lower=1)
    )

    # Group customers according to their tenure.
    data["TenureGroup"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, float("inf")],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49+ months",
        ],
    ).astype(str)

    # Convert support availability into a binary feature.
    data["HasPremiumSupport"] = (
        data["TechSupport"] == "Yes"
    ).astype(int)

    # Identify customers using automatic payment methods.
    automatic_methods = [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    data["AutoPay"] = (
        data["PaymentMethod"].isin(automatic_methods)
    ).astype(int)

    return data


def model_columns(
    df: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """Separate model features into numerical and categorical columns."""

    # Exclude the customer identifier and prediction target.
    features = [
        column
        for column in df.columns
        if column not in {"customerID", "Churn"}
    ]

    # Identify numerical columns automatically.
    numeric = (
        df[features]
        .select_dtypes(include=["number"])
        .columns.tolist()
    )

    # Remaining features are treated as categorical.
    categorical = [
        column
        for column in features
        if column not in numeric
    ]

    return features, numeric, categorical


def make_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Scale numerical features and encode categorical features."""

    _, numeric, categorical = model_columns(df)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def make_pipeline(model, df: pd.DataFrame) -> Pipeline:
    """Combine preprocessing and the selected machine-learning model."""

    preprocessor = make_preprocessor(df)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
