
"""Dataset-independent validation and preprocessing for ChurnSense AI."""

from dataclasses import dataclass

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DatasetInfo:
    """Summary of a dataset and its suitability for classification."""

    rows: int
    columns: int
    target: str
    id_column: str | None
    numeric_features: list[str]
    categorical_features: list[str]
    missing_target_rows: int
    duplicate_rows: int
    class_counts: dict


def inspect_dataset(
    df: pd.DataFrame,
    target: str,
    id_column: str | None = None,
) -> DatasetInfo:
    """Validate a labelled dataset and summarize its structure."""

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    if df.columns.duplicated().any():
        raise ValueError("The dataset contains duplicate column names.")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found.")

    if id_column is not None and id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' was not found.")

    if id_column == target:
        raise ValueError("The ID column and target column must differ.")

    missing_target_rows = int(df[target].isna().sum())
    labelled = df.loc[df[target].notna()].copy()

    if labelled[target].nunique() < 2:
        raise ValueError(
            "Classification requires at least two target classes "
            "among rows with a known target."
        )

    excluded = {target}
    if id_column is not None:
        excluded.add(id_column)

    feature_columns = [
        col for col in labelled.columns if col not in excluded
    ]

    if not feature_columns:
        raise ValueError("No feature columns remain after exclusions.")

    # Remove columns with only one observed value.
    feature_columns = [
        col for col in feature_columns
        if labelled[col].nunique(dropna=True) > 1
    ]

    if not feature_columns:
        raise ValueError("No useful feature columns were found.")

    X = labelled[feature_columns]

    numeric_features = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = [
        col for col in feature_columns
        if col not in numeric_features
    ]

    return DatasetInfo(
        rows=len(df),
        columns=len(df.columns),
        target=target,
        id_column=id_column,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        missing_target_rows=missing_target_rows,
        duplicate_rows=int(df.duplicated().sum()),
        class_counts={
            str(key): int(value)
            for key, value in labelled[target].value_counts().items()
        },
    )



def prepare_training_data(
    df: pd.DataFrame,
    target: str,
    id_column: str | None = None,
):
    """Separate labelled rows, target, and model input features."""

    inspect_dataset(df, target, id_column)

    # Exclude rows without a known target.
    labelled = df.loc[df[target].notna()].copy()

    excluded = {target}
    if id_column is not None:
        excluded.add(id_column)

    X = labelled.drop(columns=list(excluded)).copy()
    y = labelled[target].copy()


    # Convert numeric-looking text columns when conversion is safe.
    for col in X.select_dtypes(include=["object", "string"]).columns:
        cleaned = X[col].astype("string").str.strip()
        non_missing = cleaned.dropna()
        non_blank = non_missing[non_missing != ""]

        if non_blank.empty:
            continue

        converted = pd.to_numeric(non_blank, errors="coerce")

        # Convert only if every nonblank value is numeric.
        if converted.notna().all():
            X[col] = pd.to_numeric(
                cleaned.replace("", pd.NA),
                errors="coerce",
            )


    # Remove constant columns using labelled rows only.
    useful_columns = [
        col for col in X.columns
        if X[col].nunique(dropna=True) > 1
    ]
    X = X[useful_columns]

    return X, y




def make_preprocessor(
    X: pd.DataFrame,
    add_missing_flags: bool = False,
) -> ColumnTransformer:
    """Build preprocessing that learns values from training data only.

    When add_missing_flags=True, explicitly preserve missingness
    information for both numeric and categorical features.
    """

    numeric_columns = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = [
        col for col in X.columns if col not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(
                strategy="median",
                add_indicator=not add_missing_flags,
            )),
            ("scaler", StandardScaler()),
        ])
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(
                strategy="most_frequent"
            )),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore"
            )),
        ])
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    if not transformers:
        raise ValueError("No usable numeric or categorical features found.")

    if add_missing_flags:
        from sklearn.impute import MissingIndicator

        transformers.append(
            ("missing_flags", MissingIndicator(
                features="all",
                error_on_new=False,
            ), X.columns.tolist())
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )



def make_model_pipeline(
    X_train: pd.DataFrame,
    estimator,
    add_missing_flags: bool = False,
) -> Pipeline:
    """Combine preprocessing and a classifier in one pipeline."""

    return Pipeline([
        (
            "preprocessor",
            make_preprocessor(
                X_train,
                add_missing_flags=add_missing_flags,
            ),
        ),
        ("model", estimator),
    ])

