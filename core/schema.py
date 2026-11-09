"""Schema detection and target normalisation for arbitrary churn datasets."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

import pandas as pd

TARGET_HINTS = (
    "churn", "attrition", "attrited", "exited", "left",
    "cancel", "leave", "lost", "terminated", "unsubscribe",
)
POSITIVE_TOKENS = {
    "yes", "y", "true", "1", "churn", "churned", "left", "exited",
    "cancelled", "canceled", "attrited", "attrited customer",
    "lost", "closed", "terminated",
}
NEGATIVE_TOKENS = {
    "no", "n", "false", "0", "stayed", "active", "retained",
    "existing", "existing customer", "current", "open",
}

_ID_CI = re.compile(r"(^|[_\s-])id($|[_\s-])|uuid|row_?number|^index$|^unnamed", re.I)
_ID_CAMEL = re.compile(r"[a-z](Id|ID)$")


def _looks_like_id_name(name: str) -> bool:
    return bool(_ID_CI.search(name) or _ID_CAMEL.search(name))


def _token(value) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip().lower()


# ---------------------------------------------------------------- target
def detect_target_candidates(df: pd.DataFrame) -> list[tuple[str, int]]:
    """Rank binary columns by how likely they are to be the churn label."""
    last = df.columns[-1]
    scored = []
    for col in df.columns:
        if df[col].dropna().nunique() != 2:
            continue
        score = 2
        if any(hint in col.lower() for hint in TARGET_HINTS):
            score += 3
        if col == last:
            score += 1
        if _looks_like_id_name(col):
            score -= 5
        scored.append((col, score))
    return sorted(scored, key=lambda item: -item[1])


def infer_positive_label(series: pd.Series):
    """Decide which of the two values means 'churned'."""
    values = list(series.dropna().unique())
    if len(values) != 2:
        raise ValueError(
            f"Target must have exactly 2 classes, found {len(values)}."
        )
    tokens = {v: _token(v) for v in values}
    pos = [v for v in values if tokens[v] in POSITIVE_TOKENS]
    neg = [v for v in values if tokens[v] in NEGATIVE_TOKENS]

    if len(pos) == 1:
        return pos[0], "token"
    if len(pos) == 0 and len(neg) == 1:
        return [v for v in values if v != neg[0]][0], "token"
    # Fallback: churners are usually the minority class.
    return series.value_counts().idxmin(), "minority"


def normalize_target(series: pd.Series, positive_label=None):
    """Return (y as 0/1 with NA kept, positive_label, method)."""
    values = list(series.dropna().unique())
    if positive_label is None:
        positive_label, method = infer_positive_label(series)
    else:
        match = [v for v in values if _token(v) == _token(positive_label)]
        if not match:
            raise ValueError(
                f"Positive label {positive_label!r} not found in target."
            )
        positive_label, method = match[0], "user"

    y = pd.Series(pd.NA, index=series.index, dtype="Int64")
    mask = series.notna()
    y[mask] = (series[mask] == positive_label).astype(int)
    return y, positive_label, method


# ------------------------------------------------------------ features
def coerce_numeric_like(df: pd.DataFrame):
    """Convert text columns that are really numbers (blanks become NaN)."""
    df = df.copy()
    converted = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].astype("string").str.strip().replace("", pd.NA)
        if s.notna().sum() == 0:
            continue
        numeric = pd.to_numeric(s, errors="coerce")
        if numeric[s.notna()].notna().all():
            df[col] = numeric
            converted.append(col)
    return df, converted


def detect_id_columns(df: pd.DataFrame, exclude=()) -> list[str]:
    """Columns that identify a row rather than describe a customer."""
    ids = []
    for col in df.columns:
        if col in exclude:
            continue
        s = df[col].dropna()
        if len(s) <= 20:
            continue
        unique_ratio = s.nunique() / len(s)
        if unique_ratio < 0.98:
            continue
        is_int = pd.api.types.is_integer_dtype(s)
        is_text = not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        if _looks_like_id_name(col) or is_int or is_text:
            ids.append(col)
    return ids


# -------------------------------------------------------------- schema
@dataclass
class Schema:
    target: str
    positive_label: object
    label_method: str
    target_confidence: str
    id_columns: list = field(default_factory=list)
    numeric_features: list = field(default_factory=list)
    categorical_features: list = field(default_factory=list)
    dropped: dict = field(default_factory=dict)
    converted_to_numeric: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def feature_columns(self) -> list:
        return self.numeric_features + self.categorical_features

    def to_dict(self) -> dict:
        d = asdict(self)
        d["positive_label"] = str(self.positive_label)
        d["feature_columns"] = self.feature_columns
        return d


def build_schema(
    df: pd.DataFrame,
    target: str | None = None,
    positive_label=None,
    id_columns: list | None = None,
    max_missing: float = 0.6,
):
    """Analyse any dataframe. Returns (schema, X, y) for labelled rows."""
    if df.empty:
        raise ValueError("The dataset is empty.")
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError("The dataset has duplicate column names.")

    warnings = []

    # 1. target
    if target is None:
        candidates = detect_target_candidates(df)
        if not candidates:
            raise ValueError("No binary column found to use as the target.")
        target, top_score = candidates[0]
        confidence = "high" if top_score >= 5 else "low"
        if confidence == "low":
            warnings.append(
                f"Target '{target}' was guessed with low confidence; "
                "please confirm it."
            )
    else:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found.")
        confidence = "user"

    y_all, positive_label, method = normalize_target(df[target], positive_label)
    labelled_mask = y_all.notna()
    if (~labelled_mask).any():
        warnings.append(
            f"{int((~labelled_mask).sum())} rows have no target and were excluded."
        )
    df = df.loc[labelled_mask]
    y = y_all.loc[labelled_mask].astype(int)

    # 2. fix numeric-looking text, then find IDs
    features_df = df.drop(columns=[target])
    features_df, converted = coerce_numeric_like(features_df)

    dropped = {}
    ids = id_columns if id_columns is not None else detect_id_columns(features_df)
    for col in ids:
        if col in features_df.columns:
            dropped[col] = "id"

    # 3. drop constant / mostly-missing columns
    for col in features_df.columns:
        if col in dropped:
            continue
        if features_df[col].nunique(dropna=True) <= 1:
            dropped[col] = "constant"
        elif features_df[col].isna().mean() > max_missing:
            dropped[col] = f"more than {max_missing:.0%} missing"

    X = features_df.drop(columns=list(dropped))
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain.")

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    # 4. warnings
    for col in X.select_dtypes(include=["datetime", "datetimetz"]).columns:
        warnings.append(f"Date column '{col}' needs date feature extraction.")
    for col in numeric:
        corr = X[col].corr(y)
        if pd.notna(corr) and abs(corr) > 0.9:
            warnings.append(
                f"Possible leakage: '{col}' has correlation {corr:.2f} with the target."
            )

    schema = Schema(
        target=target,
        positive_label=positive_label,
        label_method=method,
        target_confidence=confidence,
        id_columns=[c for c, r in dropped.items() if r == "id"],
        numeric_features=numeric,
        categorical_features=categorical,
        dropped=dropped,
        converted_to_numeric=converted,
        warnings=warnings,
    )
    return schema, X, y