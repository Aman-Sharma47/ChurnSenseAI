"""What-if analysis: re-score a customer with one field changed."""
from __future__ import annotations

import pandas as pd

from core import registry


def _prob(pipe, features: dict) -> float:
    return float(pipe.predict_proba(pd.DataFrame([features]))[0, 1])


def what_if(model_id: str, features: dict, field: str, new_value) -> dict:
    pipe, meta = registry.load(model_id)
    schema = meta["schema"]
    if field not in schema["feature_columns"]:
        raise ValueError(f"'{field}' is not a feature of this model.")
    if field in schema["numeric_features"]:
        new_value = float(new_value)
    before = _prob(pipe, features)
    changed = dict(features)
    changed[field] = new_value
    after = _prob(pipe, changed)
    return {"field": field, "new_value": new_value,
            "before": round(before, 4), "after": round(after, 4),
            "change": round(after - before, 4)}


def best_options(model_id: str, features: dict, field: str, candidates: list) -> list[dict]:
    """Try each candidate value for one field, ranked from lowest churn risk."""
    rows = [what_if(model_id, features, field, v) for v in candidates]
    return sorted(rows, key=lambda r: r["after"])