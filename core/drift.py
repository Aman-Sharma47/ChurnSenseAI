"""Population Stability Index (PSI) drift detection against a model's training sample."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core import registry


def _psi(ref_p, new_p) -> float:
    eps = 1e-4
    r = np.clip(ref_p, eps, None)
    n = np.clip(new_p, eps, None)
    return float(np.sum((n - r) * np.log(n / r)))


def psi_categorical(ref: pd.Series, new: pd.Series):
    ref = ref.map(lambda v: "__missing__" if pd.isna(v) else str(v))
    new = new.map(lambda v: "__missing__" if pd.isna(v) else str(v))
    if len(ref) < 20 or len(new) < 20:
        return None
    cats = sorted(set(ref) | set(new))
    r = ref.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    n = new.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    return _psi(r, n)


def psi_numeric(ref: pd.Series, new: pd.Series, bins: int = 10):
    r_num = pd.to_numeric(ref, errors="coerce").dropna()
    n_num = pd.to_numeric(new, errors="coerce").dropna()
    if len(r_num) < 20 or len(n_num) < 20:
        return None
    edges = np.unique(np.quantile(r_num, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                      # binary or near-constant: treat as categories
        return psi_categorical(ref, new)
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(r_num, edges)[0] / len(r_num)
    n = np.histogram(n_num, edges)[0] / len(n_num)
    return _psi(r, n)


def status_from(max_psi: float) -> str:
    if max_psi > 0.25:
        return "significant"
    if max_psi > 0.10:
        return "moderate"
    return "stable"


def drift_report(model_id: str, new_df: pd.DataFrame) -> dict:
    """PSI of every model feature: new data vs the model's saved training sample."""
    _, meta = registry.load(model_id)
    ref = pd.read_csv(registry.REGISTRY_DIR / model_id / "background.csv")
    s = meta["schema"]
    numeric = set(s["numeric_features"])
    new_df = new_df.copy()
    new_df.columns = [str(c).strip() for c in new_df.columns]

    rows, missing = [], []
    for col in s["numeric_features"] + s["categorical_features"]:
        if col not in new_df.columns or col not in ref.columns:
            missing.append(col)
            continue
        value = (psi_numeric(ref[col], new_df[col]) if col in numeric
                 else psi_categorical(ref[col], new_df[col]))
        if value is not None:
            rows.append({"feature": col, "psi": round(float(value), 4)})

    rows.sort(key=lambda r: -r["psi"])
    max_psi = rows[0]["psi"] if rows else 0.0
    return {
        "status": status_from(max_psi),
        "max_psi": max_psi,
        "n_significant": int(sum(r["psi"] > 0.25 for r in rows)),
        "n_features": len(rows),
        "missing_columns": missing,
        "features": rows,
        "new_rows": int(len(new_df)),
    }