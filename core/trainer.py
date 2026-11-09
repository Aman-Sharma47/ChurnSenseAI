"""Generic training engine: CV comparison, threshold tuning, one-shot test evaluation."""
from __future__ import annotations
from sklearn.calibration import CalibratedClassifierCV
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, fbeta_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from core.schema import build_schema

SEED = 42


class FrameCleaner(BaseEstimator, TransformerMixin):
    """Make any raw DataFrame safe for the encoders (handles NA types, missing columns)."""

    def __init__(self, numeric=None, categorical=None):
        self.numeric = numeric
        self.categorical = categorical

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = pd.DataFrame(index=X.index)
        for col in self.numeric:
            out[col] = (
                pd.to_numeric(X[col], errors="coerce").astype(float)
                if col in X.columns else np.nan
            )
        for col in self.categorical:
            if col in X.columns:
                out[col] = X[col].map(lambda v: np.nan if pd.isna(v) else str(v)).astype(object)
            else:
                out[col] = np.nan
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.numeric) + list(self.categorical), dtype=object)


def build_model_pipeline(schema, estimator) -> Pipeline:
    parts = []
    if schema.numeric_features:
        parts.append((
            "numeric",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            schema.numeric_features,
        ))
    if schema.categorical_features:
        parts.append((
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.01,
                    max_categories=20,
                    sparse_output=False,
                )),
            ]),
            schema.categorical_features,
        ))
    return Pipeline([
        ("cleaner", FrameCleaner(schema.numeric_features, schema.categorical_features)),
        ("preprocessor", ColumnTransformer(parts, remainder="drop")),
        ("model", clone(estimator)),
    ])


def candidate_models(pos_weight: float, fast: bool = False) -> dict:
    n_rf, n_xgb = (100, 100) if fast else (300, 300)
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=12, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=n_rf, max_depth=10, min_samples_leaf=3,
            class_weight="balanced", n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=n_xgb, max_depth=4, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, scale_pos_weight=pos_weight,
            eval_metric="logloss", n_jobs=2, random_state=SEED),
    }


def score(y_true, prob, threshold=0.5) -> dict:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, prob), 4),
        "pr_auc": round(average_precision_score(y_true, prob), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def tune_threshold(y_true, prob, beta: float = 1.0) -> float:
    """Pick the cut-off that maximises F-beta (beta>1 favours recall)."""
    grid = np.linspace(0.05, 0.95, 91)
    values = [
        fbeta_score(y_true, (prob >= t).astype(int), beta=beta, zero_division=0)
        for t in grid
    ]
    return float(grid[int(np.argmax(values))])


def train(
    df: pd.DataFrame,
    target: str | None = None,
    positive_label=None,
    cv: int = 5,
    test_size: float = 0.2,
    beta: float = 1.0,
    fast: bool = False,
) -> dict:
    schema, X, y = build_schema(df, target=target, positive_label=positive_label)
    if y.value_counts().min() < cv * 2:
        raise ValueError("Too few rows in the smaller class for cross-validation.")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED)

    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    models = candidate_models(pos_weight, fast=fast)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)

    rows, oof_store = [], {}
    for name, est in models.items():
        oof = np.zeros(len(X_tr))
        aucs = []
        for tr_idx, va_idx in skf.split(X_tr, y_tr):
            pipe = build_model_pipeline(schema, est)
            pipe.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
            p = pipe.predict_proba(X_tr.iloc[va_idx])[:, 1]
            oof[va_idx] = p
            aucs.append(roc_auc_score(y_tr.iloc[va_idx], p))
        oof_store[name] = oof
        rows.append({
            "model": name,
            "cv_roc_auc": round(float(np.mean(aucs)), 4),
            "cv_auc_std": round(float(np.std(aucs)), 4),
            "cv_pr_auc": round(float(average_precision_score(y_tr, oof)), 4),
        })

    comparison = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)
    best = comparison.loc[0, "model"]
    cutoff = comparison.loc[0, "cv_roc_auc"] - comparison.loc[0, "cv_auc_std"]
    tied = comparison.loc[comparison["cv_roc_auc"] >= cutoff, "model"].tolist()

        # Calibrated final model: probabilities now match real churn frequency.
    base = build_model_pipeline(schema, models[best])
    final = CalibratedClassifierCV(base, method="isotonic", cv=3)
    final.fit(X_tr, y_tr)

    # Re-tune the threshold on calibrated out-of-fold scores (never on the test set).
    oof_cal = np.zeros(len(X_tr))
    for tr_idx, va_idx in skf.split(X_tr, y_tr):
        m = CalibratedClassifierCV(
            build_model_pipeline(schema, models[best]), method="isotonic", cv=3)
        m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
        oof_cal[va_idx] = m.predict_proba(X_tr.iloc[va_idx])[:, 1]
    threshold = tune_threshold(y_tr, oof_cal, beta=beta)

    p_test = final.predict_proba(X_te)[:, 1]

    return {
        "schema": schema,
        "pipeline": final,
        "best_model": best,
        "statistically_tied": tied,
        "comparison": comparison,
        "threshold": threshold,
        "test_at_0.5": score(y_te, p_test, 0.5),
        "test_at_tuned": score(y_te, p_test, threshold),
        "test_predictions": pd.DataFrame({"y_true": y_te.values, "probability": p_test}),
        "background": X_tr.sample(min(500, len(X_tr)), random_state=SEED),
        "n_train": len(X_tr), "n_test": len(X_te),
    }


def save_run(result: dict, folder) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["pipeline"], folder / "pipeline.joblib")
    result["comparison"].to_csv(folder / "cv_comparison.csv", index=False)
    result["test_predictions"].to_csv(folder / "test_predictions.csv", index=False)
    meta = {
        "best_model": result["best_model"],
        "statistically_tied": result["statistically_tied"],
        "threshold": result["threshold"],
        "test_at_0.5": result["test_at_0.5"],
        "test_at_tuned": result["test_at_tuned"],
        "n_train": result["n_train"], "n_test": result["n_test"],
        "schema": result["schema"].to_dict(),
    }
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return folder


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train churn models on any CSV.")
    ap.add_argument("csv")
    ap.add_argument("--target")
    ap.add_argument("--positive")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    res = train(pd.read_csv(a.csv), target=a.target, positive_label=a.positive,
                beta=a.beta, fast=a.fast)
    out = a.out or f"models/runs/{Path(a.csv).stem}"
    save_run(res, out)

    print("\nTarget:", res["schema"].target, "| positive:", res["schema"].positive_label)
    print("Warnings:", res["schema"].warnings or "none")
    print("\nCross-validation (train set only):")
    print(res["comparison"].to_string(index=False))
    print("\nBest:", res["best_model"], "| statistically tied:", res["statistically_tied"])
    print("Tuned threshold:", round(res["threshold"], 3))
    print("\nTest set at 0.5   :", res["test_at_0.5"])
    print("Test set at tuned :", res["test_at_tuned"])
    print("\nSaved to:", out)