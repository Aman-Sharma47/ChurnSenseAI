"""Train, compare, and save ChurnSense AI models."""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from src.config import MODELS_DIR, REPORTS_DIR, RANDOM_STATE, TEST_SIZE, TARGET
from src.data import load_raw_data, clean_data
from src.features import add_features, make_pipeline
from src.recommend import add_risk_actions


def candidate_models():
    """Four complementary classifiers required by the project synopsis."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=12, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=350, max_depth=10, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=2),
    }


def metrics(y_true, probability, threshold=0.50) -> dict:
    prediction = (np.asarray(probability) >= threshold).astype(int)
    return {
        "accuracy": round(accuracy_score(y_true, prediction), 4),
        "precision": round(precision_score(y_true, prediction, zero_division=0), 4),
        "recall": round(recall_score(y_true, prediction, zero_division=0), 4),
        "f1": round(f1_score(y_true, prediction, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, probability), 4),
    }


def train_all() -> pd.DataFrame:
    warnings.filterwarnings("ignore", category=FutureWarning)
    clean = add_features(clean_data(load_raw_data()))
    X = clean.drop(columns=[TARGET, "customerID"])
    y = clean[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    score_rows, trained = [], {}
    for name, estimator in candidate_models().items():
        pipeline = make_pipeline(estimator, pd.concat([X_train, y_train], axis=1))
        pipeline.fit(X_train, y_train)
        probability = pipeline.predict_proba(X_test)[:, 1]
        row = {"model": name, **metrics(y_test, probability)}
        score_rows.append(row)
        trained[name] = pipeline
    results = pd.DataFrame(score_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    best_name = results.loc[0, "model"]
    best_model = trained[best_name]
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    joblib.dump(trained, MODELS_DIR / "all_models.joblib")
    results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "random_state": RANDOM_STATE, "test_size": TEST_SIZE, "features": X.columns.tolist()}, f, indent=2)
    # Score all customers only after model selection; this is an operational prioritisation list, not test evaluation.
    full_prob = best_model.predict_proba(X)[:, 1]
    scored = add_risk_actions(clean, full_prob)
    scored.to_csv(REPORTS_DIR / "customer_risk_segments.csv", index=False)
    return results


if __name__ == "__main__":
    print(train_all().to_string(index=False))
