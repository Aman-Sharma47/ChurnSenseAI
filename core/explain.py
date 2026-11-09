"""Per-customer and global SHAP explanations for any saved model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def _inner_pipeline(model):
    """Return the fitted Pipeline inside a (possibly calibrated) model."""
    if hasattr(model, "calibrated_classifiers_"):
        return model.calibrated_classifiers_[0].estimator
    return model


class Explainer:
    def __init__(self, model, background: pd.DataFrame, max_background: int = 200):
        self.pipe = _inner_pipeline(model)
        self.cleaner = self.pipe.named_steps["cleaner"]
        self.prep = self.pipe.named_steps["preprocessor"]
        self.est = self.pipe.named_steps["model"]
        bg = background.sample(min(max_background, len(background)), random_state=42)
        self.bg = self._transform(bg)
        self.names = [n.split("__", 1)[-1] for n in self.prep.get_feature_names_out()]
        tree_models = ("XGBClassifier", "RandomForestClassifier", "DecisionTreeClassifier")
        if self.est.__class__.__name__ in tree_models:
            self.explainer = shap.TreeExplainer(self.est)
        else:
            self.explainer = shap.LinearExplainer(self.est, self.bg)

    def _transform(self, df):
        return np.asarray(self.prep.transform(self.cleaner.transform(df)), dtype=float)

    def _values(self, X):
        v = self.explainer.shap_values(X)
        if isinstance(v, list):
            v = v[1]
        v = np.asarray(v)
        if v.ndim == 3:
            v = v[:, :, 1]
        return v

    def customer(self, row: pd.DataFrame, k: int = 5) -> list[dict]:
        X = self._transform(row)
        vals = self._values(X)[0]
        order = np.argsort(-np.abs(vals))[:k]
        return [
            {"feature": self.names[i], "effect": round(float(vals[i]), 4),
             "direction": "raises risk" if vals[i] > 0 else "lowers risk"}
            for i in order
        ]

    def global_importance(self, sample: pd.DataFrame, k: int = 15) -> list[dict]:
        X = self._transform(sample.sample(min(500, len(sample)), random_state=42))
        mean_abs = np.abs(self._values(X)).mean(axis=0)
        order = np.argsort(-mean_abs)[:k]
        return [{"feature": self.names[i], "importance": round(float(mean_abs[i]), 4)}
                for i in order]