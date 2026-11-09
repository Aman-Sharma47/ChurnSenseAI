"""Cost-based decision threshold, computed from held-out test predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core import registry


def cost_curve(y_true, prob, value: float, offer_cost: float, success_rate: float = 1.0):
    """Net savings at every cut-off.
    caught churners (tp) x success_rate x value, minus every offer sent x offer_cost."""
    y = np.asarray(y_true)
    p = np.asarray(prob)
    rows = []
    for t in np.linspace(0.02, 0.98, 97):
        sent = p >= t
        tp = int(np.sum(sent & (y == 1)))
        n_sent = int(sent.sum())
        saved = tp * success_rate * value
        spent = n_sent * offer_cost
        rows.append({"threshold": round(float(t), 2), "offers_sent": n_sent,
                     "churners_reached": tp, "net_savings": round(saved - spent, 2)})
    return rows


def optimise(model_id: str, value: float, offer_cost: float, success_rate: float = 1.0) -> dict:
    d = pd.read_csv(registry.REGISTRY_DIR / model_id / "test_predictions.csv")
    rows = cost_curve(d["y_true"], d["probability"], value, offer_cost, success_rate)
    best = max(rows, key=lambda r: r["net_savings"])
    _, meta = registry.load(model_id)
    default_t = meta["threshold"]
    at_default = min(rows, key=lambda r: abs(r["threshold"] - default_t))
    return {
        "best": best,
        "at_default_threshold": at_default,
        "default_threshold": default_t,
        "test_rows": int(len(d)),
        "curve": rows,
    }