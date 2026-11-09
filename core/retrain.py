"""Monitor drift and retrain with champion/challenger promotion."""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from core import registry
from core.drift import drift_report
from core.schema import build_schema
from core.trainer import train

LOG = registry.REGISTRY_DIR.parent / "retrain_log.jsonl"


def retrain_if_better(model_id: str, new_df: pd.DataFrame, min_gain: float = 0.005) -> dict:
    """Train a challenger on new labelled data; register it only if it beats the champion."""
    champion, meta = registry.load(model_id)
    new_df = new_df.reset_index(drop=True)
    sch = meta["schema"]
    _, X, y = build_schema(new_df, target=sch["target"], positive_label=sch["positive_label"])

    X_fit, X_hold, y_fit, y_hold = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=7)

    result = train(new_df.loc[X_fit.index], target=sch["target"],
                   positive_label=sch["positive_label"], fast=True)
    challenger = result["pipeline"]

    champ_auc = float(roc_auc_score(y_hold, champion.predict_proba(X_hold)[:, 1]))
    chall_auc = float(roc_auc_score(y_hold, challenger.predict_proba(X_hold)[:, 1]))
    promoted = chall_auc >= champ_auc + min_gain

    new_id = registry.register(result, meta["dataset"]) if promoted else None
    return {
        "champion_auc": round(champ_auc, 4),
        "challenger_auc": round(chall_auc, 4),
        "holdout_rows": int(len(y_hold)),
        "promoted": promoted,
        "new_model_id": new_id,
    }


def monitor(model_id: str, new_df: pd.DataFrame, force: bool = False,
            min_gain: float = 0.005) -> dict:
    """Policy: no drift means do nothing; drift means challenger vs champion."""
    drift = drift_report(model_id, new_df)
    report = {
        "model_id": model_id,
        "checked": datetime.now().isoformat(timespec="seconds"),
        "drift_status": drift["status"],
        "max_psi": drift["max_psi"],
        "n_significant": drift["n_significant"],
        "top_drifting": drift["features"][:5],
    }
    if drift["status"] == "stable" and not force:
        report["action"] = "skipped: no significant drift"
        report["promoted"] = False
    else:
        report.update(retrain_if_better(model_id, new_df, min_gain))
        report["action"] = ("retrained and promoted the challenger" if report["promoted"]
                            else "retrained, kept the current model (challenger not better)")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(report, default=str) + "\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Check drift and retrain if needed.")
    ap.add_argument("model_id")
    ap.add_argument("csv")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print(json.dumps(monitor(a.model_id, pd.read_csv(a.csv), force=a.force), indent=2, default=str))