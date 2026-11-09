"""Train on any CSV and register the result: python -m core.run <csv>"""
import argparse
from pathlib import Path

import pandas as pd

from core.registry import register
from core.trainer import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--target")
    ap.add_argument("--positive")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()

    name = Path(a.csv).stem
    res = train(pd.read_csv(a.csv), target=a.target, positive_label=a.positive,
                beta=a.beta, fast=a.fast)
    model_id = register(res, name)

    t = res["test_at_tuned"]
    print(f"Dataset      : {name}")
    print(f"Target       : {res['schema'].target} (positive = {res['schema'].positive_label})")
    print(f"Dropped      : {res['schema'].dropped}")
    print(f"Warnings     : {res['schema'].warnings or 'none'}")
    print(f"Best model   : {res['best_model']} (tied: {res['statistically_tied']})")
    print(f"Threshold    : {res['threshold']:.2f}")
    print(f"Test ROC-AUC : {t['roc_auc']} | PR-AUC: {t['pr_auc']}")
    print(f"Recall       : {t['recall']} | Precision: {t['precision']}")
    print(f"Registered as: {model_id}")