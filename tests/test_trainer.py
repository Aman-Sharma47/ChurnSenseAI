import numpy as np
import pandas as pd

from core.trainer import train


def make_df(n=800, seed=0):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    monthly = rng.normal(70, 20, n).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
    logit = -0.04 * tenure + 0.02 * monthly + np.where(contract == "Month-to-month", 1.0, -1.0) - 0.5
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame({
        "account": [f"C{i:05d}" for i in range(n)],
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "Contract": contract,
        "Churn": np.where(churn, "Yes", "No"),
    })
    df.loc[rng.choice(n, 20, replace=False), "MonthlyCharges"] = np.nan
    return df


def test_train_end_to_end():
    res = train(make_df(), cv=3, fast=True)
    assert res["schema"].target == "Churn"
    assert res["schema"].id_columns == ["account"]
    assert len(res["comparison"]) == 4
    assert res["test_at_tuned"]["roc_auc"] > 0.70
    assert res["n_train"] + res["n_test"] == 800
    # the saved pipeline accepts a raw dataframe, even with a missing column
    raw = make_df().drop(columns=["Churn", "Contract"]).head(5)
    probs = res["pipeline"].predict_proba(raw)[:, 1]
    assert len(probs) == 5 and ((probs >= 0) & (probs <= 1)).all()