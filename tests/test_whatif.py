from core import whatif


class FakePipe:
    def predict_proba(self, df):
        import numpy as np
        p = 0.8 if df.loc[0, "Contract"] == "Month-to-month" else 0.2
        return np.array([[1 - p, p]])


def test_changing_contract_lowers_risk(monkeypatch):
    meta = {"schema": {"feature_columns": ["Contract"], "numeric_features": []}}
    monkeypatch.setattr(whatif.registry, "load", lambda _id: (FakePipe(), meta))
    r = whatif.what_if("m", {"Contract": "Month-to-month"}, "Contract", "Two year")
    assert r["before"] == 0.8 and r["after"] == 0.2 and r["change"] < 0
    opts = whatif.best_options("m", {"Contract": "Month-to-month"}, "Contract",
                               ["Month-to-month", "Two year"])
    assert opts[0]["new_value"] == "Two year"