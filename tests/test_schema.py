from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.schema import build_schema

ROOT = Path(__file__).resolve().parents[1]
TELCO = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


@pytest.mark.skipif(not TELCO.exists(), reason="Telco CSV not found")
def test_telco():
    df = pd.read_csv(TELCO)
    schema, X, y = build_schema(df)
    assert schema.target == "Churn"
    assert schema.positive_label == "Yes"
    assert schema.id_columns == ["customerID"]
    assert "TotalCharges" in schema.numeric_features
    assert len(X) == len(y) == 7043
    assert 0.25 < y.mean() < 0.28


def test_bank_style_numeric_target_and_two_ids():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "RowNumber": range(1, n + 1),
        "CustomerId": rng.permutation(np.arange(15_000_000, 15_000_000 + n)),
        "Surname": rng.choice(list("ABCDEFGHIJ"), n),
        "CreditScore": rng.integers(350, 850, n),
        "Geography": rng.choice(["France", "Spain", "Germany"], n),
        "IsActiveMember": rng.integers(0, 2, n),
        "Exited": (rng.random(n) < 0.2).astype(int),
    })
    schema, X, y = build_schema(df)
    assert schema.target == "Exited"
    assert schema.positive_label == 1
    assert set(schema.id_columns) == {"RowNumber", "CustomerId"}
    assert "Surname" in schema.categorical_features
    assert set(y.unique()) == {0, 1}


def test_credit_card_style_string_target():
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({
        "CLIENTNUM": rng.permutation(np.arange(700_000_000, 700_000_000 + n)),
        "Attrition_Flag": rng.choice(
            ["Existing Customer", "Attrited Customer"], n, p=[0.84, 0.16]
        ),
        "Customer_Age": rng.integers(25, 70, n),
        "Card_Category": rng.choice(["Blue", "Silver", "Gold"], n),
        "Notes": ["same"] * n,
    })
    schema, X, y = build_schema(df)
    assert schema.target == "Attrition_Flag"
    assert schema.positive_label == "Attrited Customer"
    assert "CLIENTNUM" in schema.id_columns
    assert schema.dropped["Notes"] == "constant"
    assert 0.1 < y.mean() < 0.22