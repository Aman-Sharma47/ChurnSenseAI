import pandas as pd

from core.chat import answer


def make():
    df = pd.DataFrame({
        "cid": ["A1", "B2", "C3", "D4"],
        "Contract": ["Month-to-month", "Two year", "Month-to-month", "One year"],
        "tenure": [1, 60, 3, 24],
        "churn_probability": [0.9, 0.05, 0.7, 0.2],
        "risk_segment": ["Critical", "Low", "Critical", "Medium"],
    })
    schema = {"id_columns": ["cid"], "numeric_features": ["tenure"],
              "categorical_features": ["Contract"], "feature_columns": ["tenure", "Contract"]}
    return df, schema


def test_top_risky():
    df, s = make()
    r = answer("who are the top 2 riskiest customers?", df, s, None, use_llm=False)
    assert r["intent"] == "top_risky"
    assert len(r["table"]) == 2 and r["table"][0]["cid"] == "A1"


def test_by_column():
    df, s = make()
    r = answer("average risk by contract", df, s, None, use_llm=False)
    assert r["intent"] == "by_column"
    assert r["table"][0]["Contract"] == "Month-to-month"


def test_explain_customer():
    df, s = make()
    r = answer("why is B2 likely to leave?", df, s, None, use_llm=False)
    assert r["intent"] == "explain_customer" and "B2" in r["answer"]


def test_summary_and_help():
    df, s = make()
    assert answer("give me a portfolio summary", df, s, None, use_llm=False)["intent"] == "summary"
    assert answer("tell me a joke", df, s, None, use_llm=False)["intent"] == "help"

from core.chat import _grounded


class FakeExplainer:
    def customer(self, row, k=5):
        return [{"feature": "tenure", "effect": 1.2, "direction": "raises risk"}]


def test_no_misleading_100_percent():
    df, s = make()
    df["churn_probability"] = [0.9999, 0.05, 0.9998, 0.2]
    r = answer("top 2 riskiest customers", df, s, None, use_llm=False)
    assert "100%" not in r["answer"] and "over 99%" in r["answer"]


def test_explain_states_the_real_value():
    df, s = make()
    r = answer("why is C3 likely to leave?", df, s, FakeExplainer(), use_llm=False)
    assert "tenure is 3" in r["answer"]
    assert "high tenure" not in r["answer"]


def test_guard_rejects_wrong_qualifier_and_summary_has_note():
    facts = "Customer C3 has a 70% churn probability (Critical risk). tenure is 3 (raises risk)"
    assert not _grounded("Likely to leave due to a high tenure.", facts, "why")
    assert _grounded("Tenure is 3, which raises the risk.", facts, "why")
    df, s = make()
    assert answer("portfolio summary", df, s, None, use_llm=False)["note"]

def test_general_question_goes_to_qa(monkeypatch):
    import core.qa as qa
    monkeypatch.setattr(qa, "general_answer", lambda q, c: ("Churn means a customer leaves.", "test"))
    df, s = make()
    r = answer("what is the probability of churn?", df, s, None, use_llm=True)
    assert r["intent"] == "general" and "leaves" in r["answer"]