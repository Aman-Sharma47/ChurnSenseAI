import pandas as pd
from src.data import clean_data
from src.recommend import risk_band, retention_recommendation

def test_clean_data_converts_total_charges_and_target():
    raw = pd.DataFrame({"customerID":["A"], "TotalCharges":[" "], "Churn":["Yes"]})
    clean = clean_data(raw)
    assert clean["Churn"].iloc[0] == 1
    assert pd.api.types.is_numeric_dtype(clean["TotalCharges"])

def test_risk_bands_are_ordered():
    assert [risk_band(x) for x in [.1, .3, .5, .8]] == ["Low", "Medium", "High", "Critical"]

def test_month_to_month_action():
    customer = pd.Series({"Contract":"Month-to-month", "TechSupport":"No", "InternetService":"DSL"})
    assert "contract" in retention_recommendation(customer, .8).lower()
