
"""Assign customer risk levels and suggest retention actions."""

import pandas as pd


def risk_band(probability: float) -> str:
    """Convert a churn probability into a risk category."""

    if probability >= 0.70:
        return "Critical"
    elif probability >= 0.45:
        return "High"
    elif probability >= 0.25:
        return "Medium"
    else:
        return "Low"


def retention_recommendation(
    customer: pd.Series,
    probability: float,
) -> str:
    """Choose a retention action using simple business rules."""

    if probability < 0.25:
        return "Maintain experience; monitor at the next review cycle."

    contract = customer.get("Contract")
    tech_support = customer.get("TechSupport")
    internet_service = customer.get("InternetService")
    payment_method = customer.get("PaymentMethod")
    tenure = customer.get("tenure", 0)

    if contract == "Month-to-month":
        return (
            "Offer a discounted 12-month contract "
            "with a clear savings estimate."
        )

    elif tech_support == "No" and internet_service != "No":
        return (
            "Offer a proactive technical-support trial "
            "and onboarding check-in."
        )

    elif payment_method == "Electronic check":
        return (
            "Incentivize automatic payment to reduce billing friction."
        )

    elif tenure <= 12:
        return (
            "Assign an early-life success call "
            "and tailor an onboarding plan."
        )

    else:
        return (
            "Contact the customer with a personalised "
            "service review and value offer."
        )


def add_risk_actions(
    df: pd.DataFrame,
    probabilities,
) -> pd.DataFrame:
    """Add predicted risk levels and actions to customer records."""

    result = df.copy()
    result["churn_probability"] = probabilities

    result["risk_segment"] = result["churn_probability"].apply(
        risk_band
    )

    result["retention_action"] = [
        retention_recommendation(customer, probability)
        for (_, customer), probability in zip(
            result.iterrows(),
            probabilities,
        )
    ]

    return result
