
"""EDA figures focused on actionable churn patterns."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, REPORTS_DIR
from src.data import clean_data, load_raw_data
from src.features import add_features


def create_eda_figures():
    """Generate churn analysis figures and a dataset summary."""

    sns.set_theme(style="whitegrid", palette="deep")

    # Prepare the dataset.
    raw_data = load_raw_data()
    df = add_features(clean_data(raw_data))

    # Save basic dataset statistics.
    blank_charges = (
        raw_data["TotalCharges"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    summary = {
        "records": len(df),
        "features": 21,
        "churn_rate": round(df["Churn"].mean(), 4),
        "missing_before_imputation": int(blank_charges),
    }

    pd.DataFrame([summary]).to_csv(
        REPORTS_DIR / "data_summary.csv",
        index=False,
    )

    # Figure 1: Overall churn distribution and contract type.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.countplot(
        data=df,
        x="Churn",
        hue="Churn",
        palette={0: "#2563eb", 1: "#ef4444"},
        legend=False,
        ax=axes[0],
    )

    axes[0].set_xticks([0, 1], labels=["Stayed", "Churned"])
    axes[0].set_title("Class distribution")

    contract_rate = (
        df.groupby("Contract", observed=True)["Churn"]
        .mean()
        .sort_values(ascending=False)
        .mul(100)
    )

    contract_rate.plot(
        kind="bar",
        color="#7c3aed",
        ax=axes[1],
    )

    axes[1].set_ylabel("Churn rate (%)")
    axes[1].set_title("Churn rate by contract type")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_overview.png", dpi=180)
    plt.close(fig)

    # Figure 2: Monthly charges and customer tenure.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.boxplot(
        data=df,
        x="Churn",
        y="MonthlyCharges",
        hue="Churn",
        palette={0: "#2563eb", 1: "#ef4444"},
        legend=False,
        ax=axes[0],
    )

    axes[0].set_xticks([0, 1], labels=["Stayed", "Churned"])
    axes[0].set_title("Monthly charges by outcome")

    tenure_order = [
        "0-12 months",
        "13-24 months",
        "25-48 months",
        "49+ months",
    ]

    tenure_rate = (
        df.groupby("TenureGroup", observed=True)["Churn"]
        .mean()
        .mul(100)
        .reindex(tenure_order)
    )

    tenure_rate.plot(
        kind="bar",
        color="#0f766e",
        ax=axes[1],
    )

    axes[1].set_ylabel("Churn rate (%)")
    axes[1].set_title("Churn rate by tenure group")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_behavior.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    create_eda_figures()
