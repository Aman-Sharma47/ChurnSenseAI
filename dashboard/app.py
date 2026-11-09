
"""Streamlit decision-support dashboard for ChurnSense AI."""

from pathlib import Path
import sys

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RANDOM_STATE, TEST_SIZE, TARGET
from src.data import clean_data, load_raw_data
from src.features import add_features
from src.recommend import risk_band, retention_recommendation


# ---------------------------------------------------
# Page configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="ChurnSense AI",
    page_icon="📉",
    layout="wide",
)


# ---------------------------------------------------
# Load model and data
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "best_model.joblib")


@st.cache_data
def load_data():
    return add_features(clean_data(load_raw_data()))


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model not found. Run "
        "`python -m src.train` from the project root first."
    )
    st.stop()

df = load_data()

features = df.drop(columns=[TARGET, "customerID"])
probabilities = model.predict_proba(features)[:, 1]

df["churn_probability"] = probabilities
df["risk_segment"] = df["churn_probability"].apply(risk_band)


# ---------------------------------------------------
# Model comparison
# ---------------------------------------------------

comparison_path = ROOT / "reports" / "model_comparison.csv"

if comparison_path.exists():
    model_comparison = pd.read_csv(comparison_path)
else:
    model_comparison = None


# ---------------------------------------------------
# Header and logo
# ---------------------------------------------------

logo_path = ROOT / "dashboard" / "assets" / "logo.png"

if logo_path.exists():
    st.image(str(logo_path), width=120)

st.title("ChurnSense AI")

st.caption(
    "Customer churn prediction and retention analytics | "
    "IBM Telco Customer Churn Dataset"
)


# ---------------------------------------------------
# Tabs
# ---------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Portfolio overview",
        "Customer prediction",
        "Model Analytics",
        "Retention queue",
    ]
)


# ===================================================
# TAB 1: PORTFOLIO OVERVIEW
# ===================================================

with tab1:
    a, b, c, d = st.columns(4)

    a.metric("Customers", f"{len(df):,}")

    b.metric(
        "Actual churn rate",
        f"{df[TARGET].mean():.1%}",
    )

    c.metric(
        "High/Critical risk",
        f"{df['risk_segment'].isin(['High', 'Critical']).sum():,}",
    )

    d.metric(
        "Mean predicted risk",
        f"{df['churn_probability'].mean():.1%}",
    )

    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            df,
            x="risk_segment",
            category_orders={
                "risk_segment": [
                    "Low",
                    "Medium",
                    "High",
                    "Critical",
                ]
            },
            color="risk_segment",
            title="Customer Risk Segmentation",
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        rate = (
            df.groupby("Contract", as_index=False)["Churn"]
            .mean()
        )

        fig = px.bar(
            rate,
            x="Contract",
            y="Churn",
            text_auto=".1%",
            title="Actual Churn Rate by Contract",
        )

        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    if model_comparison is not None and not model_comparison.empty:
        st.subheader("Model Comparison")

        st.dataframe(
            model_comparison,
            use_container_width=True,
            hide_index=True,
        )

        best_model_name = model_comparison.iloc[0]["model"]

        st.success(
            f"Best Performing Model by ROC-AUC: {best_model_name}"
        )


# ===================================================
# TAB 2: CUSTOMER PREDICTION
# ===================================================

with tab2:
    customer_id = st.selectbox(
        "Choose a customer ID",
        df["customerID"].tolist(),
    )

    customer = df.loc[df["customerID"] == customer_id].iloc[0]
    probability = float(customer["churn_probability"])

    x, y, z = st.columns(3)

    x.metric("Churn Probability", f"{probability:.1%}")
    y.metric("Risk Segment", customer["risk_segment"])
    z.metric("Contract", customer["Contract"])

    st.subheader("Recommended Next Action")

    st.info(
        retention_recommendation(customer, probability)
    )

    customer_details = customer[
        [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "InternetService",
            "TechSupport",
            "PaymentMethod",
            "Contract",
        ]
    ].to_frame("Value")

    st.dataframe(
        customer_details,
        use_container_width=True,
    )


# ===================================================
# TAB 3: MODEL ANALYTICS
# ===================================================

with tab3:
    st.subheader("Model Analytics")

    # Recreate the same stratified split used during training.
    X = df.drop(
        columns=[
            TARGET,
            "customerID",
            "churn_probability",
            "risk_segment",
        ]
    )
    y = df[TARGET].astype(int)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    y_true = y_test
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    # Confusion Matrix — render only once.
    st.markdown("### Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 3.5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        annot_kws={"size": 12},
        ax=ax,
    )

    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title("Confusion Matrix", fontsize=13)
    ax.tick_params(axis="both", labelsize=9)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    # ROC Curve — based on the held-out test set.
    st.markdown("### ROC Curve")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    roc_fig = go.Figure()

    roc_fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"AUC = {auc_score:.3f}",
        )
    )

    roc_fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash"),
            name="Random baseline",
        )
    )

    roc_fig.update_layout(
        title="ROC Curve — Held-Out Test Set",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450,
    )

    st.plotly_chart(roc_fig, use_container_width=True)

    # Feature importance for models that support it.
    st.markdown("### Feature Importance")

    try:
        estimator = model.named_steps["model"]
        feature_names = (
            model.named_steps["preprocessor"]
            .get_feature_names_out()
        )

        if not hasattr(estimator, "feature_importances_"):
            st.info(
                "The selected model does not expose tree-based "
                "feature importance."
            )
        else:
            importance = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Importance": estimator.feature_importances_,
                }
            )

            importance = importance.nlargest(15, "Importance")

            fig = px.bar(
                importance.sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 15 Most Important Features",
            )

            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

    except (AttributeError, KeyError, ValueError) as exc:
        st.info(f"Feature importance could not be displayed: {exc}")


# ===================================================
# TAB 4: RETENTION QUEUE
# ===================================================

with tab4:
    queue = (
        df.loc[
            df["risk_segment"].isin(["Critical", "High"]),
            [
                "customerID",
                "churn_probability",
                "risk_segment",
                "Contract",
                "tenure",
                "MonthlyCharges",
                "TechSupport",
                "PaymentMethod",
            ],
        ]
        .sort_values(
            "churn_probability",
            ascending=False,
        )
    )

    st.write(
        "Prioritise this list for retention outreach. "
        "Recommendations are rules-based and should be reviewed by a human."
    )

    st.download_button(
        label="📥 Download Retention Queue CSV",
        data=queue.to_csv(index=False),
        file_name="retention_queue.csv",
        mime="text/csv",
    )

    st.dataframe(
        queue,
        use_container_width=True,
        hide_index=True,
    )
