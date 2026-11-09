
"""Generate global SHAP explanations for the selected churn model."""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE, TARGET
from src.data import clean_data, load_raw_data
from src.features import add_features


def create_shap_summary(sample_size=400):
    """Create a global feature-importance plot using SHAP values."""

    # Load the selected model and prepare customer data.
    model_pipe = joblib.load(MODELS_DIR / "best_model.joblib")

    df = add_features(clean_data(load_raw_data()))
    X = df.drop(columns=[TARGET, "customerID"])

    sample = X.sample(
        n=min(sample_size, len(X)),
        random_state=RANDOM_STATE,
    )

    # Apply the same preprocessing used during model training.
    preprocessor = model_pipe.named_steps["preprocessor"]
    model = model_pipe.named_steps["model"]

    transformed = preprocessor.transform(sample)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    # Choose an explainer that supports the selected model.
    if model.__class__.__name__ == "XGBClassifier":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(transformed)
    else:
        explainer = shap.Explainer(
            model,
            transformed,
            feature_names=feature_names,
        )
        shap_values = explainer(transformed)

    # Save the global feature-importance chart.
    plt.figure(figsize=(11, 7))
    shap.plots.bar(
        shap_values,
        max_display=15,
        show=False,
    )
    plt.title("SHAP Global Feature Importance - Selected Model")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "shap_feature_importance.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    # Calculate mean absolute SHAP values for each feature.
    mean_abs = np.abs(shap_values.values).mean(axis=0)

    # Handle explanations that contain multiple model outputs.
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.mean(axis=-1)

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    })

    importance = importance.sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    importance.head(25).to_csv(
        FIGURES_DIR.parent / "shap_importance.csv",
        index=False,
    )

    print("SHAP explanation completed successfully.")
    print("Chart saved to reports/figures/shap_feature_importance.png")
    print("Feature importance saved to reports/shap_importance.csv")


if __name__ == "__main__":
    create_shap_summary()
