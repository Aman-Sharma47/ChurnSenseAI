
"""Generate evaluation visuals for ChurnSense AI models."""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split

from src.config import (
    FIGURES_DIR,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from src.data import clean_data, load_raw_data
from src.features import add_features


def create_evaluation_figures():
    """Create model comparison, confusion matrix, and ROC plots."""

    sns.set_theme(style="whitegrid")

    # Load the saved model comparison results.
    results = pd.read_csv(
        FIGURES_DIR.parent / "model_comparison.csv"
    )

    # Prepare the scores for a grouped comparison chart.
    metrics_to_compare = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    melted = results.melt(
        id_vars="model",
        value_vars=metrics_to_compare,
        var_name="metric",
        value_name="score",
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    sns.barplot(
        data=melted,
        x="model",
        y="score",
        hue="metric",
        ax=ax,
    )

    ax.set_ylim(0, 1)
    ax.set_title("Model comparison on the held-out test set")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=10)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "model_comparison.png",
        dpi=180,
    )
    plt.close(fig)

    # Recreate the same stratified test split used during training.
    clean = add_features(
        clean_data(load_raw_data())
    )

    X = clean.drop(columns=[TARGET, "customerID"])
    y = clean[TARGET]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Load the selected model saved during training.
    model_path = (
        FIGURES_DIR.parent.parent
        / "models"
        / "best_model.joblib"
    )

    model = joblib.load(model_path)
    probability = model.predict_proba(X_test)[:, 1]

    # Display the confusion matrix and ROC curve together.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        probability >= 0.5,
        display_labels=["Stayed", "Churned"],
        cmap="Blues",
        ax=axes[0],
    )

    axes[0].set_title("Confusion matrix at 0.50 threshold")

    RocCurveDisplay.from_predictions(
        y_test,
        probability,
        ax=axes[1],
    )

    axes[1].set_title("ROC curve for selected model")

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "best_model_evaluation.png",
        dpi=180,
    )
    plt.close(fig)


if __name__ == "__main__":
    create_evaluation_figures()
