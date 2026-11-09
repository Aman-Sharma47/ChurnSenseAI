# ChurnSense AI

ChurnSense AI is an academic and industry-style customer churn prediction and retention analytics system built on the official **IBM Telco Customer Churn Dataset**. It predicts the probability that a customer will leave, groups customers by risk, explains global drivers with SHAP, and recommends a next retention action.

## What is included

- Reproducible cleaning, feature engineering, EDA, training, evaluation, and SHAP scripts.
- Logistic Regression, Decision Tree, Random Forest, and XGBoost comparison.
- Saved best model, customer risk queue, visual reports, and a Streamlit dashboard.
- Learning guide, viva preparation, deployment roadmap, presentation, and Azure DevOps/Plane work-item suggestions.

## Folder guide

| Folder | Purpose |
|---|---|
| `data/` | Official IBM Telco CSV used by every pipeline stage. |
| `notebooks/` | Guided interactive learning notebook. |
| `src/` | Reusable Python modules for data, features, models, evaluation, SHAP, recommendations, and retraining. |
| `models/` | Generated serialized models and metadata after training. |
| `reports/` | Generated metrics, risk segments, and figures for review evidence. |
| `dashboard/` | Streamlit application for business users. |
| `docs/` | Theory-first learning guide, viva questions, deployment roadmap, and backlog. |
| `presentation/` | Final project-review slide deck. |
| `tests/` | Lightweight checks of cleaning and retention rules. |

## Quick start

1. Open a terminal in this folder.
2. Create a virtual environment and install packages: `python -m venv .venv` then `.venv\\Scripts\\python -m pip install -r requirements.txt` on Windows.
3. Train the models: `.venv\\Scripts\\python -m src.train`.
4. Generate EDA, evaluation, and SHAP figures:

```powershell
.venv\Scripts\python -m src.eda
.venv\Scripts\python -m src.evaluate
.venv\Scripts\python -m src.explain
```

5. Launch the dashboard: `.venv\\Scripts\\streamlit run dashboard/app.py`.
6. Run tests: `.venv\\Scripts\\python -m pytest`.

## Methodology and safe interpretation

The model is trained with a stratified 80/20 split so the churn proportion is preserved in train and test sets. ROC-AUC is the primary ranking metric because a retention team needs reliable probability ranking, while precision, recall, F1, and accuracy provide complementary views. The dashboard’s risk bands are operational priorities, not guaranteed outcomes. Retention offers require human review, fairness monitoring, and consent-aware outreach.

See [docs/learning_guide.md](docs/learning_guide.md) for the phase-by-phase explanation and implementation rationale.
