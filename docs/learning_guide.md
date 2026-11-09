# ChurnSense AI Learning Guide

This guide follows the project in the same order that you should explain it in Review-I, Review-II, and the final viva. The dataset is the IBM Telco Customer Churn Dataset: 7,043 customer records, 21 columns, binary target `Churn`.

## Phase 1 - Problem framing and data understanding

**Theory.** Churn prediction is a supervised binary classification problem. Each historical customer is labelled `Yes` or `No`; a model learns patterns associated with leaving. The output is best treated as a probability, because a retention team needs a ranked call list rather than only a yes/no label.

**Why it is required.** Clear framing decides the target, the correct metrics, and prevents target leakage. A customer identifier is not a behavioural signal, so it must never become a model input.

**ChurnSense AI application.** `Churn` is mapped from Yes/No to 1/0. `customerID` is retained for dashboard lookup and operational outreach but excluded from training.

**Expected output.** A data contract: 7,043 rows; original 21 columns; one binary target; and a feature list without `customerID` or `Churn`.

**Implementation.** `src/data.py` reads the official CSV and `src/features.py` identifies model columns.

### Libraries introduced

- **Pandas** loads CSV files into a DataFrame - a labelled table with rows and columns.
- **NumPy** supports fast numerical arrays and threshold calculations.
- **Pathlib** builds safe, operating-system-independent file paths.

## Phase 2 - Data preprocessing

**Theory.** Real datasets can contain blanks, wrong data types, duplicates, and incomparable scales. Preprocessing converts raw columns into consistent inputs. It must be fit on training data only; otherwise information from the test set leaks into training.

**Why it is required.** `TotalCharges` is stored as text because a few customers have blank values. Machine-learning estimators require numbers. Categorical columns such as `Contract` must become numerical indicators. Logistic Regression benefits when numeric variables have comparable scales.

**ChurnSense AI application.** We strip and convert `TotalCharges` to numeric, remove duplicate IDs, and replace its blank values with the median. The median is robust to large bill values. A `ColumnTransformer` standardises numeric features with `StandardScaler` and one-hot encodes categories. `handle_unknown='ignore'` lets the dashboard safely handle a category unseen at training time.

**Expected output.** A clean table with numeric `TotalCharges`, zero duplicate IDs, target encoded as 0/1, and a pipeline that transforms features consistently during training and prediction.

**Implementation.** See `clean_data()` and `make_preprocessor()`. The preprocessing object is stored inside each sklearn `Pipeline`, which guarantees that the same transformations are used for training and inference.

## Phase 3 - Exploratory data analysis (EDA)

**Theory.** EDA uses summaries and plots to understand distribution, data quality, and possible relationships. EDA generates hypotheses; it does not prove causality.

**Why it is required.** It detects anomalies before modeling and helps leaders understand the business context behind predicted risk.

**ChurnSense AI application.** `src/eda.py` produces class distribution, churn by contract type, monthly-charge distribution by outcome, and churn by tenure group. A contract chart answers whether commitment length is associated with churn; a tenure chart identifies early-life retention opportunities.

**Expected output.** `reports/data_summary.csv`, `reports/figures/eda_overview.png`, and `reports/figures/eda_behavior.png`.

**How to explain each visual.** A count plot reveals whether classes are imbalanced. A bar chart of churn *rate* is better than raw counts for comparing groups of different sizes. A box plot shows median, spread, and potential outliers in monthly charges. Every chart should be described as an association, e.g., "month-to-month customers show a higher observed churn rate," not "their contract causes churn."

## Phase 4 - Feature engineering

**Theory.** Feature engineering creates meaningful variables from existing information. Good features express a business concept simply, use only information available at prediction time, and remain interpretable.

**Why it is required.** Raw fields may hide useful structure. Derived variables can help both predictive performance and retention conversations.

**ChurnSense AI application.** We add:

- `AvgMonthlySpend = TotalCharges / max(tenure, 1)` to describe observed spending intensity.
- `TenureGroup` to turn continuous tenure into review-friendly lifecycle stages.
- `HasPremiumSupport` to distinguish customers with active tech support.
- `AutoPay` to flag automatic payment methods.

**Expected output.** An expanded feature table; no target leakage because all four features use data available for a customer at scoring time.

**Implementation.** `add_features()` is run before splitting. It contains no population-wide fitted statistic, while scaling and encoding remain inside the training pipeline.

## Phase 5 - Model development

**Theory.** A classifier learns a decision boundary between churned and retained customers. We compare both interpretable and non-linear models.

| Model | How it works | Why included |
|---|---|---|
| Logistic Regression | Models log-odds as a weighted combination of features, then converts them to probability with a sigmoid function. | Strong transparent baseline. |
| Decision Tree | Repeatedly splits data into branches that reduce class impurity. | Easy to explain as rules, but can overfit. |
| Random Forest | Averages many diverse decision trees trained on samples and random feature subsets. | Reduces the variance of one tree. |
| XGBoost | Adds shallow trees sequentially, each focusing on previous errors using gradient boosting. | Captures complex patterns and is often highly accurate on tabular data. |

**Why it is required.** One model can look good by chance or optimise the wrong trade-off. Comparison demonstrates disciplined model selection.

**ChurnSense AI application.** An 80/20 stratified split preserves the churn ratio in both sets. `random_state=42` makes the experiment reproducible. Logistic Regression, Decision Tree, and Random Forest use `class_weight='balanced'` where supported, giving the smaller churn class more influence. Tree depth and leaf-size limits reduce overfitting.

**Expected output.** `reports/model_comparison.csv`, saved pipelines in `models/`, and a selected model based primarily on held-out ROC-AUC.

**Implementation.** Run `python -m src.train`. The saved model includes preprocessing and estimator together; never manually preprocess dashboard records differently.

## Phase 6 - Evaluation and model comparison

**Theory.** A confusion matrix contains true positives (correctly flagged churn), false positives (unnecessary outreach), false negatives (missed churn), and true negatives. A probability threshold converts a score into a class.

**Why it is required.** Accuracy alone can be misleading if most customers stay. Retention has costs, so missing a valuable at-risk customer and contacting a safe customer are different mistakes.

| Metric | Formula | Churn interpretation |
|---|---|---|
| Accuracy | Correct predictions / all predictions | Broad score; insufficient alone under imbalance. |
| Precision | TP / (TP + FP) | Of those contacted, how many truly churned? Controls wasted offers. |
| Recall | TP / (TP + FN) | Of actual churners, how many were found? Controls missed opportunities. |
| F1 | Harmonic mean of precision and recall | One balance measure when both mistakes matter. |
| ROC-AUC | Area under TPR vs FPR across thresholds | Measures probability ranking independent of one threshold. |

**ChurnSense AI application.** We choose the best held-out ROC-AUC model, then report all metrics at threshold 0.50. In a real business, the threshold should be selected from offer cost, customer value, capacity, and desired recall - not copied blindly from this project.

**Expected output.** A comparison table, `model_comparison.png`, a confusion matrix, and ROC curve in `best_model_evaluation.png`.

**Implementation.** Run `python -m src.evaluate` after training.

## Phase 7 - SHAP explainability

**Theory.** SHAP (SHapley Additive exPlanations) attributes a prediction to features using ideas from cooperative game theory. A SHAP value describes how much a feature pushes a model output relative to a baseline. Positive and negative direction depend on the selected class and model output representation.

**Why it is required.** A high score without an explanation does not tell a retention manager what to investigate. Explanations improve auditability but do not prove cause-and-effect.

**ChurnSense AI application.** We generate a global bar plot using mean absolute SHAP values on a fixed sample. Large values mean a feature influenced predictions strongly on average. Feature names include one-hot category labels; this lets us see the influence of specific categories such as contract type.

**Expected output.** `reports/figures/shap_feature_importance.png` and `reports/shap_importance.csv`.

**Implementation.** Run `python -m src.explain`. For an individual case in a future upgrade, compute SHAP on the transformed single-customer record and show its top positive and negative contributors. Never phrase a SHAP result as a cause; say "the model associated this attribute with higher predicted churn."

## Phase 8 - Risk segmentation and recommendation engine

**Theory.** A probability is more actionable when converted to prioritisation bands. Business rules are transparent if they are deterministic and documented.

**Why it is required.** A call centre needs ordered work, not thousands of raw scores. Recommendation logic should make a traceable connection between observable customer conditions and an intervention.

**ChurnSense AI application.** Risk bands are Low (<0.25), Medium (0.25-0.449), High (0.45-0.699), and Critical (>=0.70). The engine prioritises a contract offer for high-risk month-to-month customers, a support trial for internet customers without support, automatic-payment incentives for electronic-check users, and early-life calls for newer customers.

**Expected output.** `reports/customer_risk_segments.csv` with a score, risk segment, and one next action per customer.

**Implementation.** See `src/recommend.py`. The rules are intentionally separate from the ML model, so a manager can audit or change them without retraining.

## Phase 9 - Streamlit dashboard and deployment

**Theory.** Deployment packages a trained pipeline as a repeatable user experience. Streamlit turns Python into a lightweight web application. Caching avoids repeatedly loading the model and data on every interface action.

**Why it is required.** Predictions only create value if users can find high-risk customers and understand what action to take.

**ChurnSense AI application.** `dashboard/app.py` contains an overview tab, an individual lookup, and a high-risk queue. The app displays actual portfolio churn separately from model risk so users do not confuse past labels with future predictions.

**Expected output.** A local browser dashboard after running `streamlit run dashboard/app.py`.

**Implementation.** Train once first, then start Streamlit. See `deployment_roadmap.md` for production hardening.

## Phase 10 - Maintenance, ethics, and limits

**Theory.** Model quality can decline when customer behaviour, products, or policy change; this is data drift and concept drift. Monitoring compares future distributions and outcomes against training baselines.

**Why it is required.** A one-time model can become stale. Churn predictions can also create unfair treatment if monitored poorly.

**ChurnSense AI application.** `src/retrain.py` is a controlled retraining entry point for a refreshed CSV with the same official schema. Before replacing a production model, compare new metrics with the current model, version the data and model, obtain approval, and roll back if degradation occurs. Audit performance across gender and senior-citizen groups; do not use sensitive characteristics to deny service or automate harmful decisions.

**Expected output.** A refreshed model, metrics, figures, and SHAP report only after validation.

**Implementation.** `python -m src.retrain --new-data path/to/refreshed.csv`. In production, schedule it only after data-quality checks and human sign-off.
