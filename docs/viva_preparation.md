# ChurnSense AI Viva Preparation

## 60-second project explanation

ChurnSense AI is a machine-learning decision-support system built on the IBM Telco Customer Churn Dataset. It predicts each customer's likelihood of leaving, groups customers into four risk segments, and recommends an explainable retention action. I cleaned `TotalCharges`, one-hot encoded categorical fields, scaled numeric fields, and created business features such as tenure group and auto-pay. I compared Logistic Regression, Decision Tree, Random Forest, and XGBoost using a stratified held-out test set. I use ROC-AUC as the primary ranking metric and also report accuracy, precision, recall, and F1. The selected pipeline is presented through a Streamlit dashboard, while SHAP explains the most influential model features. The design includes a controlled retraining path and ethics checks for drift, fairness, and human review.

## Likely questions and concise answers

1. **What is customer churn?** A customer ending or discontinuing a service. In this dataset, `Churn=Yes` is the positive class.
2. **Why is this supervised learning?** Historical customers already have the labelled outcome Yes or No, so the algorithm learns input-to-label patterns.
3. **Why is it a classification problem rather than regression?** The target is categorical/binary, not a continuous number. The model predicts a probability then a class/risk band.
4. **Why remove `customerID`?** It is an identifier with no stable behavioural meaning. Including it can create meaningless splits or accidental memorisation.
5. **Why convert `TotalCharges`?** It arrives as text because blank strings occur. Models require numerical values for charge calculations.
6. **Why median imputation?** Median is resistant to unusually large charge values and keeps new customers rather than deleting them.
7. **What is one-hot encoding?** It converts a category into binary columns, such as `Contract_Month-to-month`, without imposing a false numeric order.
8. **Why standardise numeric values?** Logistic Regression is sensitive to feature scale during optimisation; standardisation centres and scales numeric values.
9. **What is data leakage?** Using information unavailable at prediction time or information from the test set during training. It makes evaluation unrealistically optimistic.
10. **Why use a pipeline?** It bundles preprocessing and the model so each uses the same fitted transformations and prevents leakage during fitting.
11. **Why stratify the split?** It keeps the churn/non-churn ratio similar in both train and test sets.
12. **Explain Logistic Regression.** It calculates a weighted sum of inputs and transforms log-odds into a probability using the sigmoid function.
13. **Why can a Decision Tree overfit?** It can keep splitting until it memorises noise. Depth and minimum leaf size constrain it.
14. **Why does a Random Forest help?** It averages many varied trees, reducing the instability/variance of a single decision tree.
15. **Explain XGBoost.** It builds trees sequentially, with each tree correcting residual errors from previous ones, while regularisation controls complexity.
16. **Why not choose by accuracy alone?** With class imbalance, a model can be accurate by predicting mostly non-churn. Precision and recall reveal errors on churners.
17. **What does recall mean here?** Of customers who truly churn, what proportion did the model flag? Higher recall means fewer missed at-risk customers.
18. **What does precision mean here?** Of customers flagged for retention, what proportion actually churned? Higher precision means fewer unnecessary offers.
19. **What is ROC-AUC?** The probability that the model ranks a random churner above a random non-churner; it compares thresholds rather than fixing one.
20. **Why use a 0.50 threshold?** It is a starting demonstration. Production threshold should depend on offer cost, customer lifetime value, team capacity, and desired recall.
21. **What is SHAP?** A method based on Shapley values that attributes a model prediction to input features relative to a baseline.
22. **Does SHAP prove causation?** No. It describes the trained model's association, not a real-world causal effect.
23. **What does mean absolute SHAP value show?** The average magnitude of each feature's contribution, giving global importance without cancelling positive and negative effects.
24. **How are recommendations generated?** Separate documented business rules use risk plus observable conditions, for example a contract offer for high-risk month-to-month customers.
25. **Why separate rules from the model?** It keeps actions auditable and editable by business stakeholders without retraining the classifier.
26. **What is model drift?** A change in inputs, customer behaviour, or the input-outcome relationship that can reduce model performance over time.
27. **How would you deploy it?** Containerise the Streamlit app or expose a prediction API, store model/data versions, log inputs and outcomes safely, monitor performance, and retrain with approval.
28. **What are the limitations?** Dataset is historical and telecom-specific; churn labels may be delayed; correlations are not causation; a static dataset cannot prove real-time performance.
29. **How would you improve it?** Cross-validation and tuning, probability calibration, cost-sensitive thresholding, individual SHAP views, data-drift alerts, secure database integration, and A/B tests of offers.
30. **What is the social/industrial impact?** It helps teams prioritise proactive, relevant support, potentially reducing avoidable attrition; it must retain human review and fairness safeguards.

## Demonstration sequence

1. Show the official CSV and explain target/identifier.
2. Run or show the cleaning and EDA reports.
3. Open `model_comparison.csv` and justify model selection.
4. Explain the confusion matrix and ROC curve.
5. Open SHAP global importance and state that it is associative.
6. Launch dashboard; show one customer, risk score, and transparent action.
7. End with deployment, monitoring, and responsible-AI safeguards.
