# Plane Work Item Suggestions

Use these as Plane issues (or Azure DevOps work items). Suggested hierarchy: Epic -> Story -> Task.

| ID | Type | Title | Acceptance criteria | Priority |
|---|---|---|---|---|
| CS-1 | Epic | Dataset and preprocessing | Official IBM CSV is versioned; cleaning is reproducible. | High |
| CS-2 | Story | Validate official dataset schema | 7,043-record/21-column expectation is documented; `Churn` and ID checks run. | High |
| CS-3 | Task | Clean TotalCharges and duplicates | Blank values handled and no duplicate `customerID` remains. | High |
| CS-4 | Epic | Analytics and EDA | Business-relevant visuals and narrative are available. | High |
| CS-5 | Story | Produce churn-driver EDA | Contract, tenure, charge, and class-distribution visuals render. | High |
| CS-6 | Epic | Modeling and evaluation | Four required models are trained and compared. | High |
| CS-7 | Story | Build leakage-safe feature pipeline | Encoder/scaler are inside sklearn pipeline; ID excluded. | High |
| CS-8 | Story | Train model candidates | Logistic Regression, Decision Tree, Random Forest, XGBoost report held-out metrics. | High |
| CS-9 | Story | Select and persist model | Selection rationale and model metadata are saved. | High |
| CS-10 | Epic | Explainability and actioning | SHAP and retention policy convert scores into decisions. | High |
| CS-11 | Story | Generate SHAP feature explanation | Global importance chart and CSV are reproducible. | Medium |
| CS-12 | Story | Create risk segmentation | Every customer has a probability and Low/Medium/High/Critical band. | High |
| CS-13 | Story | Define retention rules | Documented, testable, human-reviewable actions exist. | High |
| CS-14 | Epic | Dashboard and delivery | Stakeholders can explore results and project evidence is complete. | High |
| CS-15 | Story | Build Streamlit dashboard | Overview, customer lookup, and high-risk queue function locally. | High |
| CS-16 | Task | Add automated tests | Core cleaning and rule tests pass. | Medium |
| CS-17 | Story | Prepare documentation and viva pack | Learning guide, roadmap, viva answers, and slides are present. | High |
| CS-18 | Epic | MLOps readiness | Retraining and monitoring design is documented. | Medium |
| CS-19 | Story | Add controlled retraining | Refreshed-schema path rebuilds artifacts; promotion requires approval. | Medium |
| CS-20 | Story | Define drift and fairness checks | Metrics, owners, thresholds, and escalation path are documented. | Medium |

## Suggested board states

`Backlog -> Ready -> In Progress -> Review -> Done`. Add labels: `data`, `ml`, `dashboard`, `documentation`, `xai`, `mlops`, and `high-priority`. For Review-II, demonstrate CS-2 through CS-15; reserve CS-18 through CS-20 as final-production readiness discussion if time is limited.
