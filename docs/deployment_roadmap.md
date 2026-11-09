# Deployment Roadmap

## Local academic deployment

1. Create `.venv`, install `requirements.txt`, train the model, and generate figures.
2. Run `streamlit run dashboard/app.py`.
3. Capture screenshots and include the model comparison, SHAP plot, and risk queue in project review evidence.

## Production architecture

```text
CRM / Billing / Support data
        -> validated feature pipeline -> versioned model -> prediction API/dashboard
        -> risk queue + approved offers -> retention actions -> observed outcomes
                                           -> monitoring -> approved retraining
```

## Required controls before production

- **Data:** schema validation, freshness checks, access control, encryption, retention policy, and consent-aware outreach.
- **Model:** version dataset/code/model, log prediction time and model version, test drift and performance monthly, and calibrate probabilities.
- **Business:** define cost/benefit threshold, approve recommendation rules, record offer acceptance, and A/B test interventions.
- **Responsible AI:** measure errors across appropriate groups, prohibit automated adverse decisions, provide human escalation, and explain model limits.
- **Reliability:** use CI tests, container image scans, model rollback, monitoring alerts, and backup/restore procedures.

## Retraining policy

Retrain only when a refreshed dataset retains the approved schema. Evaluate the candidate on a time-aware holdout where possible; compare ROC-AUC, recall, calibration, and subgroup metrics to the deployed baseline. Promote only after a documented reviewer approves it. `src/retrain.py` automates artifact rebuilding, not approval.

## Cloud options

- **Simple:** Streamlit Community Cloud or Render for a classroom demo; use only non-sensitive data.
- **Azure:** Azure App Service/Container Apps for dashboard/API, Blob Storage for versioned artifacts, Azure ML for registry and monitoring, Key Vault for secrets.
- **AWS/GCP:** equivalent managed container, object storage, model registry, scheduler, and logging services.

Never commit credentials or personally identifiable production data to the repository.
