"""ChurnSense dashboard v2: a thin client of the FastAPI service."""
import io
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="ChurnSense AI", page_icon="📉", layout="wide")
st.title("ChurnSense AI")
st.caption("Upload any churn dataset, train, and score customers. Powered by the ChurnSense API.")


def api_get(path):
    r = requests.get(f"{API}{path}", timeout=60)
    r.raise_for_status()
    return r.json()


try:
    models = api_get("/models")
except Exception:
    st.error("Cannot reach the API. Start it with: uvicorn api.main:app --reload")
    st.stop()

if not models:
    st.warning("No models yet. Use the 'Train a new model' tab first.")

labels = {
    m["model_id"]: f'{m["dataset"]} | {m["best_model"]} | AUC {m["roc_auc"]} | {m["model_id"][-6:]}'
    for m in models
}
model_id = None
if models:
    model_id = st.sidebar.selectbox("Active model", list(labels), format_func=labels.get)
    st.sidebar.caption(f"Selected: {model_id}")

tab_model, tab_train, tab_one, tab_batch, tab_mon, tab_chat, tab_cost = st.tabs(
    ["Model details", "Train a new model", "Score one customer",
     "Portfolio & retention queue", "Monitoring", "Assistant", "Business value"]
)

# ------------------------------------------------------------ model details
with tab_model:
    if model_id:
        meta = api_get(f"/models/{model_id}")
        t = meta["test_at_tuned"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best model", meta["best_model"])
        c2.metric("Test ROC-AUC", t["roc_auc"])
        c3.metric("Recall", t["recall"])
        c4.metric("Precision", t["precision"])
        st.write(f'Decision threshold: **{meta["threshold"]:.2f}** | '
                 f'trained on {meta["n_train"]:,} rows, tested on {meta["n_test"]:,} unseen rows')
        if len(meta["statistically_tied"]) > 1:
            st.info("Statistically tied models: " + ", ".join(meta["statistically_tied"]))
        s = meta["schema"]
        st.write(f'Target: **{s["target"]}** (positive = {s["positive_label"]})')
        st.write("Dropped columns:", s["dropped"] or "none")
        st.write("Warnings:", s["warnings"] or "none")

        cm = pd.DataFrame(
            [[t["tn"], t["fp"]], [t["fn"], t["tp"]]],
            index=["Actual: stays", "Actual: churns"],
            columns=["Predicted: stays", "Predicted: churns"],
        )
        st.subheader("Confusion matrix (held-out test set)")
        st.dataframe(cm)

        st.subheader("Curves (held-out test set)")
        cv = api_get(f"/models/{model_id}/curves")
        c_left, c_right = st.columns(2)

        roc = go.Figure()
        roc.add_scatter(x=cv["roc"]["fpr"], y=cv["roc"]["tpr"], name=f'AUC {t["roc_auc"]}')
        roc.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="Random")
        roc.update_layout(title="ROC curve", xaxis_title="False positive rate",
                          yaxis_title="True positive rate", height=380)
        c_left.plotly_chart(roc, width="stretch")

        pr = go.Figure()
        pr.add_scatter(x=cv["pr"]["recall"], y=cv["pr"]["precision"], name=f'PR-AUC {t["pr_auc"]}')
        pr.update_layout(title="Precision-recall curve", xaxis_title="Recall",
                         yaxis_title="Precision", height=380)
        c_right.plotly_chart(pr, width="stretch")

        st.subheader("What drives churn? (SHAP importance)")
        try:
            imp = pd.DataFrame(api_get(f"/models/{model_id}/importance"))
            fig = px.bar(imp.sort_values("importance"), x="importance", y="feature",
                         orientation="h", title="Top features by mean |SHAP value|")
            fig.update_layout(height=500)
            st.plotly_chart(fig, width="stretch")
        except Exception:
            st.info("No SHAP data for this model. Retrain it, or pick a newer model, to enable explanations.")

# ------------------------------------------------------------ train
with tab_train:
    up = st.file_uploader("Upload a CSV with a churn column", type="csv", key="train_csv")
    if up and st.button("Analyse file"):
        r = requests.post(f"{API}/datasets", files={"file": (up.name, up.getvalue())}, timeout=120)
        if r.ok:
            st.session_state["profile"] = r.json()
        else:
            st.error(r.json().get("detail", r.text))
    p = st.session_state.get("profile")
    if p:
        st.success(f'{p["rows"]:,} rows, {p["columns"]} columns, churn rate {p["churn_rate"]:.1%}')
        st.write("Detected target:", p["schema"]["target"],
                 "| positive label:", p["schema"]["positive_label"])
        st.write("Dropped:", p["schema"]["dropped"] or "none")
        if p["schema"]["warnings"]:
            st.warning(p["schema"]["warnings"])
        if st.button("Train models"):
            job = requests.post(f'{API}/datasets/{p["dataset_id"]}/train', data={"beta": 1.0}).json()["job_id"]
            with st.spinner("Training and cross-validating four models..."):
                while True:
                    js_train = api_get(f"/jobs/{job}")
                    if js_train["status"] != "running":
                        break
                    time.sleep(3)
            if js_train["status"] == "done":
                st.success(f'Done. New model: {js_train["model_id"]}. Refresh the page to select it.')
            else:
                st.error(js_train.get("error", "Training failed"))

# ------------------------------------------------------------ one customer
with tab_one:
    if model_id:
        meta = api_get(f"/models/{model_id}")
        s = meta["schema"]
        st.write("Fill in the fields you know. Blank fields are treated as missing.")
        values = {}
        cols = st.columns(3)
        for i, name in enumerate(s["numeric_features"]):
            v = cols[i % 3].text_input(name, key=f"n_{model_id}_{name}")
            if v.strip():
                try:
                    values[name] = float(v)
                except ValueError:
                    st.error(f"{name} must be a number")
        for i, name in enumerate(s["categorical_features"]):
            v = cols[(i + len(s["numeric_features"])) % 3].text_input(name, key=f"c_{model_id}_{name}")
            if v.strip():
                values[name] = v.strip()
        if st.button("Predict churn risk"):
            r = requests.post(f"{API}/models/{model_id}/predict", json={"features": values}, timeout=60)
            if r.ok:
                out = r.json()
                a, b = st.columns(2)
                a.metric("Churn probability", f'{out["churn_probability"]:.1%}')
                b.metric("Risk segment", out["risk_segment"])
                st.write("Will churn:", out["will_churn"])
                st.caption(f"Model used: {model_id}")
                if out["missing_fields"]:
                    st.caption(f'{len(out["missing_fields"])} fields were left blank.')

                # ---- why this score (SHAP)
                try:
                    ex = requests.post(f"{API}/models/{model_id}/explain",
                                       json={"features": values}, timeout=60)
                    if ex.ok:
                        st.subheader("Why this score?")
                        for f in ex.json()["top_factors"]:
                            icon = "🔺" if f["effect"] > 0 else "🔻"
                            st.write(f'{icon} **{f["feature"]}**: {f["direction"]} ({f["effect"]:+.2f})')
                    else:
                        st.warning(f"Explain failed ({ex.status_code}): {ex.text[:300]}")
                except Exception as exc:
                    st.warning(f"Explain call error: {exc}")

                # ---- retention plan (LLM with rule fallback)
                try:
                    with st.spinner("Writing retention plan (first run can take a minute)..."):
                        rec = requests.post(f"{API}/models/{model_id}/recommend",
                                            json={"features": values}, timeout=180)
                    if rec.ok:
                        plan = rec.json()
                        st.subheader("Retention plan")
                        st.caption(f'Source: {plan["source"]} (the model predicts; the LLM only explains)')
                        st.write(plan["summary"])
                        for act in plan["actions"]:
                            st.write("• " + act)
                        st.info(plan["message"])
                    else:
                        st.warning(f"Recommend failed ({rec.status_code}): {rec.text[:300]}")
                except Exception as exc:
                    st.warning(f"Recommend call error: {exc}")
            else:
                st.error(r.text)

        # ---- what-if simulator
        st.divider()
        st.subheader("What-if: change one field")
        cat_fields = s["categorical_features"]
        if cat_fields:
            wf = st.selectbox("Field to change", cat_fields, key=f"wf_{model_id}")
            options = st.text_input(
                "Values to try (comma separated, exactly as in your data)",
                key=f"wo_{model_id}",
                placeholder="for example: Month-to-month, One year, Two year")
            if st.button("Run what-if"):
                cands = [o.strip() for o in options.split(",") if o.strip()]
                if not values:
                    st.warning("Fill in the customer fields above first.")
                elif not cands:
                    st.warning("Type at least one value to try.")
                else:
                    w = requests.post(f"{API}/models/{model_id}/whatif",
                                      json={"features": values, "field": wf, "candidates": cands},
                                      timeout=60)
                    if w.ok:
                        st.dataframe(pd.DataFrame(w.json()["options"]), hide_index=True)
                        st.caption("Rows are ranked from lowest churn risk. The model shows an "
                                   "association, not proof of cause: a real change may not behave "
                                   "the same way.")
                    else:
                        st.error(w.text)

# ------------------------------------------------------------ portfolio + queue
with tab_batch:
    if model_id:
        f = st.file_uploader("Upload customers to score", type="csv", key="batch_csv")
        if f and st.button("Score file"):
            r = requests.post(f"{API}/models/{model_id}/predict-batch",
                              files={"file": (f.name, f.getvalue())}, timeout=600)
            if r.ok:
                st.session_state["scored"] = pd.read_csv(io.StringIO(r.text))
                st.session_state["scored_model"] = model_id
            else:
                st.error(r.text)

        scored = st.session_state.get("scored")
        if scored is not None and st.session_state.get("scored_model") == model_id:
            st.caption(
                "Note: if this file contains customers the model was trained on, these "
                "predictions are optimistic. Honest performance numbers are on the "
                "Model details tab (held-out test set)."
            )
            order = ["Low", "Medium", "High", "Critical"]
            a, b, c = st.columns(3)
            a.metric("Customers", f"{len(scored):,}")
            b.metric("High/Critical risk",
                     f"{scored['risk_segment'].isin(['High', 'Critical']).sum():,}")
            c.metric("Mean churn probability", f"{scored['churn_probability'].mean():.1%}")

            left, right = st.columns(2)
            seg = scored["risk_segment"].value_counts().reindex(order).fillna(0).reset_index()
            seg.columns = ["risk_segment", "count"]
            left.plotly_chart(
                px.bar(seg, x="risk_segment", y="count", color="risk_segment",
                       category_orders={"risk_segment": order},
                       title="Customer risk segmentation"),
                width="stretch",
            )

            cats = [
                col for col in scored.columns
                if col != "risk_segment"
                and not pd.api.types.is_numeric_dtype(scored[col])
                and 1 < scored[col].nunique() <= 20
            ]
            if cats:
                pick = right.selectbox("Average churn probability by", cats)
                g = scored.groupby(pick, as_index=False)["churn_probability"].mean()
                fig = px.bar(g, x=pick, y="churn_probability", text_auto=".1%",
                             title=f"Predicted churn by {pick}")
                fig.update_yaxes(tickformat=".0%")
                right.plotly_chart(fig, width="stretch")

            st.subheader("Retention queue")
            queue = scored[scored["risk_segment"].isin(["Critical", "High"])]
            st.write(f"{len(queue):,} customers at High or Critical risk, highest first.")
            st.dataframe(queue, width="stretch", hide_index=True)
            st.download_button("Download retention queue CSV", queue.to_csv(index=False),
                               "retention_queue.csv", "text/csv")

# ------------------------------------------------------------ monitoring
with tab_mon:
    if model_id:
        st.write("Upload fresh customer data with the same columns as the training data. "
                 "Include the churn column if you want to allow retraining.")
        mf = st.file_uploader("New data CSV", type="csv", key="mon_csv")

        if mf and st.button("Check drift"):
            r = requests.post(f"{API}/models/{model_id}/drift",
                              files={"file": (mf.name, mf.getvalue())}, timeout=300)
            if r.ok:
                st.session_state["drift"] = r.json()
                st.session_state["drift_model"] = model_id
            else:
                st.error(r.text)

        d = st.session_state.get("drift")
        if d and st.session_state.get("drift_model") == model_id:
            a, b, c = st.columns(3)
            a.metric("Drift status", d["status"])
            b.metric("Max PSI", d["max_psi"])
            c.metric("Features above 0.25", d["n_significant"])
            st.caption("PSI below 0.10 is stable, 0.10 to 0.25 moderate, above 0.25 significant.")
            top = pd.DataFrame(d["features"][:15])
            st.plotly_chart(
                px.bar(top.sort_values("psi"), x="psi", y="feature", orientation="h",
                       title="Drift by feature (PSI)"),
                width="stretch",
            )

        force = st.checkbox("Retrain even if no drift", value=False)
        if mf and st.button("Run monitor (retrain if needed)"):
            resp = requests.post(f"{API}/models/{model_id}/monitor",
                                 files={"file": (mf.name, mf.getvalue())},
                                 data={"force": str(force).lower()}, timeout=300)
            job = resp.json()["job_id"]
            with st.spinner("Checking drift and training a challenger..."):
                while True:
                    js = api_get(f"/jobs/{job}")
                    if js["status"] != "running":
                        break
                    time.sleep(3)
            if js["status"] == "done":
                st.success(js["action"])
                if "champion_auc" in js:
                    x, y_, z = st.columns(3)
                    x.metric("Current model AUC", js["champion_auc"])
                    y_.metric("Challenger AUC", js["challenger_auc"])
                    z.metric("Promoted", "yes" if js["promoted"] else "no")
                    if js["promoted"]:
                        st.info(f'New model: {js["new_model_id"]}. Refresh the page to select it.')
            else:
                st.error(js.get("error", "Monitor failed"))

# ------------------------------------------------------------ assistant
with tab_chat:
    if model_id:
        st.write("Load your customer file once, then ask questions about it. "
                 "Numbers come from our code; the LLM only words them.")
        cf = st.file_uploader("Customer CSV", type="csv", key="chat_csv")
        if cf and st.button("Load customers"):
            r = requests.post(f"{API}/models/{model_id}/portfolio",
                              files={"file": (cf.name, cf.getvalue())}, timeout=600)
            if r.ok:
                st.session_state["chat_ready"] = model_id
                st.session_state["chat_history"] = []
                st.success(f'{r.json()["rows"]:,} customers loaded.')
            else:
                st.error(r.text)

        if st.session_state.get("chat_ready") == model_id:
            for turn in st.session_state.get("chat_history", []):
                with st.chat_message(turn["role"]):
                    st.write(turn["text"])
                    if turn.get("table"):
                        st.dataframe(pd.DataFrame(turn["table"]), hide_index=True)
                    if turn.get("source"):
                        st.caption(f'Wording: {turn["source"]}')
                    if turn.get("note"):
                        st.caption(turn["note"])

            q = st.chat_input("Ask about your customers...")
            if q:
                hist = st.session_state["chat_history"]
                hist.append({"role": "user", "text": q})
                with st.spinner("Thinking..."):
                    resp = requests.post(f"{API}/models/{model_id}/chat",
                                         json={"message": q}, timeout=120)
                if resp.ok:
                    out = resp.json()
                    hist.append({"role": "assistant", "text": out["answer"],
                                 "table": out["table"], "source": out["source"],
                                 "note": out.get("note")})
                else:
                    hist.append({"role": "assistant", "text": f"Error: {resp.text[:200]}"})
                st.rerun()

# ------------------------------------------------------------ business value
with tab_cost:
    if model_id:
        st.write("Enter your own business numbers. The best cut-off is the one that "
                 "maximises net savings on the held-out test customers.")
        c1, c2, c3 = st.columns(3)
        value = c1.number_input("Value of keeping one customer", 0.0, 1_000_000.0, 500.0, 50.0)
        offer = c2.number_input("Cost of one retention offer", 0.0, 1_000_000.0, 50.0, 10.0)
        succ = c3.slider("Offer success rate", 0.05, 1.0, 0.30, 0.05)
        r = requests.get(f"{API}/models/{model_id}/cost",
                         params={"value": value, "offer_cost": offer, "success_rate": succ},
                         timeout=60)
        if r.ok:
            res = r.json()
            b, dflt = res["best"], res["at_default_threshold"]
            x, y_, z = st.columns(3)
            x.metric("Best cut-off", b["threshold"])
            y_.metric("Net savings (test set)", f'{b["net_savings"]:,.0f}')
            z.metric("Savings at the model's own cut-off", f'{dflt["net_savings"]:,.0f}')
            st.caption(f'At the best cut-off you send {b["offers_sent"]} offers and reach '
                       f'{b["churners_reached"]} real churners, out of {res["test_rows"]} '
                       "test customers. Savings assume the success rate above.")
            cdf = pd.DataFrame(res["curve"])
            fig = px.line(cdf, x="threshold", y="net_savings", title="Net savings by cut-off")
            fig.add_vline(x=b["threshold"], line_dash="dash")
            st.plotly_chart(fig, width="stretch")
        else:
            st.error(r.text)