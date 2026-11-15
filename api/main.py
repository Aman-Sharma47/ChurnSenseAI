"""ChurnSense API: upload, train, list, predict, explain. Run: uvicorn api.main:app --reload"""
from __future__ import annotations
from core.llm import recommend as llm_recommend
from core.chat import answer as chat_answer
from core.drift import drift_report
from core.retrain import monitor
from core.cost import optimise
from core.whatif import what_if, best_options
import io
import threading
import uuid

from pathlib import Path
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sklearn.metrics import precision_recall_curve, roc_curve

from core import registry
from core.explain import Explainer
from core.schema import build_schema
from core.trainer import train

app = FastAPI(title="ChurnSense API", version="0.3.0")

DATASETS: dict[str, dict] = {}   # id -> {"df": DataFrame, "name": str}
JOBS: dict[str, dict] = {}       # id -> status dict
_EXPLAINERS: dict = {}           # model_id -> Explainer (cached)
_lock = threading.Lock()


def _read_csv(upload: UploadFile) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(upload.file.read()))
    except Exception as exc:
        raise HTTPException(400, f"Could not read CSV: {exc}")


def _clean(value):
    """Treat blank or Swagger placeholder values as 'not provided'."""
    if value is None:
        return None
    value = value.strip()
    return None if value in ("", "string", "null", "None") else value


def risk_band(p: float, threshold: float) -> str:
    """Bands for calibrated probabilities (a score of 0.3 means about 30% churn)."""
    if p >= 0.60:
        return "Critical"
    if p >= 0.30:
        return "High"
    if p >= 0.15:
        return "Medium"
    return "Low"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/datasets")
def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV; returns a profile and the auto-detected schema."""
    df = _read_csv(file)
    try:
        schema, X, y = build_schema(df)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    ds_id = uuid.uuid4().hex[:8]
    DATASETS[ds_id] = {"df": df, "name": (file.filename or "dataset").rsplit(".", 1)[0]}
    return {
        "dataset_id": ds_id,
        "rows": len(df),
        "columns": len(df.columns),
        "churn_rate": round(float(y.mean()), 4),
        "schema": schema.to_dict(),
    }


def _run_training(job_id: str, ds_id: str, target, positive, beta):
    try:
        ds = DATASETS[ds_id]
        result = train(ds["df"], target=target, positive_label=positive, beta=beta, fast=True)
        model_id = registry.register(result, ds["name"])
        with _lock:
            JOBS[job_id] = {"status": "done", "model_id": model_id}
    except Exception as exc:
        with _lock:
            JOBS[job_id] = {"status": "failed", "error": str(exc)}


@app.post("/datasets/{ds_id}/train")
def start_training(
    ds_id: str,
    tasks: BackgroundTasks,
    target: str | None = Form(None),
    positive_label: str | None = Form(None),
    beta: float = Form(1.0),
):
    if ds_id not in DATASETS:
        raise HTTPException(404, "Unknown dataset_id")
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"status": "running"}
    tasks.add_task(
        _run_training, job_id, ds_id, _clean(target), _clean(positive_label), beta
    )
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Unknown job_id")
    return JOBS[job_id]


@app.get("/models")
def list_models():
    return registry.list_models()


@app.get("/models/{model_id}")
def model_info(model_id: str):
    try:
        _, meta = registry.load(model_id)
    except FileNotFoundError:
        raise HTTPException(404, "Model not found")
    return meta


@app.get("/models/{model_id}/curves")
def curves(model_id: str):
    """ROC and precision-recall points computed from the held-out test predictions."""
    path = registry.REGISTRY_DIR / model_id / "test_predictions.csv"
    if not path.exists():
        raise HTTPException(404, "Model not found")
    d = pd.read_csv(path)
    fpr, tpr, _ = roc_curve(d["y_true"], d["probability"])
    prec, rec, _ = precision_recall_curve(d["y_true"], d["probability"])
    step = max(1, len(fpr) // 300)
    pstep = max(1, len(prec) // 300)
    return {
        "roc": {"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist()},
        "pr": {"recall": rec[::pstep].tolist(), "precision": prec[::pstep].tolist()},
    }


class PredictRequest(BaseModel):
    features: dict


@app.post("/models/{model_id}/predict")
def predict(model_id: str, req: PredictRequest):
    try:
        pipe, meta = registry.load(model_id)
    except FileNotFoundError:
        raise HTTPException(404, "Model not found")
    expected = meta["schema"]["feature_columns"]
    unknown = [k for k in req.features if k not in expected]
    missing = [c for c in expected if c not in req.features]
    p = float(pipe.predict_proba(pd.DataFrame([req.features]))[0, 1])
    t = meta["threshold"]
    return {
        "churn_probability": round(p, 4),
        "will_churn": p >= t,
        "risk_segment": risk_band(p, t),
        "threshold": t,
        "ignored_fields": unknown,
        "missing_fields": missing,
    }


@app.post("/models/{model_id}/predict-batch")
def predict_batch(model_id: str, file: UploadFile = File(...)):
    try:
        pipe, meta = registry.load(model_id)
    except FileNotFoundError:
        raise HTTPException(404, "Model not found")
    df = _read_csv(file)
    t = meta["threshold"]
    probs = pipe.predict_proba(df)[:, 1]
    out = df.copy()
    out["churn_probability"] = probs.round(4)
    out["risk_segment"] = [risk_band(p, t) for p in probs]
    out = out.sort_values("churn_probability", ascending=False)
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=scored.csv"},
    )


def _explainer(model_id: str) -> Explainer:
    """Build (once) and cache the SHAP explainer for a model."""
    if model_id not in _EXPLAINERS:
        pipe, _ = registry.load(model_id)
        bg = pd.read_csv(registry.REGISTRY_DIR / model_id / "background.csv")
        _EXPLAINERS[model_id] = Explainer(pipe, bg)
    return _EXPLAINERS[model_id]


@app.post("/models/{model_id}/explain")
def explain(model_id: str, req: PredictRequest):
    """Top factors pushing one customer's churn risk up or down."""
    try:
        ex = _explainer(model_id)
    except FileNotFoundError:
        raise HTTPException(404, "Model or background data not found. Retrain the model.")
    return {"top_factors": ex.customer(pd.DataFrame([req.features]), k=5)}


@app.get("/models/{model_id}/importance")
def importance(model_id: str):
    """Global feature importance (mean absolute SHAP value)."""
    try:
        ex = _explainer(model_id)
    except FileNotFoundError:
        raise HTTPException(404, "Model or background data not found. Retrain the model.")
    bg = pd.read_csv(registry.REGISTRY_DIR / model_id / "background.csv")
    return ex.global_importance(bg)

@app.post("/models/{model_id}/recommend")
def recommend_plan(model_id: str, req: PredictRequest):
    """Retention plan from the prediction and SHAP factors (LLM, with rule fallback)."""
    try:
        pipe, meta = registry.load(model_id)
        ex = _explainer(model_id)
    except FileNotFoundError:
        raise HTTPException(404, "Model or background data not found. Retrain the model.")
    df = pd.DataFrame([req.features])
    p = float(pipe.predict_proba(df)[0, 1])
    factors = ex.customer(df, k=5)
    plan = llm_recommend(p, risk_band(p, meta["threshold"]), factors, req.features)
    return {"churn_probability": round(p, 4), "factors": factors, **plan}

@app.post("/models/{model_id}/drift")
def check_drift(model_id: str, file: UploadFile = File(...)):
    """PSI drift of new data against the model's training sample."""
    df = _read_csv(file)
    try:
        return drift_report(model_id, df)
    except FileNotFoundError:
        raise HTTPException(404, "Model or background data not found. Retrain the model.")


def _run_monitor(job_id: str, model_id: str, df: pd.DataFrame, force: bool):
    try:
        report = monitor(model_id, df, force=force)
        with _lock:
            JOBS[job_id] = {"status": "done", **report}
    except Exception as exc:
        with _lock:
            JOBS[job_id] = {"status": "failed", "error": str(exc)}


@app.post("/models/{model_id}/monitor")
def start_monitor(model_id: str, tasks: BackgroundTasks,
                  file: UploadFile = File(...), force: bool = Form(False)):
    """Drift check plus champion/challenger retraining, as a background job."""
    df = _read_csv(file)
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"status": "running"}
    tasks.add_task(_run_monitor, job_id, model_id, df, force)
    return {"job_id": job_id}

PORTFOLIOS: dict = {}   # model_id -> scored customer DataFrame (kept in memory)


class ChatRequest(BaseModel):
    message: str


@app.post("/models/{model_id}/portfolio")
def load_portfolio(model_id: str, file: UploadFile = File(...)):
    """Score a customer file once and keep it so the chatbot can answer questions."""
    try:
        pipe, meta = registry.load(model_id)
    except FileNotFoundError:
        raise HTTPException(404, "Model not found")
    df = _read_csv(file).reset_index(drop=True)
    probs = pipe.predict_proba(df)[:, 1]
    df["churn_probability"] = probs.round(4)
    df["risk_segment"] = [risk_band(p, meta["threshold"]) for p in probs]
    PORTFOLIOS[model_id] = df
    return {"rows": len(df)}


@app.post("/models/{model_id}/chat")
def chat(model_id: str, req: ChatRequest):
    if model_id not in PORTFOLIOS:
        raise HTTPException(409, "Load a customer file first.")
    _, meta = registry.load(model_id)
    try:
        ex = _explainer(model_id)
    except FileNotFoundError:
        ex = None
    return chat_answer(req.message, PORTFOLIOS[model_id], meta["schema"], ex, meta=meta)

@app.get("/models/{model_id}/cost")
def cost(model_id: str, value: float = 500, offer_cost: float = 50, success_rate: float = 1.0):
    try:
        return optimise(model_id, value, offer_cost, success_rate)
    except FileNotFoundError:
        raise HTTPException(404, "Model not found")

class WhatIfRequest(BaseModel):
    features: dict
    field: str
    new_value: str | float | int | None = None
    candidates: list | None = None


@app.post("/models/{model_id}/whatif")
def whatif(model_id: str, req: WhatIfRequest):
    try:
        if req.candidates:
            return {"options": best_options(model_id, req.features, req.field, req.candidates)}
        return what_if(model_id, req.features, req.field, req.new_value)
    except FileNotFoundError:
        raise HTTPException(404, "Model not found")
    except ValueError as exc:
        raise HTTPException(422, str(exc))

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/ui/")


app.mount("/ui", StaticFiles(directory=FRONTEND, html=True), name="ui")