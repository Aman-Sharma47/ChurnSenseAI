"""General Q&A for the assistant: Groq if a key is set, else Ollama, else nothing.
Only a small context sheet (aggregates, no customer rows) is ever sent."""
from __future__ import annotations

import os

import requests

from core.llm import MODEL as OLLAMA_MODEL, OLLAMA_URL
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

SYSTEM = (
    "You are the assistant inside ChurnSense AI, a customer churn prediction platform. "
    "Answer questions about churn, machine learning metrics, and the project, in plain simple "
    "language, in at most 5 sentences. Use ONLY the context below for any number about this "
    "project; never invent numbers, prices or customer details. If the question is unrelated "
    "to churn, the project or the loaded customers, politely say you can only help with those. "
    "Facts: churn means a customer leaves. A churn probability is the model's estimated chance "
    "that one customer leaves, from 0 to 1. The probabilities are calibrated, so 0.30 means "
    "about 30 in 100 such customers leave. Recall is the share of real leavers we catch; "
    "precision is the share of flagged customers who really leave. ROC-AUC is the chance a "
    "random leaver scores higher than a random stayer. SHAP shows which features pushed one "
    "prediction up or down. Drift (PSI) measures how much new data differs from training data. "
    "The ML model predicts; the LLM only explains."
)


def _groq(question: str, context: str):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    body = {"model": GROQ_MODEL, "temperature": 0.2, "max_tokens": 1200,
            "messages": [{"role": "system", "content": SYSTEM + "\n\nContext:\n" + context},
                         {"role": "user", "content": question}]}
    if "gpt-oss" in GROQ_MODEL:
        body["reasoning_effort"] = "low"
    r = requests.post(GROQ_URL, timeout=30,
                      headers={"Authorization": f"Bearer {key}"}, json=body)
    r.raise_for_status()
    text = (r.json()["choices"][0]["message"].get("content") or "").strip()
    return (text, f"groq:{GROQ_MODEL}") if text else None


def _ollama(question: str, context: str):
    r = requests.post(
        OLLAMA_URL, timeout=60,
        json={"model": OLLAMA_MODEL, "stream": False,
              "options": {"temperature": 0.2, "num_predict": 200},
              "prompt": SYSTEM + "\n\nContext:\n" + context + "\n\nQuestion: " + question})
    r.raise_for_status()
    return r.json()["response"].strip(), f"llm:{OLLAMA_MODEL}"


def general_answer(question: str, context: str):
    """Return (answer, source), or (None, None) if no LLM is reachable."""
    for fn in (_groq, _ollama):
        try:
            out = fn(question, context)
            if out and out[0]:
                return out
            if fn is _groq:
                print("[qa] Groq skipped: GROQ_API_KEY is not set in this terminal")
        except Exception as exc:
            print(f"[qa] {fn.__name__} failed: {exc}")
    return None, None