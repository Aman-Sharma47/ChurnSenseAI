"""Retention recommendations: local LLM (Ollama) with a rule-based fallback."""
from __future__ import annotations

import json
import re

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"      # change this one word to switch models
TIMEOUT = 120             # seconds; the first call loads the model into RAM


def _readable(feature: str) -> str:
    """Turn 'Contract_Month-to-month' into 'Contract is Month-to-month'."""
    if "_" in feature:
        col, val = feature.split("_", 1)
        return f"{col} is {val}"
    return feature


def _two_sentences(text: str) -> str:
    """Keep at most the first two sentences of a customer message."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:2])


def fallback_plan(probability: float, factors: list[dict]) -> dict:
    """Rule-based plan used when the LLM is unavailable. Works for any dataset."""
    raising = [_readable(f["feature"]) for f in factors if f["effect"] > 0][:3]
    if probability < 0.15:
        action = "Low risk. Maintain the current experience and review at the next cycle."
    elif raising:
        action = ("Contact the customer proactively and discuss the main risk drivers: "
                  + "; ".join(raising) + ".")
    else:
        action = "Contact the customer for a personalised service review."
    return {
        "summary": f"Churn risk is {probability:.0%}.",
        "actions": [action],
        "message": "Hello, we value you as a customer. We would like to hear your feedback "
                   "and see how we can serve you better.",
        "source": "rules",
    }


def _prompt(probability: float, risk: str, factors: list[dict], customer: dict) -> str:
    lines = [f"- {_readable(f['feature'])} ({f['direction']})" for f in factors]
    profile = ", ".join(f"{k}={v}" for k, v in list(customer.items())[:12])
    return (
        "You are a customer retention analyst. A machine learning model has already "
        "predicted this customer's churn risk. Do not change or question the prediction.\n"
        f"Churn probability: {probability:.0%} ({risk} risk)\n"
        "Main factors from the model:\n" + "\n".join(lines) + "\n"
        f"Customer profile: {profile}\n\n"
        "Write a practical retention plan.\n"
        "Rules:\n"
        "1. Suggest only actions that need no price change: a call, a check-in, "
        "onboarding help, a contract-review conversation, a feedback survey.\n"
        "2. Never mention discounts, percentages, free trials or specific prices.\n"
        "3. Never write raw field names (like MonthlyCharges) in the customer message.\n"
        "4. The message must be exactly 2 short sentences.\n"
        "Reply with JSON only, in this exact shape:\n"
        '{"summary": "one sentence", "actions": ["action 1", "action 2", "action 3"], '
        '"message": "two short friendly sentences to the customer"}'
    )


def recommend(probability: float, risk: str, factors: list[dict], customer: dict) -> dict:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": _prompt(probability, risk, factors, customer),
                  "stream": False, "format": "json",
                  "options": {"temperature": 0.3, "num_predict": 350}},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        text = r.json()["response"]
        text = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
        data = json.loads(text)
        actions = data.get("actions")
        if not isinstance(actions, list) or not data.get("summary") or not data.get("message"):
            raise ValueError("incomplete answer")
        return {
            "summary": str(data["summary"]),
            "actions": [str(a) for a in actions][:4],
            "message": _two_sentences(str(data["message"])),
            "source": f"llm:{MODEL}",
        }
    except Exception:
        return fallback_plan(probability, factors)