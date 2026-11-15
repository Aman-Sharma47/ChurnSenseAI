"""Portfolio chatbot: a rule-based router picks a tool, our code computes the facts,
and the LLM only words them (with guards that reject invented numbers or qualifiers)."""
from __future__ import annotations

import re

import pandas as pd
import requests

from core.llm import MODEL, OLLAMA_URL, _readable

TOP_WORDS = ("top", "riskiest", "highest risk", "most likely", "at risk", "worst", "risky")
SUMMARY_WORDS = ("how many", "summary", "overview", "total", "portfolio", "average", "overall")
GROUP_WORDS = ("by", "per", "each", "which", "compare", "across", "segment")
NUM = r"\d+(?:\.\d+)?"

# Words a small LLM tends to add wrongly (e.g. "high tenure" when tenure is 1 month).
QUALIFIERS = re.compile(r"\b(high|low|long|short|large|small)\b")
RISK_PHRASE = re.compile(
    r"\b(?:high|low|medium|critical)(?:\s+or\s+(?:high|critical))?\s+risk\b")

HELP_TEXT = (
    "I can answer questions about the customer file you loaded. Try: "
    "'Who are the top 10 riskiest customers?', 'Why is customer 7590-VHVEG likely to leave?', "
    "'Average churn risk by Contract', or 'Give me a portfolio summary'."
)

SUMMARY_NOTE = (
    "'Expected to churn' adds up every customer's probability. 'High or Critical' counts "
    "customers above the risk bands. They measure different things, so the numbers differ."
)


def _pct(p: float) -> str:
    """Percent text that never shows a misleading rounded 100% or 0%."""
    if p >= 0.995:
        return "over 99%"
    if p < 0.005:
        return "under 1%"
    return f"{p:.0%}"


def _id_column(df, schema):
    for c in schema.get("id_columns", []):
        if c in df.columns:
            return c
    return None


def _find_customer(message, df, id_col):
    lookup = {str(v).lower(): i for i, v in zip(df.index, df[id_col])}
    for tok in re.findall(r"[\w\-]+", message):
        if tok.lower() in lookup:
            return lookup[tok.lower()]
    return None


def _find_column(m, schema, df):
    cols = schema.get("categorical_features", []) + schema.get("numeric_features", [])
    for c in cols:
        if c in df.columns and re.search(rf"\b{re.escape(c.lower())}\b", m):
            return c
    return None


def _phrase(feature: str, row: pd.Series, numeric: set) -> str:
    """Describe a factor with the customer's real value, e.g. 'tenure is 1'."""
    if feature in numeric and feature in row.index:
        val = row[feature]
        if pd.isna(val):
            return f"{feature} is missing"
        try:
            return f"{feature} is {float(val):g}"
        except (TypeError, ValueError):
            return f"{feature} is {val}"
    return _readable(feature)


# ------------------------------------------------------------------ tools
def _top_risky(df, schema, id_col, n):
    n = max(1, min(n, 25))
    top = df.sort_values("churn_probability", ascending=False).head(n)
    extra = (schema.get("numeric_features", [])[:2] + schema.get("categorical_features", [])[:2])
    keep = ([id_col] if id_col else []) + ["churn_probability", "risk_segment"] + [
        c for c in extra if c in df.columns]
    table = top[keep].copy()
    if not id_col:
        table.insert(0, "row", table.index)
    table["churn_probability"] = table["churn_probability"].round(4)
    hi, lo = float(top["churn_probability"].iloc[0]), float(top["churn_probability"].iloc[-1])
    text = (f"The {len(top)} customers with the highest predicted churn risk have "
            f"probabilities from {_pct(hi)} down to {_pct(lo)}.")
    return {"intent": "top_risky", "text": text, "table": table.to_dict("records")}


def _explain_customer(df, schema, id_col, idx, explainer):
    row = df.loc[idx]
    p, seg, cid = float(row["churn_probability"]), row["risk_segment"], row[id_col]
    text = f"Customer {cid} has a {_pct(p)} churn probability ({seg} risk)."
    table = []
    if explainer is not None:
        numeric = set(schema.get("numeric_features", []))
        feats = [c for c in schema["feature_columns"] if c in df.columns]
        factors = explainer.customer(df.loc[[idx], feats], k=5)
        table = [{"factor": _phrase(f["feature"], row, numeric), "effect": f["effect"],
                  "direction": f["direction"]} for f in factors]
        text += " Main drivers: " + "; ".join(
            f"{t['factor']} ({t['direction']})" for t in table) + "."
    else:
        text += " Detailed reasons are not available for this model."
    return {"intent": "explain_customer", "text": text, "table": table}


def _by_column(df, col):
    key = df[col]
    if pd.api.types.is_numeric_dtype(key) and key.nunique() > 8:
        key = pd.qcut(key, 5, duplicates="drop").astype(str)
    g = (df.assign(_k=key).groupby("_k", dropna=False)
         .agg(avg=("churn_probability", "mean"), n=("churn_probability", "size"))
         .reset_index().sort_values("avg", ascending=False))
    table = [{col: str(r["_k"]), "avg_churn_probability": round(float(r["avg"]), 3),
              "customers": int(r["n"])} for _, r in g.iterrows()][:12]
    hi, lo = g.iloc[0], g.iloc[-1]
    text = (f"By {col}, the group with the highest average churn risk is {hi['_k']} "
            f"({_pct(float(hi['avg']))}) and the lowest is {lo['_k']} ({_pct(float(lo['avg']))}).")
    return {"intent": "by_column", "text": text, "table": table}


def _summary(df):
    counts = df["risk_segment"].value_counts()
    table = [{"risk_segment": s, "customers": int(counts.get(s, 0))}
             for s in ["Low", "Medium", "High", "Critical"]]
    hi = int(counts.get("High", 0) + counts.get("Critical", 0))
    text = (f"The file has {len(df):,} customers. The average churn probability is "
            f"{_pct(float(df['churn_probability'].mean()))}, and {hi:,} customers are at High or "
            f"Critical risk. About {df['churn_probability'].sum():,.0f} customers are expected to churn.")
    return {"intent": "summary", "text": text, "table": table, "note": SUMMARY_NOTE}


# ------------------------------------------------------------------ wording
def _llm_words(question, facts):
    prompt = (
        "You are an analytics assistant. Answer the question in 2 short sentences using ONLY "
        "the facts below.\n"
        "Rules: do not add any number that is not in the facts. Describe each driver exactly "
        "as written in the facts. Never say a value is high, low, long or short.\n"
        f"Question: {question}\nFacts: {facts}"
    )
    r = requests.post(OLLAMA_URL, json={
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.2, "num_predict": 150}}, timeout=60)
    r.raise_for_status()
    return r.json()["response"].strip()


def _grounded(answer_text, facts, question):
    """Reject answers with numbers or qualifier words that the facts do not contain."""
    allowed_nums = set(re.findall(NUM, facts)) | set(re.findall(NUM, question))
    if set(re.findall(NUM, answer_text)) - allowed_nums:
        return False
    a = RISK_PHRASE.sub("", answer_text.lower())
    f = RISK_PHRASE.sub("", (facts + " " + question).lower())
    return not (set(QUALIFIERS.findall(a)) - set(QUALIFIERS.findall(f)))


def _finish(res, question, use_llm):
    res["answer"], res["source"] = res["text"], "template"
    if use_llm and res["intent"] != "help":
        try:
            words = _llm_words(question, res["text"])
            if words and _grounded(words, res["text"], question):
                res["answer"], res["source"] = words, f"llm:{MODEL}"
        except Exception:
            pass
    return res

#--------------------------------------------------------------------
def _context(df, schema, meta):
    seg = df["risk_segment"].value_counts().to_dict()
    lines = [
        f"Customers loaded: {len(df)}",
        f"Average predicted churn probability: {df['churn_probability'].mean():.1%}",
        f"Customers per risk band: {seg}",
        "Risk bands: Low under 15%, Medium 15-30%, High 30-60%, Critical 60% and above.",
        f"Target column: {schema.get('target')}",
    ]
    if meta:
        t = meta["test_at_tuned"]
        lines += [f"Model: {meta['best_model']}", f"Test ROC-AUC: {t['roc_auc']}",
                  f"Test recall: {t['recall']}", f"Test precision: {t['precision']}",
                  f"Cut-off: {meta['threshold']:.2f}"]
    return "\n".join(lines)
# ------------------------------------------------------------------ router

def answer(message, df, schema, explainer, use_llm=True, meta=None):
    m = message.lower()
    id_col = _id_column(df, schema)
    idx = _find_customer(message, df, id_col) if id_col else None
    col = _find_column(m, schema, df)
    nums = re.findall(r"\b(\d{1,2})\b", m)
    n = int(nums[0]) if nums else 10

    if idx is not None:
        res = _explain_customer(df, schema, id_col, idx, explainer)
    elif col and any(re.search(rf"\b{w}\b", m) for w in GROUP_WORDS):
        res = _by_column(df, col)
    elif any(k in m for k in TOP_WORDS):
        res = _top_risky(df, schema, id_col, n)
    elif any(k in m for k in SUMMARY_WORDS):
        res = _summary(df)
    else:
        res = {"intent": "help", "text": HELP_TEXT, "table": []}
        res["answer"], res["source"] = res["text"], "template"
        if use_llm:
            from core.qa import general_answer
            text, src = general_answer(message, _context(df, schema, meta))
            if text:
                res.update(intent="general", text=text, answer=text, source=src)
        return res
    return _finish(res, message, use_llm)