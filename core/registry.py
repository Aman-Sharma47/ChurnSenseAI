"""Minimal file-based model registry: one folder per model version."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import joblib

REGISTRY_DIR = Path(__file__).resolve().parents[1] / "models" / "registry"


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower() or "dataset"


def register(result: dict, dataset_name: str, root: Path = REGISTRY_DIR) -> str:
    """Save a training result as a new version and return its model_id."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_id = f"{_slug(dataset_name)}-{stamp}"
    folder = root / model_id
    folder.mkdir(parents=True, exist_ok=True)

    joblib.dump(result["pipeline"], folder / "pipeline.joblib")
    result["comparison"].to_csv(folder / "cv_comparison.csv", index=False)
    result["test_predictions"].to_csv(folder / "test_predictions.csv", index=False)
    result["background"].to_csv(folder / "background.csv", index=False)

    meta = {
        "model_id": model_id,
        "dataset": dataset_name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "best_model": result["best_model"],
        "statistically_tied": result["statistically_tied"],
        "threshold": result["threshold"],
        "test_at_0.5": result["test_at_0.5"],
        "test_at_tuned": result["test_at_tuned"],
        "n_train": result["n_train"],
        "n_test": result["n_test"],
        "schema": result["schema"].to_dict(),
    }
    (folder / "metadata.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return model_id


def list_models(root: Path = REGISTRY_DIR) -> list[dict]:
    if not root.exists():
        return []
    items = []
    for meta_file in sorted(root.glob("*/metadata.json")):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        items.append({
            "model_id": meta["model_id"],
            "dataset": meta["dataset"],
            "created": meta["created"],
            "best_model": meta["best_model"],
            "roc_auc": meta["test_at_tuned"]["roc_auc"],
            "recall": meta["test_at_tuned"]["recall"],
        })
    return sorted(items, key=lambda m: m["created"], reverse=True)


def load(model_id: str, root: Path = REGISTRY_DIR):
    folder = root / model_id
    if not folder.exists():
        raise FileNotFoundError(f"Model '{model_id}' not found.")
    pipeline = joblib.load(folder / "pipeline.joblib")
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    return pipeline, meta