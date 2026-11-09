"""Simple scheduled-retraining entry point for future refreshed data."""
import argparse
import shutil
from pathlib import Path
from src.config import DATA_PATH
from src.train import train_all
from src.evaluate import create_evaluation_figures
from src.explain import create_shap_summary


def retrain(new_data: str | None = None):
    """Optionally replace the project CSV with a validated refreshed snapshot, then rebuild artifacts."""
    if new_data:
        source = Path(new_data)
        if not source.exists():
            raise FileNotFoundError(f"New dataset not found: {source}")
        # Keep the project contract stable: a refreshed extract must retain the official schema.
        shutil.copy2(source, DATA_PATH)
    results = train_all()
    create_evaluation_figures()
    create_shap_summary()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain ChurnSense AI on an updated official-schema CSV.")
    parser.add_argument("--new-data", help="Path to a refreshed IBM Telco-format CSV")
    args = parser.parse_args()
    print(retrain(args.new_data).to_string(index=False))
