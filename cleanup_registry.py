import shutil
from pathlib import Path

KEEP = {
    "wa-fn-usec-telco-customer-churn-20261029-082222",
    "expresso-churn-100k-20261029-082351",
}
reg = Path("models/registry")

for d in sorted(reg.iterdir()):
    if d.is_dir() and d.name not in KEEP:
        shutil.rmtree(d)
        print("deleted", d.name)

for f in ("models/all_models.joblib", "models/best_model.joblib"):
    p = Path(f)
    if p.exists():
        p.unlink()
        print("deleted", f)

print("kept:", sorted(x.name for x in reg.iterdir()))