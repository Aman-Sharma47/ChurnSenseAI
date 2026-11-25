import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
paths = [Path(p) for p in sys.argv[1:]] or sorted(
    list((ROOT / "data").glob("*.csv")) + list((ROOT / "data" / "raw").glob("*.csv")))
for p in paths:
    if p.name.startswith("demo_"):
        continue
    df = pd.read_csv(p, nrows=200_000)
    print(f"\n===== {p.name}: {df.shape[0]} rows x {df.shape[1]} columns =====")
    for c in df.columns:
        s = df[c]
        n, miss = s.nunique(dropna=True), int(s.isna().sum())
        if n > 0.98 * max(len(s) - miss, 1):
            info = "ID-like (values hidden)"
        elif pd.api.types.is_numeric_dtype(s):
            info = f"min {s.min():g}, median {s.median():g}, max {s.max():g}"
        elif n <= 12:
            info = "top: " + ", ".join(f"{k} ({v})" for k, v in s.value_counts().head(4).items())
        else:
            info = f"{n} distinct values"
        print(f"{c:28} {str(s.dtype):10} missing {miss:6} unique {n:6} | {info}")