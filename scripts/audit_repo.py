"""Read-only repo audit. Run from the project root:
   python scripts/audit_repo.py > audit_report.txt"""
import ast
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".venv", ".git", "node_modules", "__pycache__", ".pytest_cache",
        ".ipynb_checkpoints", "work", "registry"}
TEXT = {".py", ".js", ".html", ".css", ".md", ".txt", ".json", ".toml", ".yml", ".yaml"}


def files():
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP):
            yield p


def rel(p):
    return str(p.relative_to(ROOT)).replace("\\", "/")


def norm(path):
    path = re.sub(r"\$?\{[^{}\n]*\}", "{x}", path)
    return path.split("?")[0].rstrip("/")


def head(t):
    print(f"\n===== {t} =====")


head("1. PYTHON FILES (lines, top-level defs)")
defs = defaultdict(list)
pyfiles = [p for p in files() if p.suffix == ".py"]
for p in pyfiles:
    src = p.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"{rel(p)}: SYNTAX ERROR {e}")
        continue
    names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    for n in names:
        defs[n].append(rel(p))
    print(f"{rel(p)}: {len(src.splitlines())} lines; {len(names)} defs: {', '.join(names[:12])}")

head("2. DUPLICATE TOP-LEVEL NAMES ACROSS FILES (excluding tests)")
for n, locs in defs.items():
    locs = [x for x in locs if not x.startswith("tests/")]
    if len(locs) > 1:
        print(f"{n}: {locs}")

head("3. CORE MODULES NOT IMPORTED ANYWHERE (heuristic)")
allsrc = {rel(p): p.read_text(encoding="utf-8", errors="ignore") for p in pyfiles}
for p in sorted((ROOT / "core").glob("*.py")):
    stem = p.stem
    if stem == "__init__":
        continue
    pat = re.compile(rf"core(\.|\s+import\s+[^\n]*\b){stem}\b")
    users = [f for f, s in allsrc.items() if f != rel(p) and pat.search(s)]
    print(f"core/{stem}.py: used by {users if users else 'NOTHING (check manually)'}")

head("4. API ENDPOINTS (api/main.py)")
api_paths = set()
mp = ROOT / "api" / "main.py"
if mp.exists():
    tree = ast.parse(mp.read_text(encoding="utf-8", errors="ignore"))
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            for d in n.decorator_list:
                if (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
                        and d.func.attr in {"get", "post", "put", "delete"} and d.args
                        and isinstance(d.args[0], ast.Constant)):
                    api_paths.add(norm(d.args[0].value))
                    print(f"{d.func.attr.upper():5} {d.args[0].value:45} -> {n.name}()")
else:
    print("api/main.py NOT FOUND")

head("5. ENDPOINTS CALLED BY EACH UI vs DEFINED")
call_re = re.compile(r"""(/(?:models|datasets|jobs|health)[^"'`\s)]*)""")
for ui in ("frontend/app.js", "dashboard/app_v2.py"):
    p = ROOT / ui
    if not p.exists():
        print(f"{ui}: NOT FOUND")
        continue
    txt = re.sub(r"\$?\{[^{}\n]*\}", "{x}", p.read_text(encoding="utf-8", errors="ignore"))
    called = {norm(m) for m in call_re.findall(txt)}
    print(f"{ui}: calls {len(called)} paths")
    print("  called but NOT defined:", sorted(called - api_paths) or "none")
    print("  defined but never called:", sorted(api_paths - called) or "none")

head("6. SECRET SCAN (values are never printed)")
pats = [re.compile(r"gsk_[A-Za-z0-9]{20,}"),
        re.compile(r"(GROQ_API_KEY|API_KEY|SECRET|TOKEN)\s*=\s*['\"][^'\"]{8,}['\"]", re.I)]
hits = 0
for p in files():
    if p.suffix in TEXT:
        s = p.read_text(encoding="utf-8", errors="ignore")
        for pat in pats:
            if pat.search(s):
                print(f"POSSIBLE SECRET in {rel(p)}")
                hits += 1
print("possible secrets found:", hits)

head("7. GIT")
for cmd in (["git", "ls-files"], ["git", "status", "--short"]):
    try:
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60).stdout.splitlines()
        print(" ".join(cmd), "->", len(out), "lines")
        if cmd[1] == "ls-files":
            for pat in ("node_modules", "work/", ".joblib", "registry", "cleanup_registry", ".env"):
                m = [x for x in out if pat in x]
                if m:
                    print(f"  tracked files containing '{pat}': {len(m)} (e.g. {m[:2]})")
        else:
            print("  ", out[:15])
    except Exception as e:
        print("git failed:", e)

head("8. REGISTRY MODELS")
reg = ROOT / "models" / "registry"
for d in sorted(reg.iterdir()) if reg.exists() else []:
    meta = d / "metadata.json"
    info = {}
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        t = m.get("test_at_tuned", {})
        info = {"best": m.get("best_model"), "auc": t.get("roc_auc"), "acc": t.get("accuracy"),
                "recall": t.get("recall"), "precision": t.get("precision"),
                "n_train": m.get("n_train"), "n_test": m.get("n_test")}
    print(d.name, info, "| background.csv:", (d / "background.csv").exists(),
          "| test_predictions.csv:", (d / "test_predictions.csv").exists())

head("9. KEY CODE FACTS")
def grep(file, pattern):
    p = ROOT / file
    if not p.exists():
        return f"{file} NOT FOUND"
    return [f"{i+1}: {l.strip()}" for i, l in enumerate(p.read_text(encoding='utf-8', errors='ignore').splitlines())
            if re.search(pattern, l)][:6]
print("trainer background line:", grep("core/trainer.py", r'"background"'))
print("trainer calibration:", grep("core/trainer.py", r"CalibratedClassifierCV"))
print("api fast= usage:", grep("api/main.py", r"fast="))
print("retrain fast= usage:", grep("core/retrain.py", r"fast="))
print("risk_band:", grep("api/main.py", r"def risk_band|>= 0\.\d+"))
print("qa model:", grep("core/qa.py", r"GROQ_MODEL\s*="))

head("10. TESTS")
tests = [p for p in pyfiles if rel(p).startswith("tests/")]
for p in tests:
    n = sum(1 for x in ast.walk(ast.parse(p.read_text(encoding="utf-8", errors="ignore")))
            if isinstance(x, ast.FunctionDef) and x.name.startswith("test_"))
    print(f"{rel(p)}: {n} test functions")
try:
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    print("\n".join((r.stdout + r.stderr).strip().splitlines()[-6:]))
except Exception as e:
    print("pytest failed:", e)

head("11. FRONTEND FILES")
for f in ("frontend/index.html", "frontend/style.css", "frontend/app.js"):
    p = ROOT / f
    print(f, "->", f"{len(p.read_text(encoding='utf-8', errors='ignore').splitlines())} lines" if p.exists() else "NOT FOUND")
print("\nDONE")