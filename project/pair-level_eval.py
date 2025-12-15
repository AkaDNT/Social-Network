from pathlib import Path
import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def guess_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "data" / "processed").exists():
            return p
    return start.parent

ROOT = guess_root(Path(__file__))
PROC = ROOT / "data" / "processed"

print("CWD =", os.getcwd())
print("ROOT =", ROOT)
print("PROC =", PROC)
print("Exists PROC?", PROC.exists())
print("Files in PROC (first 10):", [p.name for p in sorted(PROC.glob("*"))[:10]])

preds_path = PROC / "closure_lp_preds.csv"
meta_path  = PROC / "closure_lp_meta.json"

preds = pd.read_csv(preds_path)
meta  = json.loads(meta_path.read_text(encoding="utf-8"))
w_neg = float(meta["w_neg"])

df = preds[preds["model"] == "rf"].copy()

# gộp theo (A,C): lấy max score, label = max y_true
g = df.groupby(["A", "C"], as_index=False).agg(
    y_true=("y_true", "max"),
    y_score=("y_score", "max"),
)

y = g["y_true"].to_numpy(dtype=int)
s = g["y_score"].to_numpy(dtype=float)
w = np.ones(len(y), dtype=float)
w[y == 0] = w_neg

print("pairs:", len(g))
print("AUC_w(pair):", roc_auc_score(y, s, sample_weight=w))
print("AP_w(pair):",  average_precision_score(y, s, sample_weight=w))
