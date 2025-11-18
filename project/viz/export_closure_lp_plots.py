from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGS = ROOT / "data" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

preds_path = PROCESSED / "closure_lp_preds.csv"
imp_path   = PROCESSED / "closure_lp_importances.csv"

# Đọc preds (t1,t2,t là datetime nếu có; không bắt buộc)
parse_cols = [c for c in ["t1","t2","t"] if c in pd.read_csv(preds_path, nrows=0).columns]
preds = pd.read_csv(preds_path, parse_dates=parse_cols)

models = preds["model"].astype(str).unique().tolist()

def p_at_frac(y_true, y_score, frac=0.01):
    n = len(y_score); k = max(1, int(frac*n))
    thr = np.partition(y_score, -k)[-k]
    return ((y_score >= thr).astype(int) & (y_true==1)).sum() / k

# ROC
plt.figure(figsize=(10, 4.5))
for m in models:
    df = preds[preds["model"]==m]
    y, s = df["y_true"].values, df["y_score"].values
    if len(np.unique(y)) < 2:
        continue
    fpr, tpr, _ = roc_curve(y, s)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{m} (AUC={roc_auc:.3f})")
plt.plot([0,1],[0,1],"k--",alpha=0.3)
plt.title("ROC — closure LP")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(FIGS / "lp_closure_roc.png", dpi=200); plt.close()

# PR
plt.figure(figsize=(10, 4.5))
for m in models:
    df = preds[preds["model"]==m]
    y, s = df["y_true"].values, df["y_score"].values
    if y.sum() == 0:
        continue
    prec, rec, _ = precision_recall_curve(y, s)
    ap = average_precision_score(y, s)
    plt.plot(rec, prec, label=f"{m} (AP={ap:.3f})")
plt.title("PR — closure LP")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(FIGS / "lp_closure_pr.png", dpi=200); plt.close()

# P@k
plt.figure(figsize=(7, 4))
fracs = [0.005, 0.01, 0.05]
barw = 0.8/len(fracs)
for i, frac in enumerate(fracs):
    vals = []
    for m in models:
        df = preds[preds["model"]==m]
        vals.append(p_at_frac(df["y_true"].values, df["y_score"].values, frac))
    xs = np.arange(len(models)) + i*barw
    plt.bar(xs, vals, width=barw, label=f"P@{int(frac*100)}%")
plt.xticks(np.arange(len(models)) + barw, models)
plt.ylim(0,1); plt.title("P@k — closure LP"); plt.legend()
plt.tight_layout(); plt.savefig(FIGS / "lp_closure_p_at_k.png", dpi=200); plt.close()

# Feature importance (RF)
if imp_path.exists():
    imp = pd.read_csv(imp_path).sort_values("importance", ascending=True)
    plt.figure(figsize=(7,4))
    plt.barh(imp["feature"], imp["importance"])
    plt.title("Feature importance — RF (closure LP)")
    plt.tight_layout(); plt.savefig(FIGS / "lp_closure_rf_importance.png", dpi=200); plt.close()

print("[OK] saved:",
      FIGS / "lp_closure_roc.png",
      FIGS / "lp_closure_pr.png",
      FIGS / "lp_closure_p_at_k.png")
if imp_path.exists():
    print("[OK] saved:", FIGS / "lp_closure_rf_importance.png")
