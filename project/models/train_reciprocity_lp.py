from pathlib import Path
import os
import random
import json
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import joblib

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ============== Reproducibility ==============
os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
random.seed(42)

# ============== Paths ==============
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "wiki-talk-temporal.txt"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============== Load & split time ==============
E = pd.read_csv(
    RAW,
    sep=r"\s+",
    header=None,
    names=["src", "dst", "ts"],
    dtype={"src": "int64", "dst": "int64", "ts": "int64"},
)
E["ts"] = pd.to_datetime(E["ts"], unit="s", utc=True)
E = E[E["src"] != E["dst"]].sort_values("ts").reset_index(drop=True)

cutoff = E["ts"].quantile(0.8)
E_tr = E[E["ts"] <= cutoff].copy()
E_te = E[E["ts"] > cutoff].copy()

# ============== Candidates & labels (reciprocity) ==============
ab = (
    E_tr.groupby(["src", "dst"])
    .agg(
        ab_count=("ts", "count"),
        last_ab=("ts", "max"),
    )
    .reset_index()
)

ba = (
    E_tr.groupby(["dst", "src"])
    .size()
    .reset_index(name="ba_count")
    .rename(columns={"dst": "src", "src": "dst"})
)

C = ab.merge(ba, on=["src", "dst"], how="left").fillna({"ba_count": 0})
C = C[C["ba_count"] == 0].copy()  # chỉ giữ A->B chưa từng có B->A trong train

TE_BA = (
    E_te.groupby(["dst", "src"])
    .size()
    .reset_index(name="cnt")
    .rename(columns={"dst": "src", "src": "dst"})
)

C = C.merge(TE_BA[["src", "dst"]].assign(y=1), on=["src", "dst"], how="left")
C["y"] = C["y"].fillna(0).astype(int)

# ổn định thứ tự để tách dữ liệu tái lập
C = C.sort_values(["src", "dst", "last_ab", "ab_count"]).reset_index(drop=True)

# chẩn đoán nhãn
print(
    f"[Diag] Candidates: {len(C):,} | Positives: {int(C['y'].sum()):,} | Rate: {C['y'].mean():.6f}"
)

# nếu không có positive → tạo một lượng nhỏ positive giả chỉ để pipeline không vỡ (tùy chọn)
if C["y"].sum() == 0:
    raise RuntimeError("No positives found in test for reciprocity. Consider using a wider test window or different cutoff.")

# ============== Graph & features ==============
Gd = nx.DiGraph()
Gd.add_edges_from(E_tr[["src", "dst"]].itertuples(index=False, name=None))
Gu = Gd.to_undirected()


def deg(u: int) -> int:
    return Gd.out_degree(u) + Gd.in_degree(u)


def feats(row: pd.Series) -> pd.Series:
    a, b = int(row.src), int(row.dst)
    n_a = set(Gu.neighbors(a)) if a in Gu else set()
    n_b = set(Gu.neighbors(b)) if b in Gu else set()
    inter = n_a & n_b
    cn = len(inter)
    aa = 0.0
    for z in inter:
        dz = Gu.degree(z)
        if dz > 1:
            aa += 1.0 / np.log(dz)
    pa = deg(a) * deg(b)
    recency = (cutoff - row.last_ab).days
    return pd.Series(
        {
            "common_neighbors": cn,
            "adamic_adar": aa,
            "preferential_attachment": pa,
            "ab_count": int(row.ab_count),
            "recency_days": recency,
            "deg_a": deg(a),
            "deg_b": deg(b),
        }
    )


X = C.apply(feats, axis=1)

# giữ ID để truy vết dễ (không đưa vào model)
X = X.assign(
    src=C["src"].to_numpy(), dst=C["dst"].to_numpy(), last_ab=C["last_ab"].to_numpy()
)
y = C["y"].to_numpy()

feature_cols = [
    "common_neighbors",
    "adamic_adar",
    "preferential_attachment",
    "ab_count",
    "recency_days",
    "deg_a",
    "deg_b",
]

# ============== Split train/val ==============
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ============== Models ==============
models = {
    "logreg": Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, solver="lbfgs", class_weight="balanced"
                ),
            ),
        ]
    ),
    "rf": RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
    ),
}


def safe_auc(y_true, y_score):
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def safe_ap(y_true, y_score):
    try:
        if y_true.sum() == 0:
            return float("nan")
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def precision_at_frac(y_true, y_score, frac: float):
    n = len(y_score)
    if n == 0:
        return float("nan")
    k = max(1, int(frac * n))
    idx = np.argpartition(y_score, -k)[-k:]
    return float(np.sum(y_true[idx] == 1) / k)



all_metrics = {}
all_preds_rows = []

# ============== Train & Evaluate & Save ==============
for name, clf in models.items():
    clf.fit(Xtr[feature_cols], ytr)
    p = clf.predict_proba(Xva[feature_cols])[:, 1]

    auc = safe_auc(yva, p)
    ap = safe_ap(yva, p)
    p005 = precision_at_frac(yva, p, 0.005)  # 0.5%
    p01 = precision_at_frac(yva, p, 0.01)  # 1%
    p05 = precision_at_frac(yva, p, 0.05)  # 5%

    print(
        f"[{name}] AUC={auc:.3f} AP={ap:.3f} P@0.5%={p005:.3f} P@1%={p01:.3f} P@5%={p05:.3f}"
    )

    # metrics
    all_metrics[name] = {
        "AUC": auc,
        "AP": ap,
        "P_at_0p5pct": p005,
        "P_at_1pct": p01,
        "P_at_5pct": p05,
    }

    # predictions (giữ id + features + y_true + y_score)
    preds_df = Xva[["src", "dst", "last_ab"] + feature_cols].copy()
    preds_df = preds_df.assign(y_true=yva, y_score=p, model=name)
    all_preds_rows.append(preds_df)

    # save model
    joblib.dump(clf, OUT_DIR / f"reciprocity_lp_{name}.joblib")

# save metrics
(OUT_DIR / "reciprocity_lp_metrics.json").write_text(
    json.dumps(all_metrics, ensure_ascii=False, indent=2)
)

# save preds
all_preds = pd.concat(all_preds_rows, axis=0).reset_index(drop=True)
all_preds.to_csv(OUT_DIR / "reciprocity_lp_preds.csv", index=False)

# save RF importances
rf = models.get("rf", None)
if rf is not None and hasattr(rf, "feature_importances_"):
    imp_df = pd.DataFrame(
        {"feature": feature_cols, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)
    imp_df.to_csv(OUT_DIR / "reciprocity_lp_importances.csv", index=False)

# save meta for reproducibility
meta = {
    "cutoff": cutoff.isoformat(),
    "n_candidates": int(len(C)),
    "n_positive": int(C["y"].sum()),
    "positive_rate": float(C["y"].mean()),
    "features": feature_cols,
    "models": list(models.keys()),
    "seed": 42,
}
(OUT_DIR / "reciprocity_lp_meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2)
)

print(
    "[OK] Saved:",
    OUT_DIR / "reciprocity_lp_metrics.json",
    OUT_DIR / "reciprocity_lp_preds.csv",
    OUT_DIR / "reciprocity_lp_importances.csv",
    OUT_DIR / "reciprocity_lp_logreg.joblib",
    OUT_DIR / "reciprocity_lp_rf.joblib",
    OUT_DIR / "reciprocity_lp_meta.json",
)
