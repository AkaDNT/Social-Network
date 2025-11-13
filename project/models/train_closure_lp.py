# models/train_closure_lp.py
from pathlib import Path
import os, random, json
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

# ============== Params ==============
DELTA_DAYS = 30
DELTA = pd.Timedelta(days=DELTA_DAYS)

# Giới hạn & sampling để tránh phình dữ liệu khi sinh nêm
NEG_MULT = 5            # giữ tối đa 5× negatives so với positives (tối thiểu 5k)
NEG_MIN = 5000
NEG_CAP = 200_000       # trần negatives
WEDGE_CAP_WARN = 5_000_000  # nếu vượt con số này sẽ cảnh báo

# ============== Paths ==============
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "wiki-talk-temporal.txt"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============== Load & split time ==============
E = pd.read_csv(
    RAW, sep=r"\s+", header=None, names=["src", "dst", "ts"],
    dtype={"src": "int64", "dst": "int64", "ts": "int64"}
)
E["ts"] = pd.to_datetime(E["ts"], unit="s", utc=True)
E = E[E["src"] != E["dst"]].sort_values("ts").reset_index(drop=True)

cutoff = E["ts"].quantile(0.8)
E_tr = E[E["ts"] <= cutoff].copy()
E_te = E[E["ts"] > cutoff].copy()

print(f"[Load] Train edges: {len(E_tr):,} | Test edges: {len(E_te):,} | Δ={DELTA_DAYS}d")

# ============== Index test edges for fast labeling ==============
# Map (u,v) -> np.ndarray[datetime64[ns]] (đã sort) trong TEST
TE_map = (
    E_te.groupby(["src", "dst"])["ts"]
        .apply(lambda s: s.to_numpy(dtype="datetime64[ns]"))
        .to_dict()
)

# ============== Generate wedges A->B, B->C in TRAIN ==============
def generate_wedges(df: pd.DataFrame, delta: pd.Timedelta) -> pd.DataFrame:
    """
    Sinh nêm theo node B, chỉ ghép các cạnh B->C trong [t1, t1+Δ] để tránh Descartes nổ RAM.
    Mỗi nêm có (A,B,C,t1,t2,t=max(t1,t2)).
    """
    by_dst = df.groupby("dst")   # in-edges:  A -> B  (t1)
    by_src = df.groupby("src")   # out-edges: B -> C  (t2)
    commons = set(by_dst.groups).intersection(by_src.groups)

    rows = []
    nB = len(commons)
    for idxB, B in enumerate(commons, 1):
        Ein = by_dst.get_group(B)[["src", "ts"]].rename(columns={"src": "A", "ts": "t1"}).sort_values("t1")
        Eout = by_src.get_group(B)[["dst", "ts"]].rename(columns={"dst": "C", "ts": "t2"}).sort_values("t2")

        # Mảng thời gian để search nhanh
        t2s = Eout["t2"].to_numpy(dtype="datetime64[ns]")
        Cs  = Eout["C"].to_numpy()
        t2s_pd = Eout["t2"].to_numpy()  # pandas datetime64[ns, UTC]

        for a, t1 in zip(Ein["A"].to_numpy(), Ein["t1"]):
            lo = np.datetime64(t1.to_datetime64())
            hi = np.datetime64((t1 + delta).to_datetime64())

            L = np.searchsorted(t2s, lo, side="left")
            R = np.searchsorted(t2s, hi, side="right")
            if L >= R:
                continue

            # duyệt từng B->C có t2 trong [t1, t1+Δ]
            for j in range(L, R):
                c = Cs[j]
                if a == c:
                    continue
                t2_pd = pd.Timestamp(t2s_pd[j])  # tz-aware
                t = max(t1, t2_pd)
                rows.append((a, B, c, t1, t2_pd, t))

        # Thông tin tiến độ thưa (chỉ để yên tâm khi chạy lâu)
        if idxB % 1000 == 0:
            print(f"[Wedge] processed B {idxB}/{nB} (rows so far: {len(rows):,})")

    W = pd.DataFrame(rows, columns=["A", "B", "C", "t1", "t2", "t"])
    return W

W = generate_wedges(E_tr, DELTA)
print(f"[Wedge] Total wedges generated: {len(W):,}")
if len(W) > WEDGE_CAP_WARN:
    print(f"[Warn] Wedge count is large ({len(W):,}). Consider narrowing Δ or sampling.")

# ============== Label: closure in TEST within (t, t+Δ] ==============
def has_future_edge(arr: np.ndarray, lo: np.datetime64, hi: np.datetime64) -> bool:
    if arr is None or len(arr) == 0:
        return False
    i = np.searchsorted(arr, lo, side="right")  # strictly after t
    return bool(i < len(arr) and arr[i] <= hi)

def label_row(r: pd.Series) -> int:
    lo = np.datetime64(r["t"].to_datetime64())
    hi = np.datetime64((r["t"] + DELTA).to_datetime64())
    ac = TE_map.get((r["A"], r["C"]))
    ca = TE_map.get((r["C"], r["A"]))
    return int(has_future_edge(ac, lo, hi) or has_future_edge(ca, lo, hi))

print("[Label] Labeling wedges ...")
W["y"] = W.apply(label_row, axis=1)

pos = W[W["y"] == 1]
neg = W[W["y"] == 0]
print(f"[Label] Positives: {len(pos):,} | Negatives: {len(neg):,} | Pos rate: {W['y'].mean():.6f}")

# Sampling negatives để cân bằng & tiết kiệm thời gian
if len(pos) == 0:
    # Không có positive → giữ cỡ nhỏ để pipeline vẫn chạy
    Wb = W.sample(min(len(W), max(NEG_MIN, 50_000)), random_state=42)
    print("[Warn] No positive labels found. Using a small random subset for pipeline sanity.")
else:
    keep_neg = min(len(neg), max(NEG_MIN, min(NEG_CAP, NEG_MULT * len(pos))))
    neg_smp = neg.sample(keep_neg, random_state=42) if keep_neg < len(neg) else neg
    Wb = pd.concat([pos, neg_smp], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"[Sample] Trainable wedges: {len(Wb):,}")

# ============== Graph & cached counts for features ==============
Gd = nx.DiGraph()
Gd.add_edges_from(E_tr[["src", "dst"]].itertuples(index=False, name=None))
Gu = Gd.to_undirected()

# pair counts for intensity
pair_counts = E_tr.groupby(["src", "dst"]).size().to_dict()

def deg(x: int) -> int:
    return Gd.in_degree(x) + Gd.out_degree(x)

def f_ab(a, b):
    return pair_counts.get((a, b), 0)

def features(row: pd.Series) -> pd.Series:
    a, b, c = int(row.A), int(row.B), int(row.C)
    # neighborhoods
    na = set(Gu.neighbors(a)) if a in Gu else set()
    nc = set(Gu.neighbors(c)) if c in Gu else set()
    inter = na & nc
    cn = len(inter)
    aa = sum(1.0 / np.log(Gu.degree(z)) for z in inter if Gu.degree(z) > 1)
    jacc = cn / max(1, len(na | nc))
    pa_ac = deg(a) * deg(c)
    # clustering của B (khả năng đóng tam giác quanh B)
    clus_b = nx.clustering(Gu, b) if b in Gu else 0.0
    # intensity & time
    ab_cnt = f_ab(a, b)
    bc_cnt = f_ab(b, c)
    gap = abs((row.t2 - row.t1).days)
    return pd.Series({
        "common_neighbors_ac": cn,
        "adamic_adar_ac": aa,
        "jaccard_ac": jacc,
        "preferential_attachment_ac": pa_ac,
        "deg_a": deg(a), "deg_b": deg(b), "deg_c": deg(c),
        "clustering_b": clus_b,
        "ab_count": ab_cnt, "bc_count": bc_cnt,
        "gap_days": gap,
    })

print("[Feat] Computing features ...")
X = Wb.apply(features, axis=1)
y = Wb["y"].to_numpy()

# giữ ID & thời gian để truy vết
id_cols = ["A", "B", "C", "t1", "t2", "t"]
X = X.assign(**{k: Wb[k].to_numpy() for k in id_cols})

feature_cols = [
    "common_neighbors_ac", "adamic_adar_ac", "jaccard_ac",
    "preferential_attachment_ac",
    "deg_a", "deg_b", "deg_c",
    "clustering_b",
    "ab_count", "bc_count",
    "gap_days",
]

# ============== Split train/val ==============
Xtr, Xva, ytr, yva = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
)

# ============== Models ==============
models = {
    "logreg": Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced"))
    ]),
    "rf": RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
    )
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
    thr = np.partition(y_score, -k)[-k]
    tp = ((y_score >= thr).astype(int) & (y_true == 1)).sum()
    return float(tp / k)

# ============== Train & Evaluate & Save ==============
all_metrics = {}
all_preds_rows = []

for name, clf in models.items():
    clf.fit(Xtr[feature_cols], ytr)
    p = clf.predict_proba(Xva[feature_cols])[:, 1]

    auc = safe_auc(yva, p)
    ap  = safe_ap(yva, p)
    p005 = precision_at_frac(yva, p, 0.005)
    p01  = precision_at_frac(yva, p, 0.01)
    p05  = precision_at_frac(yva, p, 0.05)

    print(f"[{name}] AUC={auc:.3f} AP={ap:.3f} P@0.5%={p005:.3f} P@1%={p01:.3f} P@5%={p05:.3f}")

    all_metrics[name] = {
        "AUC": auc, "AP": ap,
        "P_at_0p5pct": p005, "P_at_1pct": p01, "P_at_5pct": p05
    }

    preds_df = Xva[id_cols + feature_cols].copy()
    preds_df = preds_df.assign(y_true=yva, y_score=p, model=name)
    all_preds_rows.append(preds_df)

    # save model
    joblib.dump(clf, OUT_DIR / f"closure_lp_{name}.joblib")

# save metrics
(OUT_DIR / "closure_lp_metrics.json").write_text(
    json.dumps(all_metrics, ensure_ascii=False, indent=2)
)

# save preds
all_preds = pd.concat(all_preds_rows, axis=0).reset_index(drop=True)
all_preds.to_csv(OUT_DIR / "closure_lp_preds.csv", index=False)

# RF importances
rf = models.get("rf", None)
if rf is not None and hasattr(rf, "feature_importances_"):
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    imp_df.to_csv(OUT_DIR / "closure_lp_importances.csv", index=False)

# save meta
meta = {
    "cutoff": cutoff.isoformat(),
    "delta_days": DELTA_DAYS,
    "n_wedges": int(len(W)),
    "n_trainable": int(len(Wb)),
    "n_positive": int(Wb["y"].sum()),
    "positive_rate": float(Wb["y"].mean()),
    "features": feature_cols,
    "models": list(models.keys()),
    "seed": 42
}
(OUT_DIR / "closure_lp_meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2)
)

print("[OK] Saved:",
      OUT_DIR / "closure_lp_metrics.json",
      OUT_DIR / "closure_lp_preds.csv",
      OUT_DIR / "closure_lp_importances.csv",
      OUT_DIR / "closure_lp_logreg.joblib",
      OUT_DIR / "closure_lp_rf.joblib",
      OUT_DIR / "closure_lp_meta.json")
