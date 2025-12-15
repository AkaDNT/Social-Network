from __future__ import annotations

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

# ========= Reproducibility =========
os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
random.seed(42)
rng = np.random.RandomState(42)

# ========= Params =========
DELTA_DAYS = 30
DELTA = pd.Timedelta(days=DELTA_DAYS)

# Giới hạn/điều tiết để không phình RAM & thời gian
NEG_CAP        = 200_000     # tối đa negatives giữ lại (reservoir)
NEG_MIN        = 5_000       # tối thiểu negatives (warning nếu ít)
LOG_EVERY_B    = 1000        # log mỗi n node B
TIME_TRIM_DAYS = None        # ví dụ: 365 để chỉ sinh nêm trong 1 năm cuối của train

# Tránh “closure giả” khi A↔C đã tồn tại trong train
SKIP_EXISTING_AC_IN_TRAIN = True

# ========= Paths =========
ROOT = Path(__file__).resolve().parents[1]   # project/
RAW  = ROOT / "data" / "raw" / "wiki-talk-temporal.txt"
OUT  = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# ========= Load =========
E = pd.read_csv(
    RAW, sep=r"\s+", header=None, names=["src", "dst", "ts"],
    dtype={"src": "int64", "dst": "int64", "ts": "int64"},
)
E["ts"] = pd.to_datetime(E["ts"], unit="s", utc=True)
E = E[E["src"] != E["dst"]].sort_values("ts").reset_index(drop=True)

cutoff = E["ts"].quantile(0.8)
E_tr = E[E["ts"] <= cutoff].copy()
E_te = E[E["ts"] >  cutoff].copy()

# (tuỳ chọn) chỉ lấy phần train gần cutoff để bớt tổ hợp
if TIME_TRIM_DAYS is not None:
    lo = cutoff - pd.Timedelta(days=TIME_TRIM_DAYS)
    E_tr = E_tr[E_tr["ts"] >= lo].copy()

print(f"[Load] Train edges: {len(E_tr):,} | Test edges: {len(E_te):,} | Δ={DELTA_DAYS}d")

# ========= Build train graphs (for features & for skip-existing-AC check) =========
Gd = nx.DiGraph()
Gd.add_edges_from(E_tr[["src", "dst"]].itertuples(index=False, name=None))
Gu = Gd.to_undirected()

pair_counts = E_tr.groupby(["src", "dst"]).size().to_dict()

def deg(u: int) -> int:
    return Gd.in_degree(u) + Gd.out_degree(u)

def pc(u: int, v: int) -> int:
    return pair_counts.get((u, v), 0)

# ========= TEST index for labeling (sorted for searchsorted) =========
TE_map = (
    E_te.groupby(["src", "dst"])["ts"]
        .apply(lambda s: np.sort(s.to_numpy(dtype="datetime64[ns]")))
        .to_dict()
)

def has_future_edge(arr: np.ndarray, lo: np.datetime64, hi: np.datetime64) -> bool:
    """Check whether there exists timestamp in (lo, hi] in sorted array."""
    if arr is None or len(arr) == 0:
        return False
    i = np.searchsorted(arr, lo, side="right")  # (lo, ...]
    return bool(i < len(arr) and arr[i] <= hi)

# ========= Generate & sample wedges (streaming) =========
def collect_wedges_streaming(df: pd.DataFrame):
    """
    Duyệt theo B, sinh nêm A->B (t1) & B->C (t2) với t2 ∈ [t1, t1+Δ].

    - Label y=1 nếu trong TEST có A->C hoặc C->A trong (t, t+Δ] với t=max(t1,t2)
    - Giữ TẤT CẢ positives.
    - Negatives dùng reservoir sampling tới NEG_CAP.
    - Trả về: pos_rows, neg_rows, neg_seen
      (neg_seen = tổng negative đã gặp khi duyệt, trước khi reservoir sample)
    """
    pos_rows, neg_rows = [], []
    neg_seen = 0

    by_dst = df.groupby("dst")   # A->B
    by_src = df.groupby("src")   # B->C

    commons = sorted(set(by_dst.groups).intersection(by_src.groups))  # stable order
    nB = len(commons)

    for idxB, B in enumerate(commons, 1):
        Ein  = by_dst.get_group(B)[["src", "ts"]].rename(columns={"src": "A", "ts": "t1"}).sort_values("t1")
        Eout = by_src.get_group(B)[["dst", "ts"]].rename(columns={"dst": "C", "ts": "t2"}).sort_values("t2")

        # sorted arrays for time window indexing
        t2s = Eout["t2"].to_numpy(dtype="datetime64[ns]")
        t2s_pd = Eout["t2"].to_numpy()
        Cs  = Eout["C"].to_numpy(dtype=np.int64)

        for a, t1 in zip(Ein["A"].to_numpy(dtype=np.int64), Ein["t1"]):
            lo = np.datetime64(t1.to_datetime64())
            hi = np.datetime64((t1 + DELTA).to_datetime64())

            L = np.searchsorted(t2s, lo, side="left")
            R = np.searchsorted(t2s, hi, side="right")
            if L >= R:
                continue

            for j in range(L, R):
                c = int(Cs[j])
                if int(a) == c:
                    continue

                # tránh closure “giả” nếu A↔C đã tồn tại trong train (bất kỳ chiều)
                if SKIP_EXISTING_AC_IN_TRAIN:
                    if Gd.has_edge(int(a), c) or Gd.has_edge(c, int(a)):
                        continue

                t2_pd = t2s_pd[j]
                t_pd  = max(t1, t2_pd)

                lo_t = np.datetime64(t_pd.to_datetime64())
                hi_t = np.datetime64((t_pd + DELTA).to_datetime64())

                y = int(
                    has_future_edge(TE_map.get((int(a), c)), lo_t, hi_t) or
                    has_future_edge(TE_map.get((c, int(a))), lo_t, hi_t)
                )

                row = (int(a), int(B), c, t1, t2_pd, t_pd, y)
                if y == 1:
                    pos_rows.append(row)
                else:
                    neg_seen += 1
                    if len(neg_rows) < NEG_CAP:
                        neg_rows.append(row)
                    else:
                        r = rng.randint(0, neg_seen)
                        if r < NEG_CAP:
                            neg_rows[r] = row

        if idxB % LOG_EVERY_B == 0:
            print(f"[Wedge] processed B {idxB}/{nB} | pos:{len(pos_rows):,} | neg_reservoir:{len(neg_rows):,} | neg_seen:{neg_seen:,}")

    if len(neg_rows) < NEG_MIN and neg_seen > 0:
        print(f"[Info] Negatives in reservoir too small ({len(neg_rows)}). "
              f"Consider lowering TIME_TRIM_DAYS or increasing NEG_CAP.")

    return pos_rows, neg_rows, neg_seen

print("[Wedge] Streaming generation & sampling ...")
pos_rows, neg_rows, neg_seen = collect_wedges_streaming(E_tr)
print(f"[Wedge] Done. Pos kept: {len(pos_rows):,} | Neg kept (reservoir): {len(neg_rows):,} | Neg seen: {neg_seen:,}")

# ========= Build trainable DataFrame (sampled negatives + all positives) =========
cols = ["A", "B", "C", "t1", "t2", "t", "y"]
Wsel = pd.DataFrame(pos_rows + neg_rows, columns=cols)
Wsel = Wsel.sample(frac=1.0, random_state=42).reset_index(drop=True)

n_pos = int((Wsel["y"] == 1).sum())
n_neg_sampled = int((Wsel["y"] == 0).sum())

# ========= Importance weighting (to unbias base rate) =========
# Each sampled negative represents neg_seen / n_neg_sampled negatives in the full wedge universe.
if n_neg_sampled > 0 and neg_seen > 0:
    w_neg = float(neg_seen / n_neg_sampled)
else:
    w_neg = 1.0

sample_weight = np.ones(len(Wsel), dtype=float)
sample_weight[Wsel["y"].to_numpy(dtype=int) == 0] = w_neg

true_total_est = n_pos + int(neg_seen)
true_pos_rate_est = n_pos / max(1, true_total_est)

print(f"[Wedge] Trainable wedges: {len(Wsel):,} | Pos rate (sampled): {Wsel['y'].mean():.6f}")
print(f"[Weight] neg_seen={neg_seen:,} | neg_sampled={n_neg_sampled:,} | w_neg={w_neg:.3f}")
print(f"[BaseRate~] pos={n_pos:,} | total~={true_total_est:,} | pos_rate~={true_pos_rate_est:.6f}")

# ========= Feature caches =========
Bs = Wsel["B"].unique().tolist()
clus_b_map = nx.clustering(Gu, nodes=Bs) if len(Bs) else {}

def make_features(row: pd.Series) -> pd.Series:
    a, b, c = int(row.A), int(row.B), int(row.C)

    na = set(Gu.neighbors(a)) if a in Gu else set()
    nc = set(Gu.neighbors(c)) if c in Gu else set()
    inter = na & nc

    cn = len(inter)
    aa = sum(1.0 / np.log(Gu.degree(z)) for z in inter if Gu.degree(z) > 1)
    jacc = cn / max(1, len(na | nc))
    pa_ac = deg(a) * deg(c)
    clus_b = float(clus_b_map.get(b, 0.0))
    gap = abs((row.t2 - row.t1).days)

    return pd.Series({
        "common_neighbors_ac": cn,
        "adamic_adar_ac": aa,
        "jaccard_ac": jacc,
        "preferential_attachment_ac": pa_ac,
        "deg_a": deg(a), "deg_b": deg(b), "deg_c": deg(c),
        "clustering_b": clus_b,
        "ab_count": pc(a, b), "bc_count": pc(b, c),
        "gap_days": gap,
    })

print("[Feat] Computing features ...")
X = Wsel.apply(make_features, axis=1)
y = Wsel["y"].to_numpy(dtype=int)

# giữ ID/time để truy vết
id_cols = ["A", "B", "C", "t1", "t2", "t"]
for k in id_cols:
    X[k] = Wsel[k].to_numpy()

feature_cols = [
    "common_neighbors_ac", "adamic_adar_ac", "jaccard_ac",
    "preferential_attachment_ac",
    "deg_a", "deg_b", "deg_c",
    "clustering_b",
    "ab_count", "bc_count",
    "gap_days",
]

# ========= Split (include weights) =========
Xtr, Xva, ytr, yva, wtr, wva = train_test_split(
    X, y, sample_weight,
    test_size=0.2, random_state=42,
    stratify=y if len(np.unique(y)) > 1 else None
)

# ========= Models =========
# IMPORTANT:
# - Vì đã dùng sample_weight để sửa base-rate, KHÔNG nên dùng class_weight="balanced" nữa (double correction).
models = {
    "logreg": Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", class_weight=None))
    ]),
    "rf": RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1,
        class_weight=None
    )
}

def safe_auc(y_true, y_score, sample_weight=None):
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score, sample_weight=sample_weight))
    except Exception:
        return float("nan")

def safe_ap(y_true, y_score, sample_weight=None):
    try:
        if np.sum(y_true) == 0:
            return float("nan")
        return float(average_precision_score(y_true, y_score, sample_weight=sample_weight))
    except Exception:
        return float("nan")

# FIX: đúng top-k (không bị ties của RF làm P@k > 1)
def precision_at_frac(y_true, y_score, frac: float) -> float:
    n = len(y_score)
    if n == 0:
        return float("nan")
    k = max(1, int(frac * n))
    idx = np.argpartition(y_score, -k)[-k:]
    return float(np.sum(y_true[idx] == 1) / k)

# ========= Train/Eval/Save =========
all_metrics: dict[str, dict] = {}
all_preds_rows = []

for name, clf in models.items():
    # fit with sample_weight
    if isinstance(clf, Pipeline):
        clf.fit(Xtr[feature_cols], ytr, clf__sample_weight=wtr)
    else:
        clf.fit(Xtr[feature_cols], ytr, sample_weight=wtr)

    p = clf.predict_proba(Xva[feature_cols])[:, 1]

    # weighted AUC/AP (base-rate corrected)
    auc = safe_auc(yva, p, sample_weight=wva)
    ap  = safe_ap(yva, p, sample_weight=wva)

    # ranking metrics (unweighted, for comparability)
    p005 = precision_at_frac(yva, p, 0.005)
    p01  = precision_at_frac(yva, p, 0.01)
    p05  = precision_at_frac(yva, p, 0.05)

    print(f"[{name}] AUC(w)={auc:.3f} AP(w)={ap:.3f} | P@0.5%={p005:.3f} P@1%={p01:.3f} P@5%={p05:.3f}")

    all_metrics[name] = {
        "AUC_weighted": auc,
        "AP_weighted": ap,
        "P_at_0p5pct": p005,
        "P_at_1pct": p01,
        "P_at_5pct": p05
    }

    preds_df = Xva[id_cols + feature_cols].copy()
    preds_df = preds_df.assign(y_true=yva, y_score=p, model=name)
    all_preds_rows.append(preds_df)

    joblib.dump(clf, OUT / f"closure_lp_{name}.joblib")

# Save metrics
(OUT / "closure_lp_metrics.json").write_text(
    json.dumps(all_metrics, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

# Save preds
all_preds = pd.concat(all_preds_rows, axis=0).reset_index(drop=True)
all_preds.to_csv(OUT / "closure_lp_preds.csv", index=False)

# Save RF importances
rf_model = models.get("rf")
if hasattr(rf_model, "feature_importances_"):
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)
    imp_df.to_csv(OUT / "closure_lp_importances.csv", index=False)

# Save meta
meta = {
    "cutoff": cutoff.isoformat(),
    "delta_days": DELTA_DAYS,
    "time_trim_days": TIME_TRIM_DAYS,
    "neg_cap": NEG_CAP,
    "neg_min": NEG_MIN,
    "skip_existing_ac_in_train": SKIP_EXISTING_AC_IN_TRAIN,
    "n_train_edges": int(len(E_tr)),
    "n_test_edges": int(len(E_te)),

    "n_pos_kept": int(n_pos),
    "n_neg_kept": int(n_neg_sampled),
    "neg_seen_total": int(neg_seen),
    "w_neg": float(w_neg),
    "pos_rate_sampled": float(Wsel["y"].mean()),
    "pos_rate_est_true": float(true_pos_rate_est),

    "features": feature_cols,
    "models": list(models.keys()),
    "seed": 42
}
(OUT / "closure_lp_meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("[OK] Saved:",
      OUT / "closure_lp_metrics.json",
      OUT / "closure_lp_preds.csv",
      OUT / "closure_lp_importances.csv",
      OUT / "closure_lp_logreg.joblib",
      OUT / "closure_lp_rf.joblib",
      OUT / "closure_lp_meta.json")
