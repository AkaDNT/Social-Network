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

NEG_CAP       = 200_000     # tối đa negatives giữ lại (reservoir)

POS_CAP = 200_000        # max positives giữ lại (None nếu muốn giữ hết)

POS_NEG_RATIO = 1.0      # giữ tối đa pos = ratio * neg (1.0 => gần 1:1)

NEG_MIN       = 5_000       # tối thiểu negatives

LOG_EVERY_B   = 1000        # log mỗi n node B

TIME_TRIM_DAYS= None        # ví dụ: 365 để chỉ sinh nêm trong 1 năm cuối của train



# ========= Paths =========

ROOT = Path(__file__).resolve().parents[1]

RAW  = ROOT / "data" / "raw" / "wiki-talk-temporal.txt"

OUT  = ROOT / "data" / "processed"

OUT.mkdir(parents=True, exist_ok=True)



# ========= Load =========

E = pd.read_csv(

    RAW, sep=r"\s+", header=None, names=["src","dst","ts"],

    dtype={"src":"int64","dst":"int64","ts":"int64"}

)

E["ts"] = pd.to_datetime(E["ts"], unit="s", utc=True)

E = E[E["src"]!=E["dst"]].sort_values("ts").reset_index(drop=True)



cutoff = E["ts"].quantile(0.8)

E_tr = E[E["ts"]<=cutoff].copy()
E_te = E[E["ts"]>cutoff].copy()

if TIME_TRIM_DAYS is not None:
    lo = cutoff - pd.Timedelta(days=TIME_TRIM_DAYS)
    E_tr = E_tr[E_tr["ts"]>=lo].copy()

# Graph train để filter (A-C đã tồn tại trong train thì không gọi là closure mới)
Gd_filter = nx.DiGraph()
Gd_filter.add_edges_from(E_tr[["src","dst"]].itertuples(index=False, name=None))
Gu_filter = Gd_filter.to_undirected()




print(f"[Load] Train edges: {len(E_tr):,} | Test edges: {len(E_te):,} | Δ={DELTA_DAYS}d")



# ========= TEST index for labeling =========

TE_map = (
    E_te.groupby(["src","dst"])["ts"]
        .apply(lambda s: np.sort(s.to_numpy(dtype="datetime64[ns]")))
        .to_dict()
)




def has_future_edge(arr: np.ndarray, lo: np.datetime64, hi: np.datetime64) -> bool:

    if arr is None or len(arr)==0:

        return False

    i = np.searchsorted(arr, lo, side="right")  # (t, ...]

    return bool(i < len(arr) and arr[i] <= hi)



# ========= Generate & sample wedges (streaming) =========

def collect_wedges_streaming(df: pd.DataFrame):

    """

    Duyệt theo B, sinh nêm A->B (t1) & B->C (t2) với t2 ∈ [t1, t1+Δ].

    - Giữ TẤT CẢ positives (y=1).

    - Negatives dùng reservoir sampling tới NEG_CAP.

    Trả về hai list nhỏ: pos_rows, neg_rows (đã sample).

    Mỗi row: (A,B,C,t1,t2,t,y)

    """

    pos_rows, neg_rows = [], []

    neg_seen = 0



    by_dst = df.groupby("dst")   # A->B

    by_src = df.groupby("src")   # B->C

    commons = sorted(set(by_dst.groups).intersection(by_src.groups))

    nB = len(commons)



    for idxB, B in enumerate(commons, 1):

        Ein  = by_dst.get_group(B)[["src","ts"]].rename(columns={"src":"A","ts":"t1"}).sort_values("t1")

        Eout = by_src.get_group(B)[["dst","ts"]].rename(columns={"dst":"C","ts":"t2"}).sort_values("t2")



        t2s = Eout["t2"].to_numpy(dtype="datetime64[ns]")

        t2s_pd = Eout["t2"].to_numpy()

        Cs  = Eout["C"].to_numpy()



        for a, t1 in zip(Ein["A"].to_numpy(), Ein["t1"]):

            lo = np.datetime64(t1.to_datetime64())

            hi = np.datetime64((t1 + DELTA).to_datetime64())



            L = np.searchsorted(t2s, lo, side="left")

            R = np.searchsorted(t2s, hi, side="right")

            if L >= R:

                continue



            for j in range(L, R):

                c = Cs[j]

                if a == c:

                    continue

                if Gu_filter.has_edge(a, c):
                    continue

                t2_pd = pd.Timestamp(t2s_pd[j])

                t_pd  = max(t1, t2_pd)

                lo_t  = np.datetime64(t_pd.to_datetime64())

                hi_t  = np.datetime64((t_pd + DELTA).to_datetime64())



                # label ngay (streaming)

                y = int(

                    has_future_edge(TE_map.get((a, c)), lo_t, hi_t) or

                    has_future_edge(TE_map.get((c, a)), lo_t, hi_t)

                )



                row = (int(a), int(B), int(c), t1, t2_pd, t_pd, y)

                if y == 1:

                    pos_rows.append(row)

                else:

                    # reservoir for negatives

                    neg_seen += 1

                    if len(neg_rows) < NEG_CAP:

                        neg_rows.append(row)

                    else:

                        r = rng.randint(0, neg_seen)

                        if r < NEG_CAP:

                            neg_rows[r] = row



        if idxB % LOG_EVERY_B == 0:

            print(f"[Wedge] processed B {idxB}/{nB} | pos:{len(pos_rows):,} | neg_reservoir:{len(neg_rows):,}")



    # đảm bảo tối thiểu negatives

    if len(neg_rows) < NEG_MIN and neg_seen > 0:

        print(f"[Info] Negatives in reservoir too small ({len(neg_rows)}). "

              f"Consider lowering TIME_TRIM_DAYS or increasing NEG_CAP.")



    return pos_rows, neg_rows



print("[Wedge] Streaming generation & sampling ...")

pos_rows, neg_rows = collect_wedges_streaming(E_tr)

print(f"[Wedge] Done. Pos kept: {len(pos_rows):,} | Neg kept (reservoir): {len(neg_rows):,}")

def downsample(rows, k: int | None):
    if k is None or len(rows) <= k:
        return rows
    idx = rng.choice(len(rows), size=k, replace=False)
    return [rows[i] for i in idx]

# 1) cap positives
pos_keep = downsample(pos_rows, POS_CAP)

# 2) enforce ratio pos <= POS_NEG_RATIO * neg
if POS_NEG_RATIO is not None:
    target = int(POS_NEG_RATIO * len(neg_rows))
    if target > 0 and len(pos_keep) > target:
        pos_keep = downsample(pos_keep, target)

if len(pos_keep) == 0 or len(neg_rows) == 0:
    raise RuntimeError("No positives or negatives after sampling—adjust caps/trim.")


print(f"[Wedge] After balancing: Pos kept: {len(pos_keep):,} | Neg kept: {len(neg_rows):,} | Pos rate: {len(pos_keep)/(len(pos_keep)+len(neg_rows)):.4f}")



# tạo DataFrame nhỏ từ hai list (an toàn RAM)

cols = ["A","B","C","t1","t2","t","y"]

Wsel = pd.DataFrame(pos_keep + neg_rows, columns=cols)

Wsel = Wsel.sample(frac=1.0, random_state=42).reset_index(drop=True)



print(f"[Wedge] Trainable wedges: {len(Wsel):,} | Pos rate: {Wsel['y'].mean():.6f}")



# ========= Graph & caches for features =========

Gd = nx.DiGraph(); Gd.add_edges_from(E_tr[["src","dst"]].itertuples(index=False, name=None))

Gu = Gd.to_undirected()

pair_counts = E_tr.groupby(["src","dst"]).size().to_dict()



# precompute clustering cho các B xuất hiện (nhanh hơn gọi nx.clustering từng dòng)

Bs = Wsel["B"].unique().tolist()

clus_b_map = nx.clustering(Gu, nodes=Bs) if len(Bs) else {}



def deg(u:int) -> int:

    return Gd.in_degree(u) + Gd.out_degree(u)



def pc(u,v):  # pair count

    return pair_counts.get((u,v), 0)



def make_features(row: pd.Series) -> pd.Series:

    a, b, c = int(row.A), int(row.B), int(row.C)

    # neighborhoods A & C

    na = set(Gu.neighbors(a)) if a in Gu else set()

    nc = set(Gu.neighbors(c)) if c in Gu else set()

    inter = na & nc

    cn = len(inter)

    aa = sum(1.0/np.log(Gu.degree(z)) for z in inter if Gu.degree(z) > 1)

    jacc = cn / max(1, len(na | nc))

    pa_ac = deg(a) * deg(c)

    clus_b = clus_b_map.get(b, 0.0)

    gap = abs((row.t2 - row.t1).days)

    return pd.Series({

        "common_neighbors_ac": cn,

        "adamic_adar_ac": aa,

        "jaccard_ac": jacc,

        "preferential_attachment_ac": pa_ac,

        "deg_a": deg(a), "deg_b": deg(b), "deg_c": deg(c),

        "clustering_b": clus_b,

        "ab_count": pc(a,b), "bc_count": pc(b,c),

        "gap_days": gap,

    })



print("[Feat] Computing features ...")

X = Wsel.apply(make_features, axis=1)

y = Wsel["y"].to_numpy()



# giữ ID/time để truy vết

id_cols = ["A","B","C","t1","t2","t"]

for k in id_cols:

    X[k] = Wsel[k].to_numpy()



feature_cols = [

    "common_neighbors_ac","adamic_adar_ac","jaccard_ac",

    "preferential_attachment_ac",

    "deg_a","deg_b","deg_c",

    "clustering_b",

    "ab_count","bc_count",

    "gap_days",

]



# ========= Split =========

Xtr, Xva, ytr, yva = train_test_split(

    X, y, test_size=0.2, random_state=42,

    stratify=y if len(np.unique(y))>1 else None

)



# ========= Models =========

models = {

    "logreg": Pipeline([

        ("scaler", StandardScaler(with_mean=False)),

        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced"))

    ]),

    "rf": RandomForestClassifier(

        n_estimators=300, random_state=42, n_jobs=-1,

        class_weight="balanced_subsample"

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
    idx = np.argpartition(y_score, -k)[-k:]   # lấy đúng k phần tử
    return float(np.sum(y_true[idx] == 1) / k)




# ========= Train/Eval/Save =========

all_metrics = {}

all_preds_rows = []



for name, clf in models.items():

    clf.fit(Xtr[feature_cols], ytr)

    p = clf.predict_proba(Xva[feature_cols])[:,1]



    auc = safe_auc(yva, p)

    ap  = safe_ap(yva, p)

    p005= precision_at_frac(yva, p, 0.005)

    p01 = precision_at_frac(yva, p, 0.01)

    p05 = precision_at_frac(yva, p, 0.05)



    print(f"[{name}] AUC={auc:.3f} AP={ap:.3f} P@0.5%={p005:.3f} P@1%={p01:.3f} P@5%={p05:.3f}")



    all_metrics[name] = {"AUC": auc, "AP": ap,

                         "P_at_0p5pct": p005, "P_at_1pct": p01, "P_at_5pct": p05}



    preds_df = Xva[id_cols + feature_cols].copy()

    preds_df = preds_df.assign(y_true=yva, y_score=p, model=name)

    all_preds_rows.append(preds_df)



    joblib.dump(clf, OUT / f"closure_lp_{name}.joblib")



# Save metrics/preds/importances/meta

(OUT / "closure_lp_metrics.json").write_text(

    json.dumps(all_metrics, ensure_ascii=False, indent=2)

)



all_preds = pd.concat(all_preds_rows, axis=0).reset_index(drop=True)

all_preds.to_csv(OUT / "closure_lp_preds.csv", index=False)



rf = models["rf"]

if hasattr(rf, "feature_importances_"):

    imp_df = pd.DataFrame({

        "feature": feature_cols,

        "importance": rf.feature_importances_

    }).sort_values("importance", ascending=False)

    imp_df.to_csv(OUT / "closure_lp_importances.csv", index=False)



meta = {
    "pos_cap": POS_CAP,

    "pos_neg_ratio": POS_NEG_RATIO,

    "cutoff": cutoff.isoformat(),

    "delta_days": DELTA_DAYS,

    "time_trim_days": TIME_TRIM_DAYS,

    "neg_cap": NEG_CAP, "neg_min": NEG_MIN,

    "n_train_edges": int(len(E_tr)), "n_test_edges": int(len(E_te)),

    "n_pos_kept": int((Wsel["y"]==1).sum()),

    "n_neg_kept": int((Wsel["y"]==0).sum()),

    "features": feature_cols, "models": list(models.keys()), "seed": 42

}

(OUT / "closure_lp_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))



print("[OK] Saved:",

      OUT / "closure_lp_metrics.json",

      OUT / "closure_lp_preds.csv",

      OUT / "closure_lp_importances.csv",

      OUT / "closure_lp_logreg.joblib",

      OUT / "closure_lp_rf.joblib",

      OUT / "closure_lp_meta.json")
