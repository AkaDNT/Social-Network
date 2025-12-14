from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import os


from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score,
    confusion_matrix
)
from sklearn.calibration import calibration_curve
import plotly.graph_objects as go

# =========================
# Path discovery (không cần đổi cấu trúc thư mục)
# =========================
def guess_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "data" / "processed").exists():
            return p
    # fallback: thư mục chứa file dashboard
    return start.parent

def default_paths() -> dict[str, Path]:
    env_proc = os.getenv("SN_PROCESSED_DIR")
    env_root = os.getenv("SN_ROOT")

    if env_root:
        root = Path(env_root)
    else:
        root = guess_root(Path(__file__))

    processed = Path(env_proc) if env_proc else (root / "data" / "processed")
    exports = root / "data" / "exports"
    return {"root": root, "processed": processed, "exports": exports}


# =========================
# Loaders
# =========================
@st.cache_data
def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

@st.cache_data
def load_reciprocity_preds(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # last_ab có thể là string iso; parse cho đẹp
    if "last_ab" in df.columns:
        df["last_ab"] = pd.to_datetime(df["last_ab"], errors="coerce", utc=True)
    return df

@st.cache_data
def load_closure_preds(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    for c in ["t1", "t2", "t"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df

@st.cache_data
def load_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)

# =========================
# Metrics utilities
# =========================
def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))

def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))

def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    n = len(y_score)
    if n == 0:
        return float("nan")
    k = max(1, min(k, n))
    idx = np.argpartition(y_score, -k)[-k:]
    return float(y_true[idx].sum() / k)

def threshold_for_topk(y_score: np.ndarray, k: int) -> float:
    n = len(y_score)
    k = max(1, min(k, n))
    thr = float(np.partition(y_score, -k)[-k])
    return thr

def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> dict:
    y_pred = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr  = fp / (fp + tn) if (fp + tn) else float("nan")
    tpr  = rec
    acc  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan")
    return {
        "threshold": float(thr),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": float(prec), "recall": float(rec),
        "fpr": float(fpr), "tpr": float(tpr),
        "accuracy": float(acc),
        "predicted_positive": int((y_pred==1).sum())
    }

# =========================
# Plotly charts
# =========================
def plot_roc(y_true: np.ndarray, y_score: np.ndarray) -> go.Figure:
    fig = go.Figure()
    if len(np.unique(y_true)) < 2:
        fig.add_annotation(text="ROC cần cả 2 class (0/1).", showarrow=False)
        return fig
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash")))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=360, margin=dict(l=10,r=10,t=30,b=10))
    return fig

def plot_pr(y_true: np.ndarray, y_score: np.ndarray) -> go.Figure:
    fig = go.Figure()
    if y_true.sum() == 0:
        fig.add_annotation(text="PR cần có positive.", showarrow=False)
        return fig
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR"))
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=360, margin=dict(l=10,r=10,t=30,b=10))
    return fig

def plot_score_hist(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "y_true" in df.columns:
        for label in [0, 1]:
            d = df[df["y_true"] == label]["y_score"].to_numpy()
            fig.add_trace(go.Histogram(x=d, name=f"y={label}", opacity=0.6))
        fig.update_layout(barmode="overlay")
    else:
        fig.add_trace(go.Histogram(x=df["y_score"].to_numpy(), name="score"))
    fig.update_layout(xaxis_title="y_score", yaxis_title="count", height=280, margin=dict(l=10,r=10,t=30,b=10))
    return fig

def plot_calibration(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> go.Figure:
    fig = go.Figure()
    if y_true.sum() == 0 or len(np.unique(y_true)) < 2:
        fig.add_annotation(text="Calibration cần có đủ class & positives.", showarrow=False)
        return fig
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=bins, strategy="uniform")
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Calibration"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Perfect", line=dict(dash="dash")))
    fig.update_layout(xaxis_title="Mean predicted prob", yaxis_title="Empirical positive rate", height=360, margin=dict(l=10,r=10,t=30,b=10))
    return fig

# =========================
# Simple “explain row” (percentile-based)
# =========================
def row_explain_percentiles(df: pd.DataFrame, row: pd.Series, feature_cols: list[str], topn: int = 6) -> pd.DataFrame:
    out = []
    for f in feature_cols:
        if f not in df.columns:
            continue
        col = df[f].to_numpy()
        val = row[f]
        # percentile rank (robust)
        pct = float(np.mean(col <= val)) if len(col) else float("nan")
        out.append((f, float(val), pct))
    out_df = pd.DataFrame(out, columns=["feature", "value", "percentile"])
    out_df["salience"] = np.abs(out_df["percentile"] - 0.5)
    return out_df.sort_values("salience", ascending=False).head(topn)

# =========================
# App UI
# =========================
st.set_page_config(page_title="Reciprocity & Closure Dashboard", layout="wide")

paths = default_paths()
st.sidebar.header("Paths")
proc_dir = st.sidebar.text_input("Processed dir", str(paths["processed"]))
exp_dir  = st.sidebar.text_input("Exports dir (optional)", str(paths["exports"]))

PROC = Path(proc_dir)
EXP  = Path(exp_dir)

# validate
need_files = [
    "reciprocity_lp_metrics.json", "reciprocity_lp_preds.csv", "reciprocity_lp_meta.json",
    "closure_lp_metrics.json", "closure_lp_preds.csv", "closure_lp_meta.json",
]
missing = [f for f in need_files if not (PROC / f).exists()]
if missing:
    st.error("Thiếu file trong processed dir:\n- " + "\n- ".join(missing))
    st.stop()

# load core
r_meta = load_json(PROC / "reciprocity_lp_meta.json")
c_meta = load_json(PROC / "closure_lp_meta.json")

r_metrics = load_json(PROC / "reciprocity_lp_metrics.json")
c_metrics = load_json(PROC / "closure_lp_metrics.json")

r_preds = load_reciprocity_preds(PROC / "reciprocity_lp_preds.csv")
c_preds = load_closure_preds(PROC / "closure_lp_preds.csv")

# feature columns from meta (đúng theo train script)
r_feats = list(r_meta.get("features", []))
c_feats = list(c_meta.get("features", []))

# navigation
page = st.sidebar.radio("Page", ["Overview", "Reciprocity", "Closure", "Downloads"])

st.title("Social Network Dashboard — Reciprocity & Closure")

# =========================
# Overview
# =========================
if page == "Overview":
    st.subheader("Run summary (from meta.json)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Reciprocity")
        st.write({
            "cutoff": r_meta.get("cutoff"),
            "n_candidates": r_meta.get("n_candidates"),
            "n_positive": r_meta.get("n_positive"),
            "positive_rate": r_meta.get("positive_rate"),
            "models": r_meta.get("models"),
        })
        mdf = pd.DataFrame(r_metrics).T.reset_index().rename(columns={"index":"model"})
        st.markdown("**Model metrics (val split)**")
        st.dataframe(mdf, use_container_width=True)

    with col2:
        st.markdown("### Closure")
        st.write({
            "cutoff": c_meta.get("cutoff"),
            "delta_days": c_meta.get("delta_days"),
            "time_trim_days": c_meta.get("time_trim_days"),
            "neg_cap": c_meta.get("neg_cap"),
            "n_train_edges": c_meta.get("n_train_edges"),
            "n_test_edges": c_meta.get("n_test_edges"),
            "n_pos_kept": c_meta.get("n_pos_kept"),
            "n_neg_kept": c_meta.get("n_neg_kept"),
            "models": c_meta.get("models"),
        })
        mdf = pd.DataFrame(c_metrics).T.reset_index().rename(columns={"index":"model"})
        st.markdown("**Model metrics (val split)**")
        st.dataframe(mdf, use_container_width=True)

# =========================
# Reciprocity page
# =========================
elif page == "Reciprocity":
    st.subheader("Reciprocity — Interactive evaluation & explorer")

    model = st.selectbox("Model", sorted(r_preds["model"].unique()))
    df = r_preds[r_preds["model"] == model].copy()

    y_true = df["y_true"].to_numpy(dtype=int)
    y_score = df["y_score"].to_numpy(dtype=float)

    # headline metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC", f"{safe_auc(y_true, y_score):.4f}")
    c2.metric("AP", f"{safe_ap(y_true, y_score):.4f}")
    c3.metric("P@1%", f"{precision_at_k(y_true, y_score, max(1, int(0.01*len(df)))):.4f}")
    c4.metric("Pos rate (val)", f"{float(y_true.mean()):.6f}")

    # charts
    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_roc(y_true, y_score), use_container_width=True)
        st.plotly_chart(plot_calibration(y_true, y_score, bins=10), use_container_width=True)
    with right:
        st.plotly_chart(plot_pr(y_true, y_score), use_container_width=True)
        st.plotly_chart(plot_score_hist(df), use_container_width=True)

    st.markdown("---")
    st.markdown("### Threshold / Top-K analysis")

    mode = st.radio("Chọn cách đặt ngưỡng", ["By threshold", "By top-K"], horizontal=True)
    if mode == "By threshold":
        thr = st.slider("Threshold", 0.0, 1.0, float(np.quantile(y_score, 0.99)) if len(y_score) else 0.5, 0.001)
    else:
        k = st.slider("Top K (rows)", 50, min(20000, len(df)), min(2000, len(df)), 50)
        thr = threshold_for_topk(y_score, k)
        st.caption(f"Threshold tương ứng để lấy đúng ~{k} dự đoán cao nhất: **{thr:.6f}**")

    m = metrics_at_threshold(y_true, y_score, thr)
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Predicted positive", m["predicted_positive"])
    mc2.metric("Precision", f"{m['precision']:.4f}")
    mc3.metric("Recall", f"{m['recall']:.4f}")
    mc4.metric("FPR", f"{m['fpr']:.4f}")

    st.write({"tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"]})

    st.markdown("---")
    st.markdown("### Top predictions table (predicted reply is dst → src)")
    topn = st.slider("Show top N", 50, min(20000, len(df)), 2000, 50)
    top_df = df.sort_values("y_score", ascending=False).head(topn).copy()
    # thêm cột “predicted_edge”
    top_df["predicted_edge"] = top_df["dst"].astype(str) + " → " + top_df["src"].astype(str)
    show_cols = ["predicted_edge", "y_score", "y_true", "src", "dst", "last_ab"] + [c for c in r_feats if c in top_df.columns]
    show_cols = list(dict.fromkeys(show_cols))
    st.dataframe(top_df[show_cols], use_container_width=True, height=420)

    st.markdown("---")
    st.markdown("### Node / pair explorer")
    qcol1, qcol2 = st.columns(2)

    with qcol1:
        node = st.text_input("Node id (ví dụ 123)", "")
        if node.strip().isdigit():
            node_id = int(node)
            sub = df[(df["src"] == node_id) | (df["dst"] == node_id)].copy()
            st.caption(f"Rows liên quan node {node_id}: {len(sub)}")
            if len(sub):
                sub = sub.sort_values("y_score", ascending=False).head(2000)
                sub["predicted_edge"] = sub["dst"].astype(str) + " → " + sub["src"].astype(str)
                st.dataframe(sub[["predicted_edge", "y_score", "y_true", "src", "dst"]], use_container_width=True, height=300)

    with qcol2:
        st.caption("Giải thích 1 dòng (percentile so với toàn bộ val của model)")
        pick_i = st.number_input("Row index (0..N-1 theo bảng top)", min_value=0, max_value=max(0, len(top_df)-1), value=0)
        if len(top_df):
            row = top_df.iloc[int(pick_i)]
            exp = row_explain_percentiles(df, row, r_feats, topn=7)
            st.write(f"Selected predicted edge: **{int(row.dst)} → {int(row.src)}** | score={row.y_score:.6f} | y_true={int(row.y_true)}")
            st.dataframe(exp, use_container_width=True)

# =========================
# Closure page
# =========================
elif page == "Closure":
    st.subheader("Closure — Interactive evaluation & explorer")

    model = st.selectbox("Model", sorted(c_preds["model"].unique()))
    df = c_preds[c_preds["model"] == model].copy()

    y_true = df["y_true"].to_numpy(dtype=int)
    y_score = df["y_score"].to_numpy(dtype=float)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC", f"{safe_auc(y_true, y_score):.4f}")
    c2.metric("AP", f"{safe_ap(y_true, y_score):.4f}")
    c3.metric("P@1%", f"{precision_at_k(y_true, y_score, max(1, int(0.01*len(df)))):.4f}")
    c4.metric("Pos rate (val)", f"{float(y_true.mean()):.6f}")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_roc(y_true, y_score), use_container_width=True)
        st.plotly_chart(plot_calibration(y_true, y_score, bins=10), use_container_width=True)
    with right:
        st.plotly_chart(plot_pr(y_true, y_score), use_container_width=True)
        st.plotly_chart(plot_score_hist(df), use_container_width=True)

    st.markdown("---")
    st.markdown("### Threshold / Top-K analysis")

    mode = st.radio("Chọn cách đặt ngưỡng", ["By threshold", "By top-K"], horizontal=True)
    if mode == "By threshold":
        thr = st.slider("Threshold", 0.0, 1.0, float(np.quantile(y_score, 0.99)) if len(y_score) else 0.5, 0.001)
    else:
        k = st.slider("Top K (rows)", 50, min(20000, len(df)), min(2000, len(df)), 50)
        thr = threshold_for_topk(y_score, k)
        st.caption(f"Threshold tương ứng để lấy đúng ~{k} dự đoán cao nhất: **{thr:.6f}**")

    m = metrics_at_threshold(y_true, y_score, thr)
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Predicted positive", m["predicted_positive"])
    mc2.metric("Precision", f"{m['precision']:.4f}")
    mc3.metric("Recall", f"{m['recall']:.4f}")
    mc4.metric("FPR", f"{m['fpr']:.4f}")
    st.write({"tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"]})

    st.markdown("---")
    st.markdown("### Top closures table (predicted new tie is A ↔ C)")
    topn = st.slider("Show top N", 50, min(20000, len(df)), 2000, 50)
    top_df = df.sort_values("y_score", ascending=False).head(topn).copy()
    top_df["predicted_tie"] = top_df["A"].astype(str) + " ↔ " + top_df["C"].astype(str)

    base_cols = ["predicted_tie", "y_score", "y_true", "A", "B", "C", "t", "gap_days"]
    show_cols = [c for c in base_cols if c in top_df.columns] + [c for c in c_feats if c in top_df.columns]
    show_cols = list(dict.fromkeys(show_cols))  # <-- FIX
    st.dataframe(top_df[show_cols], use_container_width=True, height=420)


    st.markdown("---")
    st.markdown("### Node / triad explorer")
    qcol1, qcol2 = st.columns(2)

    with qcol1:
        node = st.text_input("Node id (A/B/C)", "")
        if node.strip().isdigit():
            nid = int(node)
            sub = df[(df["A"] == nid) | (df["B"] == nid) | (df["C"] == nid)].copy()
            st.caption(f"Rows liên quan node {nid}: {len(sub)}")
            if len(sub):
                sub = sub.sort_values("y_score", ascending=False).head(2000)
                sub["predicted_tie"] = sub["A"].astype(str) + " ↔ " + sub["C"].astype(str)
                st.dataframe(sub[["predicted_tie", "y_score", "y_true", "A", "B", "C", "t", "gap_days"]], use_container_width=True, height=300)

    with qcol2:
        st.caption("Giải thích 1 triad (percentile-based)")
        pick_i = st.number_input("Row index (0..N-1 theo bảng top)", min_value=0, max_value=max(0, len(top_df)-1), value=0)
        if len(top_df):
            row = top_df.iloc[int(pick_i)]
            exp = row_explain_percentiles(df, row, c_feats, topn=7)
            st.write(
                f"Selected triad: **A={int(row.A)} , B={int(row.B)} , C={int(row.C)}** | "
                f"pred tie A↔C | score={row.y_score:.6f} | y_true={int(row.y_true)}"
            )
            st.dataframe(exp, use_container_width=True)

# =========================
# Downloads
# =========================
else:
    st.subheader("Downloads (for Gephi / reporting)")

    st.markdown("### Processed artifacts")
    for fname in [
        "reciprocity_lp_metrics.json", "reciprocity_lp_preds.csv", "reciprocity_lp_meta.json",
        "reciprocity_lp_importances.csv",
        "closure_lp_metrics.json", "closure_lp_preds.csv", "closure_lp_meta.json",
        "closure_lp_importances.csv",
    ]:
        p = PROC / fname
        if p.exists():
            st.download_button(
                label=f"Download {fname}",
                data=p.read_bytes(),
                file_name=fname,
                mime="application/octet-stream"
            )

    st.markdown("---")
    st.markdown("### Exports (optional)")
    if EXP.exists():
        exports = sorted(EXP.glob("*.gexf")) + sorted(EXP.glob("*.graphml"))
        if exports:
            for p in exports:
                st.download_button(
                    label=f"Download {p.name}",
                    data=p.read_bytes(),
                    file_name=p.name,
                    mime="application/octet-stream"
                )
        else:
            st.info("Không thấy *.gexf/*.graphml trong exports dir.")
    else:
        st.info("Exports dir không tồn tại — nếu bạn không dùng export thì bỏ qua.")
