from pathlib import Path

import numpy as np
import pandas as pd

# ---------- Config ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "wiki-talk-temporal.txt"
DATA_OUT = ROOT / "data" / "processed"
DELTA_DAYS = 30
DELTA = pd.Timedelta(days=DELTA_DAYS)

# ---------- Load ----------
print("[ETL] Reading:", DATA_RAW)

# Đọc nhanh bằng engine "c" + sep=r"\s+"
E = pd.read_csv(
    DATA_RAW,
    sep=r"\s+",
    header=None,
    names=["src", "dst", "ts"],
    engine="c",
)

# Tiết kiệm RAM (pandas >= 2.0, cần pyarrow)
E = E.convert_dtypes(dtype_backend="pyarrow")

# Ép ID về int32 nếu an toàn
src_max = E["src"].max()
dst_max = E["dst"].max()
if (src_max <= np.iinfo(np.int32).max) and (dst_max <= np.iinfo(np.int32).max):
    E["src"] = E["src"].astype("int32")
    E["dst"] = E["dst"].astype("int32")

# Chuyển timestamp (epoch giây) sang datetime UTC tz-aware
E["ts"] = pd.to_datetime(E["ts"], unit="s", utc=True)

# Lọc self-loop và sort theo thời gian
E = E[E["src"] != E["dst"]].copy()
E = E.sort_values("ts").reset_index(drop=True)

# Tạo ym (month) không kèm tz để tránh cảnh báo khi to_period
E["ym"] = E["ts"].dt.tz_convert(None).dt.to_period("M").astype(str)

print("[ETL] Rows:", len(E))


# ---------- Metrics: Reciprocity (đã fix) ----------
def monthly_reciprocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính tỉ lệ hồi đáp theo tháng:
    Với mỗi cạnh a->b tại thời điểm t, nếu tồn tại b->a trong (t, t+Δ] thì tính là reciprocal.
    Tính rate = recip_edges / edges theo tháng.
    """
    Ef = df.rename(columns={"src": "a", "dst": "b", "ts": "t"})
    Er = df.rename(columns={"src": "b", "dst": "a", "ts": "t2"})

    J = Ef.merge(Er, on=["a", "b"], how="left")
    J = J[(J["t2"] > J["t"]) & (J["t2"] <= J["t"] + DELTA)]

    # Số cạnh reciprocal theo tháng (đếm theo (a,b,t) duy nhất)
    recip_count = (
        J.drop_duplicates(["a", "b", "t"])
        .assign(ym=lambda d: d["t"].dt.tz_convert(None).dt.to_period("M").astype(str))
        .groupby("ym")
        .size()
        .rename("recip_edges")
    )

    # Tổng cạnh theo tháng
    denom = (
        df.assign(ym=df["ts"].dt.tz_convert(None).dt.to_period("M").astype(str))
        .groupby("ym")
        .size()
        .rename("edges")
    )

    out = pd.concat([recip_count, denom], axis=1).fillna(0).reset_index(names="ym")
    out["reciprocity_rate"] = out["recip_edges"] / out["edges"].replace(0, np.nan)
    out["reciprocity_rate"] = out["reciprocity_rate"].fillna(0.0)
    out["delta_days"] = DELTA_DAYS
    # Giữ lại cả các cột đếm để debug
    return out[["ym", "reciprocity_rate", "recip_edges", "edges", "delta_days"]]


# ---------- Metrics: Triadic Closure (bền RAM) ----------
def monthly_triadic_closure(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Sinh nêm A->B và B->C trong cửa sổ [m_start-Δ, next_start)
    - Kiểm tra cạnh khép (A->C hoặc C->A) trong (t1, t1+Δ]
    - Đồng bộ so sánh thời gian về numpy.datetime64[ns] + np.searchsorted (ổn định kiểu)
    """
    df = df.copy()
    df["ym"] = df["ts"].dt.tz_convert(None).dt.to_period("M").astype(str)

    months = df["ym"].unique()
    out_rows = []

    for ym in months:
        # Mốc thời gian theo tháng (UTC tz-aware, nửa mở [start, next_start))
        m_start = pd.Period(ym).to_timestamp().tz_localize("UTC")
        next_start = m_start + pd.offsets.MonthBegin(1)

        # Cửa sổ dữ liệu để sinh nêm
        win_lo = m_start - DELTA
        win_hi_excl = next_start

        E_win = df[(df["ts"] >= win_lo) & (df["ts"] < win_hi_excl)][
            ["src", "dst", "ts"]
        ].copy()
        if E_win.empty:
            out_rows.append(
                {
                    "ym": ym,
                    "total_wedges": 0,
                    "closed_wedges": 0,
                    "closure_rate": 0.0,
                    "delta_days": DELTA_DAYS,
                }
            )
            continue

        # Dải tương lai để kiểm tra cạnh khép tới (next_start + Δ)
        fut_hi_excl = next_start + DELTA
        E_future = df[(df["ts"] > win_lo) & (df["ts"] < fut_hi_excl)][
            ["src", "dst", "ts"]
        ].copy()

        # Chỉ mục thời gian theo cặp (U,V) -> numpy array datetime64[ns]
        ts_map: dict[tuple[int, int], np.ndarray] = {}
        for (u, v), grp in E_future.groupby(["src", "dst"]):
            ts_map[(u, v)] = grp["ts"].to_numpy(
                dtype="datetime64[ns]"
            )  # đã sort theo ts

        # Nhóm theo B để sinh nêm
        E_win["B_in"] = E_win["dst"]
        E_win["B_out"] = E_win["src"]

        in_by_B = E_win.groupby("B_in")  # (A=src, B=dst, t1=ts)
        out_by_B = E_win.groupby("B_out")  # (B=src, C=dst, t2=ts)

        total_wedges = 0
        closed_wedges = 0

        common_B = set(in_by_B.groups.keys()).intersection(out_by_B.groups.keys())
        for B in common_B:
            Ein = in_by_B.get_group(B)[["src", "dst", "ts"]].rename(
                columns={"src": "A", "dst": "B", "ts": "t1"}
            )
            Eout = out_by_B.get_group(B)[["src", "dst", "ts"]].rename(
                columns={"src": "B", "dst": "C", "ts": "t2"}
            )

            # Mảng thời gian và đích cho B->C (numpy datetime64[ns])
            t2s: np.ndarray = Eout["t2"].to_numpy(dtype="datetime64[ns]")
            Cs: np.ndarray = Eout["C"].to_numpy()

            # Lặp từng in-edge A->B tại thời điểm t1
            for a, t1 in zip(Ein["A"].to_numpy(), Ein["t1"]):
                lo = np.datetime64(t1.to_datetime64())  # t1
                hi = np.datetime64((t1 + DELTA).to_datetime64())  # t1 + Δ

                # Chỉ lấy các B->C có t2 trong [t1, t1+Δ]
                j0 = np.searchsorted(t2s, lo, side="left")
                j1 = np.searchsorted(t2s, hi, side="right")
                if j0 >= j1:
                    continue

                k = j1 - j0
                total_wedges += k

                # Kiểm tra đóng nêm: tồn tại (A->C) hoặc (C->A) trong (t1, t1+Δ]
                for c in Cs[j0:j1]:
                    ok = False

                    arr = ts_map.get((a, c))  # A->C
                    if arr is not None and len(arr):
                        idx = np.searchsorted(arr, lo, side="right")  # > t1
                        if idx < len(arr) and arr[idx] <= hi:
                            ok = True

                    if not ok:
                        arr = ts_map.get((c, a))  # C->A
                        if arr is not None and len(arr):
                            idx = np.searchsorted(arr, lo, side="right")
                            if idx < len(arr) and arr[idx] <= hi:
                                ok = True

                    if ok:
                        closed_wedges += 1

        rate = (closed_wedges / total_wedges) if total_wedges > 0 else 0.0
        out_rows.append(
            {
                "ym": ym,
                "total_wedges": int(total_wedges),
                "closed_wedges": int(closed_wedges),
                "closure_rate": float(rate),
                "delta_days": DELTA_DAYS,
            }
        )

    out = pd.DataFrame(out_rows).sort_values("ym").reset_index(drop=True)
    return out


# ---------- Compute ----------
mr_all = monthly_reciprocity(E)
mc_all = monthly_triadic_closure(E)

metrics = pd.merge(mr_all, mc_all, on=["ym", "delta_days"], how="outer")
metrics = metrics.sort_values("ym").reset_index(drop=True)

# ---------- Save ----------
DATA_OUT.mkdir(parents=True, exist_ok=True)
out_csv = DATA_OUT / "metrics_monthly.csv"
metrics.to_csv(out_csv, index=False)
print("[ETL] Saved:", out_csv)

# ---------- Debug: in ra mốc baseline theo tách 80/20 ----------
months = sorted(metrics["ym"].unique())
cut_idx = int(0.8 * len(months))
months_tr = months[:cut_idx]
if months_tr:
    last_train = months_tr[-1]
    row = metrics.set_index("ym").loc[last_train]
    # Nếu trùng ym nhiều dòng (hiếm), lấy mean để in
    if isinstance(row, pd.DataFrame):
        row = row.mean(numeric_only=True)
    print(f"[DEBUG] Last train month: {last_train}")
    print(f"[DEBUG] closure_rate at last train = {row.get('closure_rate', np.nan)}")
    print(
        f"[DEBUG] reciprocity_rate at last train = {row.get('reciprocity_rate', np.nan)}"
    )

# --- Topic stats (optional). Không có nhãn, đặt topic='All'
metrics_topic = metrics.copy()
metrics_topic["topic"] = "All"
metrics_topic = (
    metrics_topic.groupby(["topic"])
    .agg(
        reciprocity_rate=("reciprocity_rate", "mean"),
        closure_rate=("closure_rate", "mean"),
        messages=("edges", "sum"),
    )
    .reset_index()
)
metrics_topic.to_csv(DATA_OUT / "topic_stats.csv", index=False)
print("[ETL] Saved:", DATA_OUT / "topic_stats.csv")
