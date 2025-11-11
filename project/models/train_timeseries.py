import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "metrics_monthly.csv"
OUT_R = ROOT / "data" / "processed" / "forecast_reciprocity.json"
OUT_C = ROOT / "data" / "processed" / "forecast_closure.json"

print("[TS] Reading:", DATA)
M = pd.read_csv(DATA)
M["ym"] = pd.PeriodIndex(M["ym"], freq="M").astype(str)
M = M.sort_values("ym")

# Tạo index tháng đầy đủ
months = sorted(M["ym"].unique())
cut_idx = int(0.8 * len(months))
months_tr = months[:cut_idx]
months_te = months[cut_idx:]

print(
    f"[TS] Train months: {months_tr[0]}..{months_tr[-1]}  | Test: {months_te[0]}..{months_te[-1]}"
)


def fit_forecast(col: str, out_path: Path):
    y_tr = (
        M[M["ym"].isin(months_tr)].set_index("ym")[col].astype(float).reindex(months_tr)
    )
    y_te = (
        M[M["ym"].isin(months_te)].set_index("ym")[col].astype(float).reindex(months_te)
    )

    # Baseline naive
    yhat_naive = pd.Series(y_tr.iloc[-1], index=y_te.index)

    # SARIMAX: thông số khởi điểm hợp lý cho dữ liệu tháng
    model = SARIMAX(y_tr, order=(1, 0, 0), seasonal_order=(0, 1, 1, 12), trend="n")
    fit = model.fit(disp=False)
    yhat = pd.Series(fit.forecast(steps=len(y_te)).values, index=y_te.index)

    mae = mean_absolute_error(y_te, yhat)
    mape = mean_absolute_percentage_error(y_te.replace(0, 1e-9), yhat)
    mae_b = mean_absolute_error(y_te, yhat_naive)
    mape_b = mean_absolute_percentage_error(y_te.replace(0, 1e-9), yhat_naive)
    print(
        f"[{col}] SARIMAX  MAE={mae:.4f} MAPE={mape:.3f} | Naive MAE={mae_b:.4f} MAPE={mape_b:.3f}"
    )

    payload = {
        "train_range": [months_tr[0], months_tr[-1]],
        "test_range": [months_te[0], months_te[-1]],
        "metric": col,
        "pred": [{"ym": k, "yhat": float(v)} for k, v in yhat.items()],
        "baseline": [{"ym": k, "yhat": float(v)} for k, v in yhat_naive.items()],
        "mae": float(mae),
        "mape": float(mape),
        "mae_naive": float(mae_b),
        "mape_naive": float(mape_b),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print("[TS] Saved:", out_path)


fit_forecast("reciprocity_rate", OUT_R)
fit_forecast("closure_rate", OUT_C)
