from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGS = ROOT / "data" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

def plot_forecast(json_path: Path, title: str, out_png: Path):
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    pred = pd.DataFrame(payload["pred"])
    base = pd.DataFrame(payload["baseline"])

    pred["ym"] = pd.PeriodIndex(pred["ym"], freq="M").to_timestamp()
    base["ym"] = pd.PeriodIndex(base["ym"], freq="M").to_timestamp()

    plt.figure(figsize=(10,4.5))
    plt.plot(pred["ym"], pred["yhat"], label="SARIMAX")
    plt.plot(base["ym"], base["yhat"], label="Baseline (Last value)", linestyle="--")
    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel(payload["metric"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("[saved]", out_png)

plot_forecast(PROCESSED / "forecast_reciprocity.json",
              "Forecast — reciprocity_rate",
              FIGS / "ts_reciprocity_forecast.png")

plot_forecast(PROCESSED / "forecast_closure.json",
              "Forecast — closure_rate",
              FIGS / "ts_closure_forecast.png")
