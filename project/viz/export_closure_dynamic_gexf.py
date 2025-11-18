from pathlib import Path
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EXPORTS = ROOT / "data" / "exports"
EXPORTS.mkdir(parents=True, exist_ok=True)

preds = pd.read_csv(PROCESSED / "closure_lp_preds.csv", parse_dates=["t"])
MODEL = "rf"; TOPK = 5000
subset = preds[preds["model"]==MODEL].sort_values("y_score", ascending=False).head(TOPK)

G = nx.DiGraph()
for _, r in subset.iterrows():
    a, c = int(r["A"]), int(r["C"])
    start = r["t"]
    attrs = {
        "weight": float(r["y_score"]),
        "type": "predicted_closure",
        "start": pd.to_datetime(start, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    G.add_node(a); G.add_node(c)
    G.add_edge(a, c, **attrs)

out = EXPORTS / f"closure_lp_dynamic_{MODEL}_top{TOPK}.gexf"
nx.write_gexf(G, out)
print("[saved]", out)
