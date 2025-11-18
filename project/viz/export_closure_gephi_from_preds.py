from pathlib import Path
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EXPORTS = ROOT / "data" / "exports"
EXPORTS.mkdir(parents=True, exist_ok=True)

preds_path = PROCESSED / "closure_lp_preds.csv"
parse_cols = [c for c in ["t1","t2","t"] if c in pd.read_csv(preds_path, nrows=0).columns]
preds = pd.read_csv(preds_path, parse_dates=parse_cols)

# Chọn model & top-K
MODEL = "rf"    # hoặc "logreg"
TOPK  = 5000    # số cạnh muốn export

subset = (preds[preds["model"]==MODEL]
          .sort_values("y_score", ascending=False)
          .head(TOPK))

# Các cột đặc trưng (tồn tại trong file preds)
feature_cols = [c for c in [
    "common_neighbors_ac","adamic_adar_ac","jaccard_ac","preferential_attachment_ac",
    "deg_a","deg_b","deg_c","clustering_b","ab_count","bc_count","gap_days"
] if c in subset.columns]

G = nx.DiGraph()

for _, r in subset.iterrows():
    a, b, c = int(r["A"]), int(r["B"]), int(r["C"])
    G.add_node(a); G.add_node(c)
    # Edge A->C đại diện cho khả năng đóng nêm
    attrs = {
        "weight": float(r["y_score"]),
        "model": str(r["model"]),
        "B": b
    }
    # Thời gian nêm (nếu có)
    for tc in ["t1","t2","t"]:
        if tc in subset.columns and pd.notna(r[tc]):
            attrs[tc] = pd.to_datetime(r[tc], utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Thêm features
    for f in feature_cols:
        v = r[f]
        attrs[f] = float(v) if pd.notna(v) else v

    G.add_edge(a, c, **attrs)

# Xuất GEXF
gexf_path = EXPORTS / f"closure_lp_{MODEL}_top{TOPK}.gexf"
nx.write_gexf(G, gexf_path)
print("[saved]", gexf_path)

# Xuất GraphML (tuỳ chọn)
graphml_path = EXPORTS / f"closure_lp_{MODEL}_top{TOPK}.graphml"
nx.write_graphml(G, graphml_path)
print("[saved]", graphml_path)
