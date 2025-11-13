from pathlib import Path
import pandas as pd
import networkx as nx
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EXPORTS = ROOT / "data" / "exports"
EXPORTS.mkdir(parents=True, exist_ok=True)

preds = pd.read_csv(PROCESSED / "reciprocity_lp_preds.csv")

# Chọn model tốt nhất theo P@1% bạn đang dùng, ví dụ "rf"
MODEL = "rf"
TOPK = 5000  # số cạnh đề xuất muốn xuất (lọc theo score)
subset = preds[preds["model"]==MODEL].sort_values("y_score", ascending=False).head(TOPK)

# Tạo graph có trọng số & thuộc tính
G = nx.DiGraph()
for _, r in subset.iterrows():
    u, v = int(r["src"]), int(r["dst"])
    G.add_node(u); G.add_node(v)
    G.add_edge(u, v,
               weight=float(r["y_score"]),
               model=str(r["model"]),
               common_neighbors=float(r["common_neighbors"]),
               adamic_adar=float(r["adamic_adar"]),
               preferential_attachment=float(r["preferential_attachment"]),
               ab_count=int(r["ab_count"]),
               recency_days=float(r["recency_days"]),
               deg_a=float(r["deg_a"]),
               deg_b=float(r["deg_b"])
               )

# Xuất GEXF (đọc tốt bởi Gephi)
gexf_path = EXPORTS / f"reciprocity_lp_{MODEL}_top{TOPK}.gexf"
nx.write_gexf(G, gexf_path)  # GEXF 1.2draft
print("[saved]", gexf_path)

# Nếu muốn GraphML:
graphml_path = EXPORTS / f"reciprocity_lp_{MODEL}_top{TOPK}.graphml"
nx.write_graphml(G, graphml_path)
print("[saved]", graphml_path)
