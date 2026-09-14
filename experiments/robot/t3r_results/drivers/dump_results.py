"""dump_results.py <json_path> — print all completed cells (sr not None) sorted, for ingestion."""
import json, sys
p = sys.argv[1]
r = json.load(open(p))
for k in sorted(r):
    v = r[k]
    if v.get("sr") is not None:
        print(f"{k}\tsr={v['sr']:.4f}\tn={v.get('n')}\tkept={v.get('kept')}\tprune={v.get('prune_pct')}")
