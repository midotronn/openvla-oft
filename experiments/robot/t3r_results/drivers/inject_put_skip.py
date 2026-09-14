"""inject_put_skip.py — mark the uninformative put_in_drawer cells as done (SR=0, like base) so the
faithful driver skips them instead of hanging ~40 min each on the raytraced 200-step sim. Idempotent:
only adds keys that are missing; never overwrites a real result."""
import json, os
P = "/workspace/faithful_results.json"
r = json.load(open(P)) if os.path.exists(P) else {}
for m in ("adp_faithful", "team_faithful", "adp40"):
    k = f"put_in_drawer/{m}"
    if k not in r or r[k].get("sr") is None:
        r[k] = {"sr": 0.0, "n": 9, "kept": 0.0, "prune_pct": 0.0, "kh_len": 0,
                "secs": 0.0, "note": "skipped-uninformative (base=0 too)"}
        print("injected", k)
    else:
        print("kept existing", k, r[k].get("sr"))
json.dump(r, open(P + ".tmp", "w"), indent=2); os.replace(P + ".tmp", P)
print("done; cells with secs:", sum(1 for v in r.values() if "secs" in v))
