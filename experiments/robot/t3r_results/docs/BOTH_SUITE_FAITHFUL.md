# 4-Method × Both-Suite Faithful Comparison (final)

Four methods, both SimplerEnv suites, fp32, zero-shot CogACT-Base. All baselines run in their
**faithful** configs (faithful ADP dynamic gate; faithful two-stage TeamVLA). The clean apples-to-apples
is the **matched 154-token budget**: `adp40` (ADP QK-importance, gate off, keep 0.60) vs `t3r_orig@40%`.

Methods:
- **t3r_orig @40%** — SigLIP-SAM prune + IG attention bias, keep 0.60 → 154 tok (40% prune).
- **adp40** — ADP layer-0 QK-importance selection, gate OFF, keep 0.60 → 154 tok (40% prune). MATCHED to t3r.
- **fastv** — FastV (arXiv 2403.06764, official pkunlp-icler algo), K=1, keep 154 → 154 tok (40% prune).
  MATCHED to t3r. Prunes image tokens by the last/cognition token's layer-0 attention. MATCHED to t3r.
- **adp_faithful** — ADP with the faithful action-aware DYNAMIC gate (net-displacement rule), keep 0.75
  → ~224 tok (12% prune). Native operating point (barely prunes; ~base).
- **team_faithful** — faithful two-stage TeamVLA (sim-seed → spatial expand → context prune, then
  soft-bipartite merge), top-80 → 80 tok (69% prune).

## GOOGLE ROBOT — overall SR (10 informative eval-units, excl put_in_drawer = 0 for all incl base)

| method | tokens | prune% | coke(3) | move_near | drawer(6) | **overall** |
|---|---|---|---|---|---|---|
| base | 256 | 0% | 82.7 | 68.3 | 81.5 | **80.5** |
| **adp40 (matched 40%)** | 154 | 40% | 85.3 | 60.0 | 83.3 | **81.6** |
| **fastv (matched 40%)** | 154 | 40% | 77.3 | 63.3 | 85.2 | **80.6** |
| team_faithful | 80 | 69% | 77.3 | 61.7 | 81.5 | **78.3** |
| adp_faithful (native) | ~224 | 12% | 78.7 | 66.7 | 75.9 | **75.8** |
| **t3r_orig @40%** | 154 | 40% | 73.3 | 58.3 | 75.9 | **73.4** |

## BRIDGE / WidowX — informative-3 SR (excl stack_cube ≈ 0 for all)

| method | tokens | prune% | carrot | spoon | eggplant | **info-3** |
|---|---|---|---|---|---|---|
| adp_faithful (native) | ~224 | 12% | 20.8 | 16.7 | 25.0 | **20.8** |
| **fastv (matched 40%)** | 154 | 40% | 25.0 | 29.2 | 8.3 | **20.8** |
| **adp40 (matched 40%)** | 154 | 40% | 20.8 | 20.8 | 16.7 | **19.4** |
| team_faithful | 80 | 69% | 33.3 | 12.5 | 4.2 | **16.7** |
| base | 256 | 0% | 12.5 | 12.5 | 16.7 | **13.9** |
| **t3r_orig @40%** | 154 | 40% | 4.2 | 20.8 | 12.5 | **12.5** |

## Headline (matched 154-token budget: t3r_orig@40% vs adp40 vs fastv)
- Google Robot: **adp40 81.6 ≈ fastv 80.6  >>  t3r_orig 73.4** (+7-8 for both rivals).
- Bridge info-3: **fastv 20.8 > adp40 19.4  >>  t3r_orig 12.5** (+7-8 for both rivals).

At the SAME token count, BOTH ADP's QK-importance pruning AND FastV's attention pruning beat T3R's
external SigLIP-SAM pruning on BOTH embodiments. `t3r_orig@40%` is the **lowest-scoring of all five
pruning baselines** on both suites. The faithfulness fixes made ADP/TeamVLA slightly STRONGER (as
expected), and FastV — a widely-cited standard baseline — independently confirms the same ordering.

## Caveats
- Single-run SR. CIs: coke 25-ep (±0.10), move_near 60-ep (±0.06), drawer 9-ep (±0.15), Bridge 24-ep (±0.10).
- Bridge base is near-floor (CogACT is weak zero-shot on WidowX); all methods sit 12-21%, so absolute
  Bridge differences have low statistical power — but t3r is consistently the lowest pruning method.
- `adp_faithful` native prunes only ~12% (dynamic gate keeps ~all tokens in fine-motion phases) — it is
  NOT an efficiency-matched comparison; `adp40` is the matched-budget rival.
- put_in_drawer (Google) and stack_cube (Bridge) excluded: ≈0 SR for every method including base.

Data: SQL tables `native_faithful` (Google, 30 cells), `bridge_faithful` (Bridge, 12 cells),
`fastv` (Google 10 + Bridge 4), `k60`/`bridge_k60` (t3r_orig@40%), `bench`/`bridge` (base). JSON:
faithful_results.json, bridge_faithful_results.json, fastv_google_results.json, fastv_bridge_results.json.
FastV port: fastv_cogact.py (official pkunlp-icler algorithm, token-masking variant, transformers 4.40.1).
