# Baseline faithfulness audit (vs primary sources) — 2026-07-02

## ADP (VLA-ADP, arXiv 2509.22093, repo chen7086/VLA-ADP — REAL, full code)
Source: prune_v2_config.json + modeling_prismatic.py (_compute_qk_importance_generic, qk_keep_split).
QK IMPORTANCE (my _adp_importance vs ADP _compute_qk_importance_generic): FAITHFUL — exact match:
  input_layernorm -> q_proj(text)/k_proj(vis) -> multihead -> scores=q@kT*scale -> mean over (heads,text). layer0.
keep_ratio 0.75: match. 
qk_keep_split=[0.4,0.6]: MULTI-CAMERA ONLY (splits keep budget across camera images; only runs if num_images>1).
  SimplerEnv=single camera -> ADP falls to `else: topk(importance, keep_total)` = EXACTLY my single top-k. So
  split is IRRELEVANT single-cam; my selection is FAITHFUL for single-camera.
=> adp64 (DYNAMIC=0, keep0.25): FAITHFUL (gate off; pure QK top-64 = ADP single-cam selection). SOLID.
Dynamic gate (adp_native) APPROXIMATE, mismatches:
  - delta_method: mine SUMS per-step action-norms (arc-length); ADP uses NET endpoint eef displacement norm(p[-1]-p[0]).
  - gate INPUT: mine = commanded action magnitude; ADP = achieved sim eef pose (obs robot0_eef_pos).
  - initial_state: mine default 1; ADP config 0.
  - limit_consecutive_pruned (force keep after 3): NOT implemented.
  - extrema decision (up=max(prev), dn=min(prev)): MATCHES.
  => adp_native (78.1) is an approximation of the gate; adp64 (78.3, the fair headline) is faithful.

## TEAM-VLA (arXiv 2512.09927 "Token Expand-Merge" — paper REAL; repo Jasper-aaa/TEAM-VLA = PLACEHOLDER, no code)
"top-80" IS a real paper value: "Pruning vs Merging (Final top-80)"; ablation top-50/80/110/130
  (LIBERO-Object uses u=0.35, m=130). So my TEAM_TOPK=80 default is legit (matches paper reference), NOT fabricated.
BUT my port is a SIMPLIFICATION — NOT faithful to the 2-stage method:
  Paper = (1) token PRUNE *before* backbone: similarity-sample + context-sample(u) + spatial EXPAND (Conv kernel K),
          then (2) action-guided soft-bipartite MERGE at a MIDDLE/deeper layer, keeping top-M.
  My port = SINGLE soft-bipartite merge at the PROJECTOR (pre-backbone) by text-cosine top-K. Omits stage-1
  expand-prune, the middle-layer placement, u (context proportion), K (kernel). Directionally "merge-based
  reduction" but mechanism not exact.

## BIG CROSS-CUTTING FINDING (T3R-favorable, honest)
ALL THREE methods are natively LIBERO TWO-CAMERA (agent + wrist):
  - ADP's qk_keep_split=[0.4,0.6] is literally a two-camera keep-budget split.
  - TEAM-VLA uses wrist-view + agent-view.
  - T3R keeps the full wrist camera, prunes the 3rd-person view (~43% effective).
My SimplerEnv comparison runs ALL of them SINGLE-camera -> out-of-regime for EVERY method, not just T3R.
The baselines' native defaults (ADP 0.75 w/ cam-split, TEAM top-80) are calibrated for two cameras.
=> "single-camera 75%" is out-of-distribution for the whole field; levels the framing.
