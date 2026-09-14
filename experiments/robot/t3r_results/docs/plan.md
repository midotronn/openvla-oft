
## FAITHFUL ADP + FAITHFUL TWO-STAGE TeamVLA (user: `do both fixes for accurate numbers`) [RUNNING]
Both fixes implemented in baselines_cogact.py + validated:
- Fix 1 ADP faithful gate: _MotionGate rewritten to NET vector-sum displacement ||Sum(pos-deltas)||
  (not arc-length sum-of-norms), initial=0, hysteresis band, max_consec_prune=3. Smoke: gate toggles
  PRUNE/KEEP healthily (was static before).
- Fix 2 TeamVLA two-stage (TEAM_TWOSTAGE=1): stage-1 sim-seed -> KxK Conv spatial EXPAND -> +u-context
  hard PRUNE; stage-2 soft-bipartite MERGE survivors -> <=80 anchors. Bug found+fixed: CogACT calls the
  projector TWICE per step (256 raw -> merge to 80, then RE-ENTRANT 80). Old _team_merge silently
  tolerated (no grid reshape); new spatial Conv crashed on non-square 80. GUARD added: skip reduction
  unless input is the raw 16x16=256 grid (return unchanged) -> no double-reduce, no crash. Smoke SR=0.92.
- kept_hist accounting confirmed in driver (kh_len=2000), smoke's empty read was a script artifact.

FIRST FAITHFUL RESULT (Google Robot, fp32, run_faithful_driver.py -> faithful_results.json):
| unit | method | SR | kept | prune% |
| coke_upright | adp_faithful | 0.880 | 227 | 11% |
=> Faithful ADP gate prunes LESS than old adp_native (227/11% vs 192/25%): the faithful net-displacement
gate spends more time in KEEP (256-tok fine phases). SR 0.88 ties base. CONFIRMS: faithful baselines are
STRONGER, not weaker -- accurate but does NOT help t3r. Full 11-unit x 2-method run in progress (new

ative_faithful SQL table planned; preserves old 
ative data). Auto-restart loop faithful_loop.sh + keepalive.

## 4-METHOD x BOTH-SUITE COMPARISON (user request) — GOOGLE ROBOT DONE, BRIDGE RUNNING
User wants: t3r_orig@40%, teamVLA_faithful, adp_faithful (native), adp_faithful forced@40% (adp40).
adp40 = ADP QK-importance selection, gate OFF, ADP_KEEP=0.60 -> 154 tok (MATCHED to t3r_orig@40%).
Data in SQL native_faithful (Google) + bridge_faithful (pending). faithful_results.json (33 cells).

GOOGLE ROBOT (fp32, 10 informative units excl put_in_drawer=0-for-all):
| method | tok | prune% | coke | move_near | drawer6 | overall10u |
| base | 256 | 0% | 82.7 | 68.3 | 81.5 | 80.5 |
| adp40 (forced 40%) | 154 | 40% | 85.3 | 60.0 | 83.3 | 81.6 |
| team_faithful (2-stage) | 80 | 69% | 77.3 | 61.7 | 81.5 | 78.3 |
| adp_faithful (dyn gate) | ~224 | 12% | 78.7 | 66.7 | 75.9 | 75.8 |
| t3r_orig @40% | 154 | 40% | 73.3 | 58.3 | 75.9 | 73.4 |
KEY (matched 154-tok budget): adp40 81.6 >> t3r_orig@40% 73.4 (+8.2). Faithful ADP QK-pruning beats
T3R SigLIP-SAM pruning at the SAME token count. team_faithful (2-stage merge) 78.3 also > t3r@40%.
adp_faithful native barely prunes (224 tok/12%) -> ~base. NOTE: single-run, drawer n=9 (+/-.15 CI),
coke/move_near 25-60ep; adp40 81.6 ~ base 80.5 within noise. Faithful fixes did NOT weaken baselines
(as predicted) -> does NOT help t3r's relative standing. put_in_drawer skipped (raytraced hang; 0 for all
incl base; excluded from all reported metrics -- consistent with prior benchmarks).
INFRA: fp32 model ~30GB fits 44GB; a stale-zombie python caused an OOM pileup -> hardened
run_all_faithful.sh to pkill stale driver before each (re)launch. Orchestration = run_all_faithful.sh
(Google 33 then Bridge 12), detached + passive keepalive faithful_monitor3.sh.

## 4-METHOD BRIDGE (faithful) — DONE. Both suites now complete. See files/BOTH_SUITE_FAITHFUL.md
BRIDGE informative-3 SR (excl stack_cube~0; SQL bridge_faithful):
| method | tok | prune% | carrot | spoon | eggplant | info-3 |
| adp_faithful (native) | ~224 | 12% | 20.8 | 16.7 | 25.0 | 20.8 |
| adp40 (matched 40%) | 154 | 40% | 20.8 | 20.8 | 16.7 | 19.4 |
| team_faithful | 80 | 69% | 33.3 | 12.5 | 4.2 | 16.7 |
| base | 256 | 0% | 12.5 | 12.5 | 16.7 | 13.9 |
| t3r_orig @40% | 154 | 40% | 4.2 | 20.8 | 12.5 | 12.5 |
HEADLINE (matched 154-tok, adp40 vs t3r@40%): Google 81.6 vs 73.4 (+8.2); Bridge 19.4 vs 12.5 (+6.9).
At matched budget ADP QK-pruning > T3R SigLIP-SAM pruning on BOTH suites; t3r_orig@40% is LOWEST of the
four on both. Faithful fixes made baselines slightly stronger (as predicted) -> do NOT help t3r.
Bridge near-floor (low power). put_in_drawer (raytraced) skipped=0 like base. DONE. GPU freed.

## FastV BASELINE ADDED (user request) — RUNNING
Official FastV (arXiv 2403.06764, pkunlp-icler) ported to CogACT in fastv_cogact.py. Algorithm: run K
LLM layers, at layer K rank image tokens by the LAST token's attention (avg over heads) from layer K-1,
keep top-ATTENTION_RANK, prune rest for deeper layers. Adapted to transformers 4.40.1 LlamaModel.
Used the TOKEN-MASKING variant (accuracy-identical to inplace-drop since the cognition=last token attends
to the same kept set, and CogACT reads cognition from the last token; keeps seq length constant so
hidden_states structure matches base exactly). Config: K=1, keep=154 (40% prune) = MATCHED to t3r_orig@40%
& adp40 (user: 'k=1 and r=0.85 two-camera == r=40 single-camera'). Layout confirmed at smoke: S=276 =
1 BOS + 256 vision + 19 text -> SYS_LENGTH=1, IMG=256. SMOKE coke_upright SR=0.88 (ties base 0.88, beats
t3r@40% 0.73) keep=154/40% verified. Full suite: run_fastv_google.py (10 units, put_in_drawer excluded)
+ run_fastv_bridge.py (4 tasks). Orchestrated run_fastv_all.sh detached + fastv_monitor.sh keepalive.
Results -> fastv_google_results.json / fastv_bridge_results.json (SQL tables TBD).

## FastV DONE — both suites. SQL table astv. See BOTH_SUITE_FAITHFUL.md (now 5 methods).
FastV (K=1, keep=154, 40% prune, token-masking, fp32):
GOOGLE overall10u=80.6 (coke 77.3, move_near 63.3, drawer6 85.2).
BRIDGE info-3=20.8 (carrot 25.0, spoon 29.2, eggplant 8.3; stack 0).
MATCHED 154-tok budget (t3r@40% vs adp40 vs fastv):
  Google: adp40 81.6 ~ fastv 80.6 >> t3r 73.4
  Bridge info-3: fastv 20.8 > adp40 19.4 >> t3r 12.5
=> BOTH adp40 AND fastv (standard widely-cited baseline) beat t3r_orig@40% at matched budget on BOTH
suites. t3r_orig@40% is the LOWEST of all 5 pruning baselines on both. Honest result confirmed & robust.
GPU freed, all runs complete.

## LIGHTER-PRUNE SWEEP @192 (25% prune) — RUNNING (user: 'yes' to keep-192 sweep)
Tests whether t3r closes the gap vs adp/fastv at gentler pruning (25%/192 vs the 40%/154 point).
3 matched methods at keep=192: t3r192 (T3R prune+bias keep=0.75), adp192 (ADP QK gate-off keep=0.75),
fastv192 (FastV K=1 keep=192). Both suites (Google 10 units + Bridge 4). Drivers: run_sweep_t3r192.py,
run_sweep_af192.py (SWEEP_SUITE env). Orchestration run_sweep_all.sh (af first to derisk new driver) +
sweep_monitor.sh keepalive. Results -> sweep_{t3r192,af192}_{google,bridge}_results.json.
Validated: coke_upright/adp192 SR=0.84 kept=192 prune=25% (driver works). 42 cells, ~4-5hr.
