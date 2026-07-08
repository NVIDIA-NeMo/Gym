# Parity Evidence: NeMo Gym Port vs Native EnterpriseOps-Gym

This document consolidates the evidence that this resources server + `simple_agent` /
`turn_logging_agent` reproduce the upstream [EnterpriseOps-Gym](https://github.com/ServiceNow/EnterpriseOps-Gym)
harness's scoring behavior. All live comparisons: `azure/openai/gpt-4.1-mini` via an
OpenAI-compatible gateway, temperature 0.0, max 16384 output tokens, concurrency 8,
oracle mode, the same live MCP containers and seed databases for both harnesses.

## 1. Static parity (unit level)

- `verifier_engine.py` is a line-for-line port of upstream `benchmark/verifier.py` +
  the scoring semantics of `benchmark/executor.py`, pinned by **golden fixtures generated
  by running the original implementation** on shared inputs
  (`tests/generate_parity_golden.py`, 19 extraction + 14 comparison cases, byte-identical).
- Intentionally preserved upstream quirks (do not "fix" without breaking leaderboard
  comparability):
  - **Verifier name-collapse**: results are keyed by verifier name; duplicate-named
    verifiers overwrite so only the last per name is scored (upstream `executor.py:474`).
    An earlier same-named failure can be masked. `strict_verifiers: true` switches the
    reward to every-verifier-counts for RL.
  - Verifiers with unknown `gym_name` are skipped entirely (not counted).
  - Loose comparison semantics (`0 != "0"`; type errors -> failed comparison).
  - Tool observations are `json.dumps(<MCP jsonrpc result>)`, matching what the upstream
    harness feeds its model; tool merge order is per task's `gym_servers_config` order.
- Tool schemas advertised to the model pass through the upstream schema cleaner
  (nullable-type collapse, required-list demotion), bound non-strict.
- The public oracle split contains **only `database_state` verifiers** (3,496/3,496), so
  LLM-judge (`response_check`) semantics do not affect any number below; the judge
  temperature is pinned to 0.0 regardless for full-split parity.

## 2. Live task-level parity (12 CSM tasks, Phase 2)

Same 12 tasks through both harnesses: **12/12 identical success/fail outcomes, 12/12
identical collapsed-verifier counts and name sets**.

## 3. Full public split, single runs (649 tasks)

| Run | Harness / agent | Macro success % |
|---|---|---|
| A | port / simple_agent | 16.4 |
| B | native EOG | 16.8 |
| C | port / turn_logging_agent | 17.8 |

- Per-task agreement: A↔B 90.4% (McNemar exact p = 0.90), C↔A 93.7% (p = 0.21),
  C↔B 89.9% (p = 0.46). Disagreements symmetric in every pairing.
- Collapsed-verifier scoring structure identical on 100% of cleanly completed tasks in
  every pairing.
- Stratified paired bootstrap (10k): every pairwise macro delta CI spans zero
  (e.g. B−A [−2.2, +3.0] pp). **Single-run macro deltas under ~3 pp are unreadable** on
  this benchmark/model/gateway.
- Turn-0 prompts across harnesses differ by a constant −2 tokens (−0.04%): prompt
  serialization is byte-equivalent and ruled out as a divergence driver.

## 4. Definitive variance experiment (k=5 per implementation, 6,480 rollouts)

Design: 5 interleaved blocks with alternating implementation order; full container-fleet
restart before every pass (fd-leak reset); 645-task panel present in all 10 passes.

| Metric | Result |
|---|---|
| mean@5 macro | port **16.54 ± 0.73** vs native **16.76 ± 0.99** → Δ −0.23 pp |
| Per-task preference | **67 tasks favor port, 67 favor native**, 511 tied (sign p = 1.0) |
| Outcome flip rate, within-implementation | 6.57% (port), 6.05% (native) |
| Outcome flip rate, between implementations | 9.80% → excess **+3.49 pp, permutation p < 0.002** |
| First-tool-call divergence | 2.4–2.6% within, 8.60% between (byte-equal prompts) |
| Sensitivity (passes 1–4 only) | Δ −0.31 pp (unchanged) |

**Conclusion.** The implementations are **score-equivalent to within ±0.7 pp with no
directional bias**. A statistically robust but direction-free effect exists between the
two API serving paths (Responses API vs Chat Completions at the gateway): it flips ~3.5%
more borderline task outcomes than a same-implementation rerun, rooted at the very first
sampled action, and moves no aggregate or per-task scores. It is a property of the
inference gateway, not of either harness. Eliminating it requires deterministic
self-hosted serving (vLLM, greedy + fixed seed).

**Calibration guidance.** ~6.3% of task outcomes flip on *any* rerun at temperature 0
through a hosted gateway; ~20% of tasks are flaky. Report mean@k (k≥5 → ±0.7 pp) rather
than single-run scores.

## 5. Operational findings (upstream-relevant)

- Most EOG MCP containers leak ~5 file descriptors per task lifecycle; at the default
  ulimit (1024) a container serves ~200 tasks before seeding fails with
  "unable to open database file". Restart containers between large runs (or use
  `gym_url_pools` replicas).
- The upstream runner (`utils/task_queue_worker.py`) never retrieves task results, so
  task exceptions are silently dropped (~2 tasks/pass observed); its executor also leaves
  orphaned database files when seeding fails mid-retry.
- `ng_collect_rollouts` aborts the batch on a terminal `/run` error; reruns with
  `+resume_from_cache=true` complete exactly the missing tasks. A continue-on-error
  collection mode would remove the manual resume step.
