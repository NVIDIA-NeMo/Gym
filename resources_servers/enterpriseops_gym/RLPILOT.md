# RL Rollout-Collection Pilot: GRPO Readiness with Zero Code Changes

Companion to [PARITY.md](PARITY.md) (score equivalence) and [PERF.md](PERF.md)
(throughput at scale). This document reports the pilot that validates the existing
integration as an RL rollout collector — token-ID capture, reward signal, and
group-variance (trainability) measurement — using **configuration only**: no code
changes to the resources server, agent, or model server.

## Setup

Same stack as [PERF.md](PERF.md): `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` on
vLLM (4× H100, TP=4, reasoning enabled), live MCP container fleet, oracle public
split. Rollouts collected with the stock `ng_collect_rollouts` +
`enterpriseops_gym_simple_agent`.

The one non-default switch RL needs:

```
++policy_model.responses_api_models.vllm_model.return_token_id_information=true
```

With it, every sampled assistant message carries `generation_token_ids`,
`generation_log_probs`, and `prompt_token_ids` (plus `routed_experts` for this MoE
model) — the `_ForTraining` payload NeMo RL consumes. Without it, rollouts are
silently token-free: **verify the resolved config** (`curl
localhost:11000/global_config_dict_yaml | grep return_token_id_information`) before
any collection run intended for training.

## Design and runs

k rollouts per task; a task group is *mixed* when its rollouts disagree — mixed
groups are where GRPO's group-relative advantage is nonzero, so the fraction of
mixed groups is the trainability signal.

| Run | Tasks | k | Sampling | Success | Binary-mixed groups |
|---|---|---:|---|---:|---:|
| v1 | first 20 rows | 4 | temp 1.0 | 0/80 | — (token IDs absent) |
| v2 | first 20 rows | 4 | temp 0.6, top_p 0.95 | 1/80 | — (subset artifact, below) |
| t0 control | first 20 rows | 1 | temp 0.0 | 1/20 | — |
| **v3** | **20 mixed-band tasks** | **8** | **temp 0.6, top_p 0.95** | **89/160 (55.6%)** | **15/20** |

All runs: `max_output_tokens: 16384`, concurrency 32 (v3: 160 rollouts in 22 min —
consistent with the PERF.md throughput at this concurrency).

## The task-subset trap (the pilot's main methodological lesson)

v1/v2 initially read as "sampling destroys performance" (0–1% success vs the
22–25% full-split baseline). Every intermediate hypothesis — reasoning runaway,
truncation, token-ID-flag side effects, fd-leaked fleet — was eliminated by
inspection; the control run then failed at **temperature 0 too**, and scoring the
PERF.md sweep's own five temp-0 passes on the same 20 tasks gave 0–1/20 (mean
strict pass rate 0.41–0.50, identical to the pilot's 0.45–0.51).

The cause: `+limit=N` takes the **first N rows of the benchmark file, which is not
shuffled** — the head of the file is a single-domain block of tasks that define
3–12 verifiers each and are nearly unwinnable under all-or-nothing scoring (agents
passed ~50% of verifiers on them in every run, at every temperature). Nothing was
broken and sampling cost nothing; the subset was unrepresentative.

**Rule: never pilot or smoke-test with `+limit` on this benchmark.** Select tasks
deliberately (below), or shuffle first.

## Curriculum selection from repeat-run data

The PERF.md sweep doubles as k=5 temp-0 outcome data per task. Of the 645-task
panel, **190 tasks (29%) are in the mixed band** (1–4 successes out of 5) — the
natural GRPO curriculum, since tasks that flip on rerun are exactly the tasks
where sampled groups will mix. v3 used the 20 tasks closest to 50% (2–3 wins/5).

Selection mechanics: rollout rows join to benchmark rows via `_ng_task_index`
(the row's position in the input file). Content-based keys do **not** survive the
collector's input materialization — hash-matching rollouts to raw benchmark rows
silently matches zero rows.

## v3 results — the go/no-go numbers

| Metric | Result | Reading |
|---|---|---|
| Token-ID coverage | 156/156 assistant messages | training data flows end-to-end |
| Success | 89/160 = 55.6% | dead-center of the sweep-predicted 40–60% band → temp 0.6 sampling costs nothing |
| Binary-mixed groups | **15/20** | nonzero GRPO advantage on 75% of curriculum tasks at k=8 |
| Strict-mixed groups | 17/20 | fractional reward (`strict_pass_rate`) has gradient almost everywhere |
| Mean within-group binary std | **0.341** | strong advantage signal (0/1-reward maximum is 0.5) |

## Recommended GRPO configuration

- **Curriculum**: tasks with repeat-run success in the ~20–80% band (190/645 on
  this model; recompute per policy model from a k≥5 temp-0 sweep). Refresh as the
  policy improves — the band moves.
- **Reward**: binary `reward` (leaderboard-comparable) as the objective;
  `strict_pass_rate` (per-verifier fraction, name-collapse disabled — see
  PARITY.md §1) as shaping or fallback where binary groups are unmixed.
- **Sampling**: temperature 0.6, top_p 0.95 (Nemotron reasoning recommendation),
  `max_output_tokens` 16384. Temp 1.0 measured strictly worse on this model.
- **Group size**: k=8 gave 75% mixed groups on a 40–60%-band curriculum.
- **Eval**: mean@k, never single runs — this stack's reasoning models flip ~15% of
  task outcomes on any rerun (PARITY.md §5).

## Operational notes for long collection runs

- **Restart the MCP fleet every ~200 task lifecycles** (fd leak, PARITY.md §6);
  a training loop's rollout phase hits this budget quickly.
- **Point `RAY_TMPDIR` at a large scratch mount.** With the default `/tmp`, each
  `ng_run` leaves a Ray session directory behind; orphaned `raylet`/`gcs_server`
  processes also outlive `pkill -f ng_run` and pin deleted files until killed.
  A filled root disk here fails with Python's "No usable temporary directory".
- **Watch syslog growth** on vLLM hosts (~GB/day observed); vacuum journals and
  truncate before multi-day runs.

Raw artifacts: `results/rl_pilot/` on the collection host (`rollouts_v2.jsonl`,
`rollouts_t0_control.jsonl`, `pilot_tasks_v3.jsonl`, `rollouts_v3.jsonl` +
aggregate metrics).
