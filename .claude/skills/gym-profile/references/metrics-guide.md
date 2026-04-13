# NeMo Gym Reward Profiling Metrics

Self-contained reference for interpreting rollout results and reward distributions.

---

## Core metrics

| Metric | Definition | Formula |
|--------|-----------|---------|
| **pass@1** | Average reward across all rollouts | `mean(all_rewards)` |
| **pass@k** | Fraction of tasks where at least one of k rollouts succeeded | `mean(max_reward_per_task >= pass_threshold)` |
| **avg_reward** | Mean reward per task across its rollouts | Per-task: `mean(task_rewards)` |
| **max_reward** | Highest reward for a task across all rollouts | Per-task: `max(task_rewards)` |

For binary rewards (0.0/1.0), pass@1 equals success rate. For non-binary rewards, pass@1 is the average including partial values.

---

## pass_threshold

Controls what counts as "pass" for pass@k calculation.

```bash
# Default: only full credit counts as pass
ng_reward_profile ... +pass_threshold=1.0

# Count partial credit (0.5) as pass
ng_reward_profile ... +pass_threshold=0.5
```

Critical for judge-based benchmarks with non-binary rewards. A threshold of 1.0 means only full credit counts; 0.5 means the judge fallback path (reward_if_full_generation_succeeds) also counts.

---

## Variance and num_repeats

**Target:** Variance < 1% across runs on the same model.

```bash
ng_collect_rollouts ... +num_repeats=5
```

| Situation | Action |
|-----------|--------|
| Variance < 1% | Sufficient repeats |
| Variance 1-3% | Increase to num_repeats=10 |
| Variance > 3% | Investigate: nondeterministic verification? Flaky tool execution? |

High variance sources:
- **Temperature**: `temperature=1.0` is standard for rollouts. Lower values reduce variance but also reduce diversity.
- **Nondeterministic verification**: External services, network-dependent tools
- **Flaky subprocess execution**: Timeouts, resource contention

---

## Suspicious patterns

| Pattern | Likely cause | Action |
|---------|-------------|--------|
| All tasks at 0% | Extraction bug, not model failure | Check `failure_reason` or `extracted_model_code` |
| All tasks at 100% | Trivial tasks or broken verification | Audit `verify()` logic |
| >30% of tasks at 100% (ceiling effect) | Tasks too easy; adds noise, not signal | Trim or weight down |
| >30% of tasks at 0% (floor effect) | Bugs or genuinely too hard | Inspect `failure_reason` to distinguish |
| Rewards not in {0.0, 1.0} | Partial credit | Check if intentional (judge fallback, combined reward) |
| Thinking model scores << instruct model | Think-block stripping failure | Check `reasoning_format_violation_rate` |
| Uniform distribution across models | Benchmark doesn't differentiate capability levels | Check task difficulty distribution |

### Ideal distribution

Tasks should have 10-90% pass rates with **model separation** — different models should score differently on the same task. Tasks where all models agree (0% or 100%) are uninformative for training.

---

## Per-task difficulty analysis

Sort tasks by pass rate to identify ceiling and floor effects:

```python
import json

tasks = []
with open("profiled.jsonl") as f:
    for line in f:
        tasks.append(json.loads(line))

tasks.sort(key=lambda t: t["avg_reward"])

# Floor tasks (0% pass rate)
floor = [t for t in tasks if t["avg_reward"] == 0.0]
print(f"Floor: {len(floor)} tasks ({100*len(floor)/len(tasks):.0f}%)")
for t in floor:
    print(f"  task {t['task_index']}: {t.get('failure_reason', 'unknown')}")

# Ceiling tasks (100% pass rate)
ceiling = [t for t in tasks if t["avg_reward"] == 1.0]
print(f"Ceiling: {len(ceiling)} tasks ({100*len(ceiling)/len(tasks):.0f}%)")

# Middle (useful for training)
middle = [t for t in tasks if 0 < t["avg_reward"] < 1.0]
print(f"Middle: {len(middle)} tasks ({100*len(middle)/len(tasks):.0f}%)")
```

---

## Thinking model diagnostics

When comparing instruct vs thinking model on the same benchmark:

| Field | What to check |
|-------|--------------|
| `reasoning_format_violation_rate` | High value (>0.1) → think-block stripping is broken |
| `extracted_model_code` | Compare between models — if thinking model's extraction includes `<think>` content, that's the bug |
| Per-task comparison | Tasks where instruct passes but thinking fails → likely extraction issue on those tasks |

If thinking model scores lower than instruct:
1. Check `reasoning_format_violation_rate` — if high, stripping is the issue
2. Compare `extracted_model_code` between models on the same task
3. Check per-task: tasks with divergent scores point to specific extraction patterns

---

## Judge-specific diagnostics

| Partial reward value | Source | Config field |
|---------------------|--------|--------------|
| 0.5 (default) | `check_full_generation_on_fail` fallback succeeded | `reward_if_full_generation_succeeds` |
| 0.0 (when expected 1.0) | `check_twice_swap` disagreed | `reward_if_swap_fails` |
| 0.3 (example) | Combined reward: SAFE but low quality | `reward_if_quality_low` |

If seeing unexpected partial rewards:
1. Read `judge_evaluations` from rollout JSONL
2. Check which reward path was taken (primary, swap_fail, fallback)
3. Verify the config values match expectations
