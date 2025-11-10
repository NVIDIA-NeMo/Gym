(training-rollout-patterns)=

# Command Reference

Essential commands and parameters for common rollout collection scenarios.

---

## By Training Type

Choose the optimal configuration for your training algorithm with these preset command patterns:

::::{tab-set}

:::{tab-item} SFT

```bash
ng_collect_rollouts +agent_name=AGENT +input_jsonl_fpath=INPUT +output_jsonl_fpath=OUTPUT +responses_create_params.temperature=0.2 +num_samples_in_parallel=20
```

:::

:::{tab-item} DPO

```bash
ng_collect_rollouts +agent_name=AGENT +input_jsonl_fpath=INPUT +output_jsonl_fpath=OUTPUT +responses_create_params.temperature=0.7 +num_repeats=3 +num_samples_in_parallel=10
```

:::

:::{tab-item} RL

```bash
ng_collect_rollouts +agent_name=AGENT +input_jsonl_fpath=INPUT +output_jsonl_fpath=OUTPUT +responses_create_params.temperature=0.5 +num_samples_in_parallel=15
```

:::

:::{tab-item} Evaluation

```bash
ng_collect_rollouts +agent_name=AGENT +input_jsonl_fpath=INPUT +output_jsonl_fpath=OUTPUT +responses_create_params.temperature=0.1 +responses_create_params.seed=42 +num_samples_in_parallel=5
```

:::

::::

---

## Common Parameters

Essential parameters for controlling rollout collection behavior across all training types:

```bash
# Basic
+agent_name=my_agent                           # Which agent to use
+input_jsonl_fpath=tasks.jsonl                 # Input dataset
+output_jsonl_fpath=rollouts.jsonl             # Output file

# Sampling
+responses_create_params.temperature=0.5       # Randomness (0.0-2.0)
+responses_create_params.top_p=0.95            # Nucleus sampling
+responses_create_params.max_output_tokens=512 # Output length limit
+responses_create_params.seed=42               # For reproducibility

# Collection Control
+limit=1000                                    # Process only first N tasks
+num_repeats=3                                 # Repeat each task N times
+num_samples_in_parallel=20                    # Concurrent requests
```

---

## Parameter Guide

Recommended value ranges for key parameters based on training objectives and infrastructure:

::::{tab-set}

:::{tab-item} Temperature

```yaml
0.0-0.2: Deterministic, consistent (SFT, evaluation)
0.3-0.5: Balanced exploration (RL, general training)
0.6-0.8: High diversity (DPO, preference pairs)
0.9-1.0: Maximum diversity (exploration, research)
```

:::

:::{tab-item} Parallelism

```yaml
1-5:   Sequential/debug, hosted APIs with rate limits
10-15: Moderate throughput, balanced load
20-30: High throughput, local GPU servers
40+:   Maximum throughput, multi-GPU or high-tier APIs
```

:::

:::{tab-item} Repeats

```yaml
1: Single sample per task (SFT, evaluation)
2-4: Preference pairs (DPO, RLAIF)
5-10: Exploration and variance analysis
```

:::

::::

---

## Quick Inspection

Verify and analyze collected rollouts with these command-line utilities:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Task
  - Command
* - View single rollout
  - `head -1 rollouts.jsonl | jq '.'`
* - Check rewards
  - `jq '.reward' rollouts.jsonl`
* - Compute average metrics
  - `jq -s 'map(.reward) | add/length' rollouts.jsonl`
* - Filter by reward threshold
  - `jq 'select(.reward >= 0.8)' rollouts.jsonl > filtered.jsonl`
* - Count rollouts
  - `wc -l rollouts.jsonl`
* - Interactive viewer
  - `ng_viewer +input_jsonl_fpath=rollouts.jsonl`
```
