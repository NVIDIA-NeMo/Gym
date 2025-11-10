(training-rollout-patterns-quick-reference)=

# Quick Command Reference

Essential commands and parameters for common rollout collection scenarios.

---

## By Training Type

### SFT

```bash
ng_collect_rollouts +agent_name=AGENT +input_jsonl_fpath=INPUT +output_jsonl_fpath=OUTPUT +responses_create_params.temperature=0.2 +num_samples_in_parallel=20
```

### DPO

```bash
ng_collect_rollouts +agent_name=AGENT +input_jsonl_fpath=INPUT +output_jsonl_fpath=OUTPUT +responses_create_params.temperature=0.7 +num_repeats=3 +num_samples_in_parallel=10
```

### RL

```bash
ng_collect_rollouts +agent_name=AGENT +input_jsonl_fpath=INPUT +output_jsonl_fpath=OUTPUT +responses_create_params.temperature=0.5 +num_samples_in_parallel=15
```

### Evaluation

```bash
ng_collect_rollouts +agent_name=AGENT +input_jsonl_fpath=INPUT +output_jsonl_fpath=OUTPUT +responses_create_params.temperature=0.1 +responses_create_params.seed=42 +num_samples_in_parallel=5
```

---

## Common Parameters

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

### Temperature

```yaml
0.0-0.2: Deterministic, consistent (SFT, evaluation)
0.3-0.5: Balanced exploration (RL, general training)
0.6-0.8: High diversity (DPO, preference pairs)
0.9-1.0: Maximum diversity (exploration, research)
```

### Parallelism

```yaml
1-5:   Sequential/debug, hosted APIs with rate limits
10-15: Moderate throughput, balanced load
20-30: High throughput, local GPU servers
40+:   Maximum throughput, multi-GPU or high-tier APIs
```

### Repeats

```yaml
1: Single sample per task (SFT, evaluation)
2-4: Preference pairs (DPO, RLAIF)
5-10: Exploration and variance analysis
```

---

## Quick Inspection

### View single rollout

```bash
head -1 rollouts.jsonl | jq '.'
```

### Check rewards

```bash
jq '.reward' rollouts.jsonl
```

### Compute average metrics

```bash
jq -s 'map(.reward) | add/length' rollouts.jsonl
```

### Filter by reward threshold

```bash
jq 'select(.reward >= 0.8)' rollouts.jsonl > filtered.jsonl
```

### Count rollouts

```bash
wc -l rollouts.jsonl
```

### Interactive viewer

```bash
ng_viewer +input_jsonl_fpath=rollouts.jsonl
```

---

## Post-Processing Scripts

### Shuffle dataset

```bash
shuf input.jsonl > output.jsonl
```

### Split into train/val

```bash
total=$(wc -l < dataset.jsonl)
train_size=$((total * 90 / 100))

head -n $train_size dataset.jsonl > train.jsonl
tail -n +$((train_size + 1)) dataset.jsonl > val.jsonl
```

### Merge multiple files

```bash
cat file1.jsonl file2.jsonl file3.jsonl > merged.jsonl
```

### Sample N random lines

```bash
shuf -n 1000 large_dataset.jsonl > sample.jsonl
```
