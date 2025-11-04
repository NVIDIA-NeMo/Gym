(training-rollout-sampling-evaluation)=

# Evaluation Sampling Strategy

Configure for benchmarking: reproducible, deterministic evaluation with minimal variance.

:::{card}

**Task**: Generate reproducible evaluation measurements for comparing models and tracking progress over time.

^^^

**This guide shows you how to**:

1. Configure for deterministic, low-variance evaluation
2. Set up reproducible benchmarking with fixed seeds
3. Compare results across models and checkpoints
4. Detect statistically significant improvements

:::

---

## Before You Start

Ensure you have these prerequisites before running evaluation:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Requirement
  - Details
* - **Get Started completed**
  - Complete {doc}`../../../get-started/collecting-rollouts` first
* - **Servers running**
  - Agent and model servers with deterministic inference support
* - **Evaluation objective**
  - Understanding of benchmark metrics and comparison methodology
* - **Benchmark dataset**
  - Fixed evaluation set in JSONL format (100-5K samples)
* - **Baseline results**
  - Previous model's scores for comparison (optional)
```

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← New? Try Get Started
:::

---

## Run Evaluation

Generate reproducible, deterministic evaluation results for benchmarking.

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=eval_benchmark.jsonl \
    +output_jsonl_fpath=eval_model_v1.jsonl \
    +responses_create_params.temperature=<temperature> \
    +responses_create_params.seed=<seed> \
    +responses_create_params.top_p=<top_p> \
    +num_samples_in_parallel=<parallelism> \
    +limit=null
```

**Configuration**: For evaluation, use very low temperature and fixed seed for reproducibility, single samples per task, and low parallelism to minimize variance. Refer to {doc}`parameters` for parameter explanations.

---

## Model Comparison Workflow

### 1. Evaluate Multiple Models

```bash
#!/bin/bash
# Compare 3 model versions on same benchmark

BENCHMARK="eval_benchmark.jsonl"
MODELS=("model_v1" "model_v2" "model_v3")
SEED=42

for model in "${MODELS[@]}"; do
    echo "Evaluating $model..."
    
    ng_collect_rollouts \
        +agent_name=$model \
        +input_jsonl_fpath=$BENCHMARK \
        +output_jsonl_fpath=eval_${model}.jsonl \
        +responses_create_params.temperature=0.1 \
        +responses_create_params.seed=$SEED \
        +num_samples_in_parallel=5
    
    # Extract metrics
    avg_reward=$(jq -s 'map(.reward) | add/length' eval_${model}.jsonl)
    accuracy=$(jq -s 'map(select(.accuracy != null)) | map(.accuracy) | add/length' eval_${model}.jsonl)
    
    echo "$model: reward=$avg_reward, accuracy=$accuracy"
done
```

### 2. Compare Results

```bash
# Simple comparison
echo "model_v1: $(jq -s 'map(.reward) | add/length' eval_model_v1.jsonl)"
echo "model_v2: $(jq -s 'map(.reward) | add/length' eval_model_v2.jsonl)"
echo "model_v3: $(jq -s 'map(.reward) | add/length' eval_model_v3.jsonl)"
```

For statistical significance testing, use your preferred analysis tools (scipy, R, etc.).

---

## Best Practices

### Separate Eval from Training

**Keep evaluation data pristine**:
```bash
# Never train on eval set
eval_set="eval_benchmark.jsonl"  # Hold-out test set
train_set="train_data.jsonl"     # Training data only
```

### Version Control Results

```bash
# Save with model version and timestamp
timestamp=$(date +%Y%m%d_%H%M%S)
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=eval_benchmark.jsonl \
    +output_jsonl_fpath=eval_results/model_v2_${timestamp}.jsonl \
    +responses_create_params.temperature=0.1 \
    +responses_create_params.seed=42
```

### Run Multiple Times for Confidence

Even with low temperature, measure variance:

```bash
#!/bin/bash
# Run evaluation 3 times with different seeds

for seed in 42 123 456; do
    ng_collect_rollouts \
        +agent_name=my_agent \
        +input_jsonl_fpath=benchmark.jsonl \
        +output_jsonl_fpath=eval_seed${seed}.jsonl \
        +responses_create_params.temperature=0.1 \
        +responses_create_params.seed=$seed \
        +num_samples_in_parallel=5
done

# Compute mean and std across runs
python -c "
import json
import statistics

runs = []
for seed in [42, 123, 456]:
    with open(f'eval_seed{seed}.jsonl') as f:
        rewards = [json.loads(line)['reward'] for line in f]
        runs.append(statistics.mean(rewards))

print(f'Mean: {statistics.mean(runs):.3f} ± {statistics.stdev(runs):.3f}')
"
```

---

## Verify Reproducibility

Test that same seed produces consistent results:
```bash
# Run twice with same seed
ng_collect_rollouts ... +responses_create_params.seed=42 \
    +output_jsonl_fpath=eval_run1.jsonl
ng_collect_rollouts ... +responses_create_params.seed=42 \
    +output_jsonl_fpath=eval_run2.jsonl

# Compare results
diff <(jq '.reward' eval_run1.jsonl) <(jq '.reward' eval_run2.jsonl)
```

If results differ, verify seed is being respected by your model server.
