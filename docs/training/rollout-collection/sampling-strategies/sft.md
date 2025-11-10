(training-rollout-sampling-sft)=

# SFT Sampling Strategy

Configure for supervised fine-tuning: consistent, high-quality demonstrations at scale.

:::{card}

**Task**: Generate 10K-1M supervised fine-tuning demonstrations with consistent, high-quality agent behaviors.

^^^

**This guide shows you how to**:

1. Configure temperature and sampling for consistent demonstrations
2. Optimize parallelism for large-scale dataset generation
3. Validate data quality and filter low-reward rollouts
4. Prepare final training format for SFT

:::

---

## Before You Start

Ensure you have these prerequisites before generating SFT data:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Requirement
  - Details
* - **Get Started completed**
  - Complete {doc}`../../../get-started/collecting-rollouts` first
* - **Servers running**
  - Agent and model servers collecting rollouts successfully
* - **Training objective**
  - Understanding of SFT (learning from single high-quality demonstrations)
* - **Task dataset**
  - Input prompts in JSONL format (10K+ recommended)
* - **Infrastructure**
  - Local GPU or hosted API with sufficient throughput capacity
```

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← New? Try Get Started
:::

---

## Collect SFT Data

Generate consistent demonstrations for supervised fine-tuning by collecting rollouts with appropriate temperature and parallelism settings.

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=sft_train.jsonl \
    +output_jsonl_fpath=sft_rollouts.jsonl \
    +responses_create_params.temperature=<temperature> \
    +responses_create_params.top_p=<top_p> \
    +num_samples_in_parallel=<parallelism>
```

**Configuration**: For SFT, use low temperature for consistency, single samples per task, and high parallelism for throughput. Refer to {doc}`parameters` for parameter explanations.

### Expected Output

```
Found 10000 rows!
Querying with 20 concurrent requests
Collecting rollouts: 100%|████████████| 10000/10000 [08:32<00:00, 19.52it/s]
{
    "reward": 0.734,
    "accuracy": 0.689
}
```

**Throughput**: 15-20 samples/sec typical for hosted API or local 70B model.

---

## Post-Collection Workflow

Process and prepare your collected rollouts for training.

### 1. Check Reward Distribution

```bash
jq '.reward' sft_rollouts.jsonl | \
  python -c "
import sys
import statistics

rewards = [float(x) for x in sys.stdin]
print(f'Count:  {len(rewards)}')
print(f'Mean:   {statistics.mean(rewards):.3f}')
print(f'Median: {statistics.median(rewards):.3f}')
print(f'Stdev:  {statistics.stdev(rewards):.3f}')
print(f'Min:    {min(rewards):.3f}')
print(f'Max:    {max(rewards):.3f}')
"
```

Use these metrics to understand your task difficulty and set an appropriate filtering threshold.

### 2. Filter for High Quality

Keep only high-reward demonstrations:

```bash
# Common threshold: 0.8
jq 'select(.reward >= 0.8)' sft_rollouts.jsonl > sft_filtered.jsonl

# Check filtering rate
original=$(wc -l < sft_rollouts.jsonl)
filtered=$(wc -l < sft_filtered.jsonl)
echo "Kept $filtered / $original ($(python -c "print(f'{$filtered/$original*100:.1f}%')"))"
```

Adjust the threshold based on your task difficulty. If most rollouts fail, lower the threshold or see {doc}`../optimize-for-training/identify-bottleneck`.

### 3. Shuffle for Training

Randomize order to avoid sequence biases:

```bash
shuf sft_filtered.jsonl > sft_train_ready.jsonl
```

**Why shuffle**: Prevents training from learning dataset ordering biases.

---

## Troubleshooting

::::{tab-set}

:::{tab-item} Low Success Rate

**Problem**: Few rollouts pass filtering

```bash
# Lower filtering threshold
jq 'select(.reward >= 0.6)' sft_rollouts.jsonl > sft_filtered.jsonl

# Or increase temperature for more exploration
ng_collect_rollouts ... +responses_create_params.temperature=0.3
```

:::

:::{tab-item} Collection Too Slow

**Problem**: <10 samples/sec throughput

```bash
# Increase parallelism
ng_collect_rollouts ... +num_samples_in_parallel=30

# Or reduce output length
ng_collect_rollouts ... +responses_create_params.max_output_tokens=512
```

See {doc}`../optimize-for-training/index` for comprehensive optimization.

:::

:::{tab-item} Responses Too Repetitive

**Problem**: Many identical completions

```bash
# Increase temperature
ng_collect_rollouts ... +responses_create_params.temperature=0.4
```

:::

::::
