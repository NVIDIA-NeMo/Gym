(training-rollout-sampling-validation)=

# Measuring Success

Validate that your sampling strategy produces the intended data characteristics.

:::{card}

**Task**: Verify that collected rollouts match your training objectives by measuring distribution, diversity, and quality.

^^^

**This guide shows you how to**:

1. Analyze reward distributions for your strategy
2. Measure response diversity and uniqueness
3. Validate quality signals and filter thresholds
4. Compare results against expected patterns

:::

---

## Before You Start

Ensure you have these prerequisites before validating rollouts:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Requirement
  - Details
* - **Rollouts collected**
  - Output JSONL file from `ng_collect_rollouts` with reward/metrics
* - **Strategy selected**
  - Know which strategy you used (SFT, DPO, RL, etc.)
* - **Python environment**
  - Python with `jq` for metrics extraction and analysis
* - **Expected patterns**
  - Understanding of what distributions your strategy should produce
```

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← New? Try Get Started
:::

---

## Automatic Metrics

NeMo Gym automatically aggregates metrics after collection:

```json
{
  "reward": 0.734,
  "accuracy": 0.689,
  "avg_tool_calls": 2.3
}
```

Use these instant metrics to validate sampling strategy effectiveness before detailed analysis.

See {doc}`../../about/concepts/rollout-collection-fundamentals` for how automatic metric aggregation works.

---

## Key Validation Metrics

### Reward Distribution

The shape of your reward distribution should match your strategy.

**Check distribution**:


```bash
jq '.reward' rollouts.jsonl | python -c "
import sys
import statistics

rewards = [float(x) for x in sys.stdin]

# Basic stats
print(f'Count:  {len(rewards)}')
print(f'Mean:   {statistics.mean(rewards):.3f}')
print(f'Median: {statistics.median(rewards):.3f}')
print(f'Stdev:  {statistics.stdev(rewards):.3f}')
print(f'Min:    {min(rewards):.3f}')
print(f'Max:    {max(rewards):.3f}')

# Percentiles
sorted_rewards = sorted(rewards)
p25 = sorted_rewards[len(sorted_rewards) // 4]
p75 = sorted_rewards[3 * len(sorted_rewards) // 4]
print(f'P25:    {p25:.3f}')
print(f'P75:    {p75:.3f}')
"
```

Use these metrics to understand your data's characteristics for your specific task and strategy.

**Expected patterns by strategy**:

- **SFT**: High mean (0.7-0.9), low variance
- **DPO**: Moderate mean (0.5-0.7), moderate variance for quality spread
- **RL**: Initially low, should improve across iterations
- **Evaluation**: Task-dependent, but should be stable across runs

---

## Response Diversity

Measure uniqueness of completions.

**Count unique responses**:


```bash
jq -r '.output[] | select(.type=="message") | .content' rollouts.jsonl | \
  sort | uniq | wc -l

# Compare to total
total=$(wc -l < rollouts.jsonl)
unique=$(jq -r '.output[] | select(.type=="message") | .content' rollouts.jsonl | sort | uniq | wc -l)

echo "Diversity: $unique / $total = $(python -c "print(f'{$unique/$total:.1%}')")"
```

---

## Success Rate

Percentage of rollouts meeting quality threshold.

```bash
# Success rate at threshold 0.8
success=$(jq 'select(.reward >= 0.8)' rollouts.jsonl | wc -l)
total=$(wc -l < rollouts.jsonl)

echo "Success rate: $success / $total = $(python -c "print(f'{$success/$total:.1%}')")"
```

---

## Troubleshooting

::::{tab-set}

:::{tab-item} Very Low Success Rate

**Problem**: >80% of rollouts fail quality threshold

```bash
failure_rate=$(jq 'select(.reward < 0.3)' rollouts.jsonl | wc -l)
total=$(wc -l < rollouts.jsonl)
echo "Failure rate: $(python -c "print(f'{$failure_rate/$total:.1%}')")"
```

**Solution**: Task may be too difficult, or verification too strict. Adjust threshold or increase temperature.

:::

:::{tab-item} All Rewards Identical

**Problem**: No variance in reward values

```bash
jq '.reward' rollouts.jsonl | sort | uniq
```

**Solution**: If single value, check temperature > 0 and verify verification logic isn't constant.

:::

:::{tab-item} Nonsensical Outputs

**Problem**: Responses are gibberish or off-topic

```bash
# Random sample of 5 rollouts
jq '.output[] | select(.type=="message") | .content' rollouts.jsonl | shuf -n 5
```

**Solution**: Reduce temperature or top_p.

:::

::::
