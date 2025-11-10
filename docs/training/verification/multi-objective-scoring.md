(training-verification-multi-objective)=

# Multi-Objective Scoring

Advanced guide for combining multiple reward signals in custom resource servers.

:::{card}

**Audience**: Building custom resource servers with multiple objectives

^^^

**You'll learn how to**:

1. Combine metrics with weighted combinations
2. Design hierarchical objectives
3. Monitor multiple metrics during training
4. Handle trade-offs between objectives

:::

**Prerequisites**: Basic verification from {doc}`../../get-started/verifying-agent-results` and {doc}`custom-patterns-cookbook`.

:::{tip}
**Using existing servers?** Most built-in servers (mcqa, comp_coding) use single objectives. Only multineedle and library_judge_math use multi-objective patterns—you likely don't need this guide.
:::

---

## When You Need Multi-Objective Scoring

Use multi-objective scoring when building custom verification for tasks with:

* **Multiple success criteria** - Correctness + efficiency + style
* **Conflicting objectives** - Speed vs thoroughness trade-offs
* **Hierarchical goals** - Must-have requirements + nice-to-have optimizations
* **Rich training signal** - Track multiple dimensions for analysis

**Built-in examples**: `multineedle` (accuracy + set_overlap), `library_judge_math` (library + judge)

---

## Automatic Metric Aggregation

NeMo Gym automatically aggregates **any numeric field** returned from `verify()`:

```python
async def verify(self, body: YourVerifyRequest) -> YourVerifyResponse:
    return YourVerifyResponse(
        **body.model_dump(),
        reward=0.85,         # Automatically aggregated
        correctness=0.90,    # Automatically aggregated
        efficiency=0.75      # Automatically aggregated
    )
```

After collection, all numeric fields are averaged across rollouts. See {doc}`../datasets/prepare-for-training` for using these metrics.

---

## Weighted Combination

Combine multiple objectives into single composite reward.

### Basic Pattern

Weight objectives by importance:

```python
async def verify(self, body: YourVerifyRequest) -> BaseVerifyResponse:
    # Calculate individual scores (0.0 to 1.0)
    correctness = check_correctness(response)
    efficiency = check_efficiency(response)
    style = check_style(response)
    
    # Weighted combination
    reward = (
        0.6 * correctness +  # Primary: 60%
        0.3 * efficiency +   # Secondary: 30%
        0.1 * style          # Tertiary: 10%
    )
    
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

**Key principle**: Weights should sum to 1.0 for interpretability.

### Choosing Weights

**Guidelines**:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Objective Type
  - Weight Range
* - **Primary (must-have)**
  - 0.5–0.7
* - **Secondary (important)**
  - 0.2–0.3
* - **Tertiary (nice-to-have)**
  - 0.1–0.2
```

**Example weight distributions**:

* **Correctness-dominant**: 0.7 correctness + 0.2 efficiency + 0.1 style
* **Balanced**: 0.5 accuracy + 0.3 relevance + 0.2 completeness
* **Quality-focused**: 0.6 correctness + 0.4 quality (binary efficiency)

### Testing Weight Combinations

Experiment with different weights:

```python
# Test different weight configurations
configs = [
    {"correctness": 0.8, "efficiency": 0.2},  # Correctness-heavy
    {"correctness": 0.6, "efficiency": 0.4},  # Balanced
    {"correctness": 0.5, "efficiency": 0.5},  # Equal weight
]

for config in configs:
    reward = (config["correctness"] * correctness_score + 
              config["efficiency"] * efficiency_score)
    # Evaluate if reward distribution is reasonable
```

**Goal**: Ensure high-reward rollouts align with your quality definition.

---

## Multiple Independent Metrics

Track multiple objectives without combining into single reward.

### Pattern: Return Multiple Fields

```python
async def verify(self, body: YourVerifyRequest) -> YourVerifyResponse:
    # Calculate metrics independently
    correctness = check_correctness(response)
    efficiency = check_efficiency(response)
    coverage = check_coverage(response)
    
    # Primary reward (choose most important)
    reward = correctness
    
    # Return all metrics
    return YourVerifyResponse(
        **body.model_dump(),
        reward=reward,         # Primary training signal
        correctness=correctness,  # Track separately
        efficiency=efficiency,    # Track separately
        coverage=coverage         # Track separately
    )
```

**Built-in example**: `resources_servers/multineedle/`

```python
# From multineedle/app.py:97-105
return MultiNeedleVerifyResponse(
    reward=float(accuracy),  # Primary metric
    accuracy=accuracy,        # Boolean metric
    set_overlap=set_overlap,  # Continuous metric
    original_term_minefield_hit=...,  # Additional tracking
    order_instruction_following_failure=...,  # Additional tracking
)
```

**Source**: `resources_servers/multineedle/app.py:86-105`

### When to Use Multiple Metrics

**Best for**:

* Exploratory analysis - Not sure which metric matters most yet
* Monitoring - Track multiple dimensions during training
* Post-hoc filtering - Filter rollouts by different criteria
* A/B testing - Compare different reward formulations

**Downstream usage**:

```python
import json

rollouts = [json.loads(line) for line in open('rollouts.jsonl')]

# Filter by different criteria
high_correctness = [r for r in rollouts if r['correctness'] > 0.9]
high_efficiency = [r for r in rollouts if r['efficiency'] > 0.8]
both_high = [r for r in rollouts if r['correctness'] > 0.9 and r['efficiency'] > 0.8]
```

---

## Hierarchical Objectives

Enforce must-have requirements before optimizing secondary goals.

### Pattern: Threshold-Based Weighting

```python
async def verify(self, body: YourVerifyRequest) -> BaseVerifyResponse:
    correctness = check_correctness(response)
    efficiency = check_efficiency(response)
    
    # Hierarchical: correctness is gating
    if correctness < 0.5:
        # Failed primary objective - low reward regardless of efficiency
        reward = correctness * 0.3
    else:
        # Met primary objective - now optimize efficiency too
        reward = 0.6 * correctness + 0.4 * efficiency
    
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

**Effect**: Agent must meet primary objective before secondary objective contributes.

### Pattern: Multiplicative Gating

```python
# Correctness gates efficiency reward
if is_correct:
    reward = 0.8 + 0.2 * efficiency  # 0.8 to 1.0 based on efficiency
else:
    reward = 0.3 * partial_correctness  # Max 0.3 if incorrect
```

**Effect**: Incorrect answers can never score high, regardless of efficiency.

---

## Common Patterns

See {doc}`custom-patterns-cookbook` Pattern 5 for complete multi-objective implementation.

### Quick Examples

**Correctness + Efficiency**:
```python
reward = 0.7 * correctness + 0.3 * efficiency
```

**Accuracy + Completeness**:
```python
reward = 0.6 * accuracy + 0.4 * completeness
```

**Precision + Recall** (F1-score):
```python
reward = 2 * (precision * recall) / (precision + recall)
```

---

## Monitoring Multi-Objective Training

### Track Metrics Over Time

After each collection run, monitor all metrics:

```bash
# Run 1
ng_collect_rollouts ... +output_jsonl_fpath=rollouts_iter1.jsonl
# Output: {"reward": 0.65, "correctness": 0.75, "efficiency": 0.60}

# Run 2 (after training)
ng_collect_rollouts ... +output_jsonl_fpath=rollouts_iter2.jsonl
# Output: {"reward": 0.71, "correctness": 0.82, "efficiency": 0.65}
```

**Analyze**:

* Did all metrics improve?
* Did optimizing one hurt another? (trade-off)
* Is improvement aligned with weights?

### Analyze Trade-Offs

```python
import json
import matplotlib.pyplot as plt

rollouts = [json.loads(line) for line in open('rollouts.jsonl')]

correctness = [r['correctness'] for r in rollouts]
efficiency = [r['efficiency'] for r in rollouts]

# Scatter plot to visualize trade-off
plt.scatter(correctness, efficiency)
plt.xlabel('Correctness')
plt.ylabel('Efficiency')
plt.title('Trade-off Analysis')
plt.show()
```

**Look for**:

* Negative correlation? (trade-off exists)
* Pareto frontier? (best achievable balance)
* Clusters? (different strategies)

### Adjusting Weights

If training optimizes wrong objective:

1. **Increase weight** on underperforming objective
2. **Re-collect rollouts** with adjusted reward
3. **Compare distributions** before and after
4. **Iterate** until balance matches goals

**Example**:

```yaml
iteration_1:
  weights: {correctness: 0.6, efficiency: 0.4}
  results: {correctness: 0.85, efficiency: 0.50}
  # Efficiency too low!

iteration_2:
  weights: {correctness: 0.5, efficiency: 0.5}  # Increase efficiency weight
  results: {correctness: 0.82, efficiency: 0.65}
  # Better balance
```

---

## Design Checklist

Before deploying multi-objective scoring:

- [ ] **Primary objective is clear** - One metric must be most important
- [ ] **Weights sum to 1.0** - For interpretability
- [ ] **Metrics are normalized** - All in [0.0, 1.0] range
- [ ] **Trade-offs are understood** - Know which objectives conflict
- [ ] **Hierarchical gating if needed** - Must-have vs nice-to-have
- [ ] **Tested on sample data** - Reward distribution makes sense
- [ ] **Monitoring plan** - Track all metrics during training


---

## Built-in Multi-Objective Examples

### MultiNeedle Resource Server

**Task**: Extract multiple values from long context.

**Metrics**:

```python
# From multineedle/app.py
accuracy = expected == actual  # Binary: all correct?
set_overlap = len(set(actual) & set(expected)) / len(expected)  # Partial credit
original_term_minefield_hit = ...  # Did agent avoid trap?
order_instruction_following_failure = ...  # Order matters?

reward = float(accuracy)  # Primary metric is binary accuracy
```

**Source**: `resources_servers/multineedle/app.py:86-105`

**Design choice**: Primary reward is binary, but track partial credit metrics separately.

### Library Judge Math Resource Server

**Task**: Math problem with symbolic + judge verification.

**Metrics**:

```python
# Hybrid approach
library_reward = symbolic_equivalence_check()  # Fast, deterministic
judge_reward = llm_judge_equivalence() if library_inconclusive else None

# Combine with preference for library
if library_reward is not None:
    reward = library_reward
elif judge_reward is not None:
    reward = judge_reward
else:
    reward = 0.0
```

**Source**: `resources_servers/library_judge_math/app.py:118`

**Design choice**: Sequential fallback (library first, judge if needed) rather than weighted average.

---

## Common Pitfalls

:::{dropdown} Conflicting Objectives
**Problem**: Optimizing one metric hurts another (e.g., speed vs thoroughness)

**Solution**: Use hierarchical gating or adjust weights to prioritize
:::

:::{dropdown} Unbalanced Scales
**Problem**: Metrics on different scales (0-100 vs 0.0-1.0)

**Example**:
```python
# Bug: metrics on different scales
reward = 0.5 * accuracy + 0.5 * response_length  # Wrong!
# accuracy ∈ [0, 1], response_length ∈ [0, 500]
```

**Solution**: Normalize all metrics to [0.0, 1.0]
:::

:::{dropdown} Correlation Masking
**Problem**: Two metrics highly correlated—combined weight too high

**Example**: `reward = 0.5 * accuracy + 0.5 * correctness` (same metric, doubled weight!)

**Solution**: Use orthogonal metrics (independent dimensions)
:::

:::{dropdown} Ignoring Primary Objective
**Problem**: Secondary objectives dominate because primary is too hard

**Solution**: Use hierarchical gating or adjust task difficulty
:::

---

## Related Topics

* {doc}`custom-patterns-cookbook` - Complete multi-objective implementation (Pattern 5)
* {doc}`../datasets/prepare-for-training` - Using multi-metric rollouts for training
* {doc}`../data-quality/index` - Validate multi-objective reward distributions
* {doc}`../rollout-collection/optimize-for-training/production-scale` - Monitor metrics during collection

---

## Next Steps

:::{button-ref} ../datasets/prepare-for-training
:color: primary
:outline:
:ref-type: doc

Prepare Multi-Metric Data →
:::

Or return to {doc}`index` for verification overview.
