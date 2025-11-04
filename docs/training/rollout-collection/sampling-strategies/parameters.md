(training-rollout-sampling-parameters)=

# Understanding Sampling Parameters

Learn what temperature, top_p, and num_repeats control and how they affect rollout characteristics.

Three key parameters shape your rollouts—understand each one to configure them effectively for your training goal.

---

## Quick Reference

Choose parameters based on your training goal:

```{list-table}
:header-rows: 1
:widths: 25 20 15 15 25

* - Goal
  - Temperature
  - Top-p
  - Num Repeats
  - Learn More
* - **Consistent demonstrations** (SFT)
  - Low
  - 0.9-0.95
  - 1
  - {doc}`sft`
* - **Preference pairs** (DPO)
  - Higher
  - 0.9
  - 3-4
  - {doc}`dpo`
* - **Balanced exploration** (RL)
  - Moderate
  - 0.9-0.95
  - 1
  - {doc}`rl`
* - **Reproducible eval**
  - Very low
  - 0.95
  - 1
  - {doc}`evaluation`
* - **Behavioral research**
  - High
  - 0.9
  - 5+
  - {doc}`research`
```

---

## Core Parameters

Explore each parameter in detail—understand what it controls, when to use different values, and how it affects rollout behavior.

::::{tab-set}

:::{tab-item} Temperature
:sync: temperature

**What it controls**: Randomness in token selection

**Scale**: 0.0 (deterministic) to 2.0 (extremely random)

**Command**: `+responses_create_params.temperature=VALUE`

### Behavior by Range

Different temperature ranges produce distinct rollout characteristics.

```{list-table}
:header-rows: 1
:widths: 20 40 40

* - Temperature
  - Behavior
  - Use Case
* - **0.1-0.3**
  - Consistent, repetitive responses
  - SFT demonstrations, evaluation
* - **0.4-0.6**
  - Balanced exploration
  - RL training, general use
* - **0.7-0.9**
  - Diverse, creative responses
  - DPO preference pairs, research
* - **1.0-1.5**
  - Very diverse, sometimes erratic
  - Extreme exploration, bootstrapping
```

```{dropdown} How Temperature Works
:icon: gear
:color: success

Temperature sharpens or flattens the probability distribution over next tokens:

- **Lower values** → Model picks most likely tokens consistently
- **Higher values** → Model explores less likely tokens more often

**Visual representation**:
~~~
Low Temperature (0.1)          High Temperature (1.0)
     ▲                              ▲
 95% │█                         30% │▄
     │█                             │█
     │█                             │█
 5%  │▁▁▁▁                      20% │▅▄▃
     └─────► tokens                 └─────► tokens
   (peaked)                       (spread out)
~~~
```

```{dropdown} Examples
:icon: code
:color: success

**Temperature 0.2** (Consistent):
~~~
User: "What's 2+2?"
Sample 1: "The answer is 4."
Sample 2: "The answer is 4."
Sample 3: "2+2 equals 4."
~~~

**Temperature 0.7** (Diverse):
~~~
User: "What's 2+2?"
Sample 1: "The answer is 4."
Sample 2: "Let me calculate: 2+2=4"
Sample 3: "That's simple math. Two plus two equals four."
~~~
```

:::

:::{tab-item} Top-p
:sync: top_p

**What it controls**: Diversity by probability mass threshold

**Scale**: 0.0 to 1.0 (typically 0.9-0.95)

**Command**: `+responses_create_params.top_p=VALUE`

### Recommended Values

Start with these proven values for most use cases.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Value
  - When to Use
* - **0.9-0.95**
  - Default, works well for most cases
* - **0.8**
  - More focused, fewer low-probability tokens
* - **1.0**
  - No filtering, all tokens considered
```

**Does not need tuning in most cases**—temperature provides enough control. Lower to 0.8-0.9 if model produces nonsensical outputs at high temperature.

```{dropdown} How Top-p Works
:icon: gear
:color: success

Model samples only from the smallest set of tokens whose cumulative probability exceeds top_p. Dynamically adjusts vocabulary size based on model's confidence.

**Visual example**:
~~~
All tokens ranked by probability:
Token A: 40% │████████████
Token B: 25% │████████
Token C: 15% │█████
Token D: 10% │███
Token E: 5%  │█
Token F: 3%  │
Token G: 2%  │

With top_p=0.9:
- Cumulative: A(40%) + B(65%) + C(80%) + D(90%)
- Sample from: A, B, C, D only
- Ignore: E, F, G (low probability tail)
~~~
```

:::

:::{tab-item} Num Repeats
:sync: repeats

**What it controls**: Multiple rollouts per input task

**Scale**: 1 (default) to any integer

**Command**: `+num_repeats=VALUE`

### Use Cases

Select the right number of repeats based on your training or research objective.

```{list-table}
:header-rows: 1
:widths: 30 25 45

* - Use Case
  - Typical Value
  - Purpose
* - **SFT training**
  - 1
  - Single demonstration per task
* - **DPO training**
  - 3-4
  - Generate preference pairs
* - **Variance measurement**
  - 5
  - Assess consistency with mean@k
* - **Behavioral research**
  - 5+
  - Observe multiple strategies
```

**Cost Impact**: Linear increase—`num_repeats=3` with 1,000 tasks = 3,000 rollouts.

**Configuration options**:

- **Command-line**: `+num_repeats=3` (applies to entire run)
- **Dataset config**: Set default per-dataset in config YAML

  ```yaml
  datasets:
  - name: validation
    num_repeats: 10  # Use pass@10 for code benchmarks
  ```

```{dropdown} How Num Repeats Works
:icon: gear
:color: success

Each task is repeated consecutively in the output—input `[A, B, C]` with `num_repeats=3` produces `[A, A, A, B, B, B, C, C, C]`.

**Guaranteed grouping structure**:
- Every `num_repeats` consecutive lines share the same input
- Groups are stable and predictable (not random or interleaved)
- Post-processing can rely on this structure

**Example with num_repeats=3**:
~~~python
# Every 3 consecutive rollouts = 1 group
groups = [rollouts[i:i+3] for i in range(0, len(rollouts), 3)]

# Group 1: rollouts[0], rollouts[1], rollouts[2] → same input
# Group 2: rollouts[3], rollouts[4], rollouts[5] → same input
# ...
~~~

This makes DPO pairing straightforward—sort each group by reward and select chosen/rejected pairs.
```

```{dropdown} DPO Training Pattern
:icon: code
:color: success

~~~bash
+num_repeats=3
# Creates groups of 3 rollouts per task
# Select highest reward as "chosen"
# Select lower reward as "rejected"
~~~
```

:::

::::

---

## Parameter Interactions

Parameters work together—understand their combined effects.

::::{tab-set}

:::{tab-item} Temperature + Top-p

```{list-table}
:header-rows: 1
:widths: 30 30 40

* - Configuration
  - Effect
  - Primary Use
* - **Both low**<br/>(temp=0.2, top_p=0.8)
  - Very focused, deterministic
  - Evaluation, consistent SFT
* - **Both moderate**<br/>(temp=0.5, top_p=0.95)
  - Balanced exploration
  - RL, general training
* - **High temp, moderate top_p**<br/>(temp=0.9, top_p=0.9)
  - Diverse but not chaotic
  - DPO, research
* - **Both high**<br/>(temp=1.2, top_p=1.0)
  - Maximum randomness
  - Extreme exploration (rare)
```

:::

:::{tab-item} Temperature + Num Repeats

```{list-table}
:header-rows: 1
:widths: 35 30 35

* - Configuration
  - Effect
  - Primary Use
* - **Low temp + repeats=1**<br/>(temp=0.2)
  - Single consistent sample per task
  - SFT
* - **Moderate temp + repeats=3-4**<br/>(temp=0.5)
  - Multiple diverse attempts per task
  - DPO preference pairs
* - **High temp + repeats=5+**<br/>(temp=0.9)
  - Explore behavioral variance
  - Research, analysis
```

:::

::::

---

## Parameter Sources and Defaults

Understand where parameter values come from and how to ensure reproducibility.

### Default Behavior

If you don't specify a parameter, your **model server determines the default**:

```{list-table}
:header-rows: 1
:widths: 30 25 45

* - Parameter
  - Common Defaults
  - Recommendation
* - **temperature**
  - 1.0 (OpenAI)<br/>varies (vLLM)
  - Always set explicitly for reproducibility
* - **top_p**
  - 1.0 (most servers)
  - Set to 0.9-0.95 for controlled sampling
* - **seed**
  - None (random)
  - Required for reproducible evaluation
* - **max_output_tokens**
  - Server-dependent
  - Set to control cost and latency
```

For evaluation and benchmarking, explicitly set all parameters rather than relying on defaults.

### Override Precedence

Parameters can be set in multiple places—CLI overrides take highest priority:

```bash
# Input JSONL contains per-task settings
{"responses_create_params": {"temperature": 0.5}, ...}

# CLI override applies to all tasks
ng_collect_rollouts ... +responses_create_params.temperature=0.7

# Result: CLI value (0.7) is used for all tasks
```

**Precedence order** (highest to lowest):

1. CLI parameters (`+responses_create_params.X`)
2. Input JSONL per-task parameters
3. Dataset config defaults
4. Model server defaults

This enables per-task customization while allowing global overrides when needed.

---

## Verification Speed Considerations

Your resource server's verification approach affects collection throughput and strategy viability.

### Verification Types

Resource servers verify rollouts using different approaches with distinct performance characteristics:

```{list-table}
:header-rows: 1
:widths: 25 40 35

* - Verification Type
  - Examples
  - Characteristics
* - **Binary/Pattern Matching**
  - MCQA answer extraction, boxed answer detection, string comparison
  - Fast (< 10ms)<br/>✅ Ideal for high-volume collection
* - **Execution-Based**
  - Code execution (comp_coding), Python evaluation, sandbox testing
  - Moderate (10-500ms)<br/>⚠️ May need lower parallelism
* - **LLM Judge**
  - Equivalence checking via external LLM, semantic comparison
  - Slow (500ms-2s)<br/>⚠️ Can bottleneck high-throughput strategies
```

### Impact on Sampling Strategies

**High-Throughput Strategies** (SFT, RL iterations):
- Fast verification enables aggressive parallelism (20-40 concurrent)
- Slow verification may require reduced parallelism (5-10 concurrent)
- LLM judge verification can bottleneck RL's iterative collection loops

**Lower-Volume Strategies** (DPO, Evaluation):
- Verification speed less critical with moderate parallelism
- Quality of verification more important than speed

### Optimization Tips

```{dropdown} If Verification is Bottlenecking
:icon: tools
:color: info

**For Execution-Based Verification**:
- Tune parallelism: {doc}`../optimize-for-training/tune-parallelism`
- Consider running resource server on dedicated hardware

**For LLM Judge Verification**:
- **Hybrid verification**: Use fast pre-filter before calling judge (see `library_judge_math` for pattern)
  - Example: Try symbolic math verification first, only call judge on failures
  - Reduces judge API calls by 50-80% for mathematical equivalence tasks
- **Judge model selection**: Configure faster judge model via `judge_model_server` in resource server config
- **RL-specific**: For iterative RL with high-volume collection, fast verification is critical—consider execution-based or pattern-matching verification instead

**Diagnosis**:
- Check {doc}`../optimize-for-training/identify-bottleneck` to confirm verification is limiting factor
- Monitor resource server response times during collection
```

**Note**: Verification design belongs in resource server development. This section focuses on how verification speed affects your sampling strategy choices.

---

## Advanced Parameters

Beyond temperature and top_p, additional parameters offer fine-grained control.

```{dropdown} Additional Sampling Parameters
:icon: gear
:color: info

**Penalty parameters** (reduce repetition):
- `frequency_penalty` (0.0-2.0): Penalize tokens based on frequency
- `presence_penalty` (0.0-2.0): Penalize tokens that appeared at all

**Control parameters**:
- `stop`: Stop sequences (string or list)
- `max_output_tokens`: Output length limit
- `seed`: Fixed seed for reproducibility

**Model-specific**:
- `reasoning_effort`: For reasoning models (low/medium/high)
- `logit_bias`: Adjust specific token probabilities

Most use cases only need temperature, top_p, and seed. Use advanced parameters when standard controls don't achieve desired behavior.
```

---

## Choosing Values

Use this decision tree to select parameter values based on your specific goal and requirements.

```{mermaid}
flowchart TD
    Start[Choose parameters] --> Goal{What is your goal?}
    
    Goal -->|Consistency| Low[Low temperature<br/>0.1-0.3]
    Goal -->|Balance| Med[Medium temperature<br/>0.4-0.6]
    Goal -->|Diversity| High[High temperature<br/>0.7-0.9]
    
    Low --> Repeats1{Need multiple<br/>samples?}
    Med --> Repeats2{Need multiple<br/>samples?}
    High --> Repeats3{Need multiple<br/>samples?}
    
    Repeats1 -->|No| SFT[SFT Pattern<br/>temp=0.2, repeats=1]
    Repeats1 -->|Yes| Eval[Eval Pattern<br/>temp=0.1, repeats=3]
    
    Repeats2 -->|No| RL[RL Pattern<br/>temp=0.5, repeats=1]
    Repeats2 -->|Yes| Mixed[Mixed Pattern<br/>temp=0.5, repeats=2-3]
    
    Repeats3 -->|Yes| DPO[DPO Pattern<br/>temp=0.7, repeats=3-4]
    Repeats3 -->|Very Many| Research[Research Pattern<br/>temp=0.9, repeats=5+]
```


---

## Validation

Check if your parameter choices produce intended characteristics.

```{dropdown} Expected Reward Distributions
:icon: graph
:color: info

~~~{list-table}
:header-rows: 1
:widths: 30 70

* - Configuration
  - Expected Reward Pattern
* - **Low temp (0.2)**
  - Peaked at high values (0.7-1.0), low variance
* - **Medium temp (0.5)**
  - Broad distribution (0.3-0.9), moderate variance
* - **High temp (0.9)**
  - Wide spread (0.0-1.0), high variance
~~~

```

```{dropdown} Expected Diversity
:icon: checklist
:color: info

Measure unique completions:

~~~bash
# Count unique final responses
jq -r '.output[-1].content' rollouts.jsonl | sort | uniq | wc -l
# Divide by total rollouts to get diversity ratio
~~~

**Benchmarks**:
- **Low temp (0.2)**: 30-50% unique
- **Medium temp (0.5)**: 60-80% unique
- **High temp (0.9)**: 85-95% unique
```

### Common Issues

Troubleshoot parameter configurations that produce unexpected results.

::::{tab-set}

:::{tab-item} All Rollouts Identical

**Problem**: Outputs show no variation

**Causes**:

- ❌ Temperature too low
- ❌ Seed set to fixed value

**Fix**: Try `temp ≥ 0.3`

:::

:::{tab-item} Nonsensical Outputs

**Problem**: Generated text doesn't make sense

**Causes**:

- ❌ Temperature too high
- ❌ Top-p too high

**Fix**: Try `temp ≤ 0.8` and `top_p=0.9`

:::

:::{tab-item} No Variance in Repeats

**Problem**: Repeats produce identical results

**Causes**:

- ❌ Temperature at 0.0
- ❌ Deterministic mode enabled

**Fix**: Try `temp ≥ 0.2`

:::

::::

---

## Next Steps

Now that you understand the parameters, choose your strategy:

::::{tab-set}

:::{tab-item} Training

- **{doc}`sft`** - Supervised fine-tuning strategy
- **{doc}`dpo`** - Direct preference optimization strategy
- **{doc}`rl`** - Reinforcement learning strategy

:::

:::{tab-item} Evaluation

- **{doc}`evaluation`** - Benchmarking strategy

:::

:::{tab-item} Exploration

- **{doc}`research`** - Behavioral analysis strategy

:::

::::

:::{seealso}
**Deep dive**: {doc}`validation` for metrics and validation techniques
:::
