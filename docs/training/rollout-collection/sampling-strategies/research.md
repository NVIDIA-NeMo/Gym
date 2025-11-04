(training-rollout-sampling-research)=

# Research Sampling Strategy

Configure for behavioral exploration: discovering capabilities, failure modes, and edge cases.

:::{card}

**Task**: Explore agent capabilities through diverse rollouts to identify behavioral patterns and improvement opportunities.

^^^

**This guide shows you how to**:

1. Configure high-diversity sampling for broad exploration
2. Generate multiple attempts per task to observe variance
3. Analyze failure modes and edge case behaviors
4. Identify patterns that inform agent improvements

:::

---

## Before You Start

Ensure you have these prerequisites before exploring agent behavior:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Requirement
  - Details
* - **Get Started completed**
  - Complete {doc}`../../../get-started/collecting-rollouts` first
* - **Servers running**
  - Agent and model servers collecting rollouts successfully
* - **Research goal**
  - Clear questions about agent behavior (failure modes, capabilities, etc.)
* - **Task dataset**
  - Small diverse sample (50-200 tasks) for rapid experimentation
* - **Analysis plan**
  - Method for inspecting rollouts (scripts, manual review, metrics)
```

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← New? Try Get Started
:::

---

## Explore Agent Behavior

Generate diverse rollouts to discover capabilities, failure modes, and behavioral patterns.

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=research_tasks.jsonl \
    +output_jsonl_fpath=research_exploration.jsonl \
    +responses_create_params.temperature=<temperature> \
    +num_repeats=<num_repeats> \
    +limit=<limit> \
    +num_samples_in_parallel=<parallelism>
```

**Configuration**: For research, use high temperature for diversity, many repeats (5-10) to observe variance, small sample sizes for rapid iteration, and moderate parallelism. Refer to {doc}`parameters` for parameter explanations.

---

## Analysis Workflows

### Interactive Inspection

```bash
# Launch interactive viewer
ng_viewer +input_jsonl_fpath=research_exploration.jsonl
```

Navigate through rollouts to spot interesting patterns.

### Identify Failure Modes

```bash
# Extract low-reward rollouts
jq 'select(.reward < 0.3)' research_exploration.jsonl > failures.jsonl

echo "Failure rate: $(wc -l < failures.jsonl) / 250"

# Manually inspect first failure
head -1 failures.jsonl | jq '.output'
```

**Look for**:
- Common error patterns
- Tool misuse
- Reasoning failures

### Discover Successful Strategies

```bash
# Find high-reward outliers
jq 'select(.reward > 0.9)' research_exploration.jsonl > successes.jsonl

# Compare tool usage in successes
jq '.output[] | select(.type=="function_call") | .name' successes.jsonl | \
  sort | uniq -c | sort -rn
```

---

## Research Questions

```{dropdown} What tools does the agent prefer?
:icon: tools
:color: info

~~~bash
jq '.output[] | select(.type=="function_call") | .name' research_exploration.jsonl | \
  sort | uniq -c | sort -rn
~~~

**Example output**:
~~~
    187 calculator
     43 search
     12 code_interpreter
      8 browser
~~~

**Insight**: Agent heavily prefers calculator, underutilizes other tools.

```

```{dropdown} What reasoning patterns emerge?
:icon: comment-discussion
:color: info

Manually review rollouts to identify patterns:

~~~bash
# Sample 10 random high-reward rollouts
jq 'select(.reward >= 0.8)' research_exploration.jsonl | shuf -n 10 > sample_good.jsonl

# Review each
for i in {1..10}; do
    echo "=== Sample $i ==="
    head -n $i sample_good.jsonl | tail -n 1 | jq '.output[] | select(.type=="message") | .content'
    echo ""
done
~~~

```

---

## Use Cases

::::{tab-set}

:::{tab-item} Debugging Agent Prompts

Test prompt variations with high diversity:

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=prompt_variants.jsonl \
    +output_jsonl_fpath=prompt_exploration.jsonl \
    +responses_create_params.temperature=0.9 \
    +num_repeats=3 \
    +limit=20

# Compare success rates
jq -r '.prompt_variant + "," + (.reward | tostring)' prompt_exploration.jsonl | \
  awk -F',' '{sum[$1]+=$2; count[$1]++} END {for (v in sum) print v, sum[v]/count[v]}'
```

:::

:::{tab-item} Capability Discovery

Test on diverse, challenging tasks:

```bash
# Collect on hard/edge cases
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=edge_cases.jsonl \
    +output_jsonl_fpath=capability_test.jsonl \
    +responses_create_params.temperature=0.9 \
    +num_repeats=5 \
    +limit=30

# Analyze which types succeed
jq -r '.task_type + "," + (.reward | tostring)' capability_test.jsonl | \
  awk -F',' '{sum[$1]+=$2; count[$1]++} END {for (type in sum) print type, sum[type]/count[type]}'
```

:::

:::{tab-item} Failure Analysis

Deep dive into specific failure mode:

```bash
# Filter for specific failure pattern
jq 'select(.reward < 0.2 and (.output | map(select(.type=="function_call")) | length) == 0)' \
  research_exploration.jsonl > no_tool_failures.jsonl

echo "Tasks where agent failed to use any tool: $(wc -l < no_tool_failures.jsonl)"

# Examine first few
head -3 no_tool_failures.jsonl | jq '.responses_create_params.input[1].content'
```

:::

::::
