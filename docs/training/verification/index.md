(training-verification)=

# Verification for Training

Design effective reward signals and verification strategies that drive successful reinforcement learning.

:::{seealso}
For deep understanding of verification theory, see {doc}`../../about/concepts/verifying-agent-results`.
:::

---

## Choose Your Path

:::::{grid} 1 1 2 2
:gutter: 3

::::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Using Existing Resource Servers
**Most users** (90%) - Pick from 13 built-in servers

Jump to [Server Selection Guide](#server-selection-guide) below to find the right server for your task.

+++
{bdg-primary}`recommended` {bdg-secondary}`quick-start`
::::

::::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Building Custom Verification
**Advanced users** (10%) - Create custom resource servers

See {doc}`custom-patterns-cookbook` for copy-paste verification patterns.

+++
{bdg-secondary}`advanced` {bdg-secondary}`custom`
::::

::::::

---

## The verify() Method

All resource servers implement verification through a standardized `verify()` method with universal structure:

```python
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    # Your domain-specific verification logic here
    reward = calculate_reward(body.response, body.responses_create_params)
    
    return BaseVerifyResponse(
        **body.model_dump(),
        reward=reward  # Required: 0.0 to 1.0
    )
```

### Universal Aspects

What's the same across all resource servers:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Aspect
  - Details
* - **API signature**
  - `async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse`
* - **Input structure**
  - Receives task (`responses_create_params`) + agent trajectory (`response`)
* - **Output structure**
  - Returns `reward` field (required, 0.0–1.0 by convention) + optional metrics
* - **Integration**
  - Called automatically by `ng_collect_rollouts` after each agent execution
* - **Metric aggregation**
  - Any numeric field returned is automatically averaged across rollouts
```

### What Varies

Your domain-specific implementation determines:

* **Verification logic** - How you score the response (exact match, LLM judge, execution, etc.)
* **Reward type** - Binary (0.0/1.0) vs continuous (0.0–1.0 range)
* **Additional metrics** - Extra fields for tracking (accuracy, efficiency, etc.)

### Common Verification Patterns

::::{tab-set}

:::{tab-item} Binary Verification
```python
# Simple correct/incorrect
reward = 1.0 if answer == expected else 0.0
```
**Use for**: SFT data, exact match tasks, pass/fail scenarios
:::

:::{tab-item} Continuous Scoring
```python
# Partial credit for quality
reward = calculate_similarity(answer, expected)  # 0.0-1.0
```
**Use for**: DPO pairs, quality-based training, nuanced evaluation
:::

:::{tab-item} Multi-Objective
```python
# Balance multiple goals
reward = 0.6 * correctness + 0.3 * efficiency + 0.1 * style
```
**Use for**: Complex tasks, multiple criteria, weighted objectives
:::

::::

---

## Server Selection Guide

Choose verification based on your task and training algorithm.

### By Task Type

```{list-table}
:header-rows: 1
:widths: 30 35 35

* - Task Type
  - Recommended Server
  - Reward Pattern
* - **Multiple Choice (A/B/C/D)**
  - mcqa
  - Binary (0.0 or 1.0)
* - **Math Problems**
  - library_judge_math
  - Continuous (symbolic + judge)
* - **Code Generation**
  - comp_coding
  - Binary (tests pass/fail)
* - **Open-Ended QA**
  - equivalence_llm_judge
  - Continuous (semantic equivalence)
* - **Instruction Following**
  - instruction_following
  - Binary (constraints met/not)
* - **JSON/Structured Output**
  - structured_outputs
  - Binary (schema valid/invalid)
* - **Multi-Needle Extraction**
  - multineedle
  - Binary + multi-metric
* - **Python Code for Math**
  - python_math_exec
  - Binary (correct result)
```

### By Training Algorithm

```{list-table}
:header-rows: 1
:widths: 25 40 35

* - Algorithm
  - Reward Needs
  - Recommended Servers
* - **SFT**
  - Binary or high-threshold continuous
  - mcqa, comp_coding, instruction_following
* - **DPO**
  - Continuous with clear separation (≥0.2 gap)
  - library_judge_math, equivalence_llm_judge
* - **PPO/RL**
  - Continuous with rich signal
  - multineedle, library_judge_math
```

### Reward Patterns Explained

::::{tab-set}

:::{tab-item} Binary (0.0 or 1.0)
**Characteristics**: Clear pass/fail, high confidence

**Servers**: mcqa, comp_coding, instruction_following, python_math_exec

**Best for**: SFT data filtering

**Training implications**: Clean datasets, but limited signal for DPO/PPO
:::

:::{tab-item} Continuous (0.0–1.0)
**Characteristics**: Nuanced quality, partial credit, rich gradients

**Servers**: library_judge_math, equivalence_llm_judge, multineedle

**Best for**: DPO pairs, PPO training

**Training implications**: More sophisticated strategies possible
:::

:::{tab-item} Multi-Metric
**Characteristics**: Primary reward + additional tracking metrics

**Servers**: multineedle (accuracy + set_overlap), library_judge_math (library + judge)

**Best for**: Multi-objective optimization, analysis

**Training implications**: Track trade-offs, filter by multiple criteria
:::

::::

**Next**: After selecting a server, prepare data with {doc}`../datasets/prepare-for-training`.

---

## Advanced Topics

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`code-square;1.5em;sd-mr-1` Custom Patterns Cookbook
:link: custom-patterns-cookbook
:link-type: doc

**Copy-paste patterns** for building custom resource servers. Six common verification patterns with complete code examples.
+++
{bdg-secondary}`advanced` {bdg-secondary}`patterns` {bdg-secondary}`cookbook`
:::

:::{grid-item-card} {octicon}`versions;1.5em;sd-mr-1` Multi-Objective Scoring
:link: multi-objective-scoring
:link-type: doc

**Advanced guide** for combining multiple reward signals or balancing conflicting objectives in custom verification.
+++
{bdg-secondary}`advanced` {bdg-secondary}`multi-objective` {bdg-secondary}`composite`
:::

::::


## Verification in the Training Pipeline

```
Rollout Generation
    ↓
[Agent Execution]        ← Tools, reasoning, response
    ↓
[Verification]           ← Score performance
    ↓
    Reward Signal        → to training framework
```

**Flow**: Generated rollout → Verification function → Reward score → Training data

**Next**: {doc}`../data-quality/index` for filtering by reward thresholds


## Next Steps

:::{button-ref} ../datasets/prepare-for-training
:color: primary
:outline:
:ref-type: doc

Prepare Rollouts for Training →
:::

:::{tip}
**Building custom verification?** See {doc}`custom-patterns-cookbook` for copy-paste patterns.

**Already have rollouts?** See {doc}`../datasets/prepare-for-training` to convert them into training-ready datasets.
:::

```{toctree}
:hidden:
:maxdepth: 1

custom-patterns-cookbook
multi-objective-scoring
```

