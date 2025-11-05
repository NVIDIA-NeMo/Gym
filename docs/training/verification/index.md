(training-verification)=

# Verification for Training

Verification scores agent performance and generates reward signals that drive reinforcement learning. Every resource server evaluates agent responses and returns numerical rewards (0.0-1.0) indicating quality.

:::{seealso}
For deep understanding of verification theory, see {doc}`../../about/concepts/verifying-agent-results`.
:::

---

## Choose Your Path

:::::{grid} 1 1 2 2
:gutter: 3

::::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Using Existing Servers
**Recommended** for 90% of users

Pick from 13 built-in resource servers covering MCQA, math, code generation, open-ended QA, and more.

+++
{bdg-primary}`Most users start here`

:::{button-ref} server-selection
:color: primary
:outline:
:ref-type: doc

Server Selection Guide →
:::
::::

::::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Building Custom Verification
**Advanced** for specialized domains

Create custom resource servers with copy-paste verification patterns.

+++
{bdg-secondary}`Advanced users`

:::{button-ref} custom-patterns-cookbook
:color: secondary
:outline:
:ref-type: doc

Custom Patterns Cookbook →
:::
::::

:::::

---

## Quick Server Picker

Choose based on your task:

```{list-table}
:header-rows: 1
:widths: 40 30 30

* - Your Task
  - Use This Server
  - Guide
* - **Multiple choice (A/B/C/D)**
  - mcqa
  - {ref}`Binary, SFT <training-verification-server-selection>`
* - **Math problems**
  - library_judge_math
  - {ref}`Continuous, DPO/PPO <training-verification-server-selection>`
* - **Code generation**
  - comp_coding
  - {ref}`Binary, SFT <training-verification-server-selection>`
* - **Open-ended QA**
  - equivalence_llm_judge
  - {ref}`Continuous, DPO/PPO <training-verification-server-selection>`
* - **Other tasks**
  - See full guide →
  - {doc}`server-selection`
```

---

## What Verification Provides

**Input**: Complete agent interaction
- Original task/prompt
- Tools called by agent
- Final response generated

**Output**: Reward signal
- Primary `reward` field (0.0-1.0)
- Optional additional metrics
- Automatic aggregation across rollouts

**Purpose**: Creates training signal for:
- **SFT** - Filter for correct examples
- **DPO** - Create preference pairs
- **PPO** - Provide gradient signals

---

## Workflow

```
1. Select Server → Use server-selection.md
   ↓
2. Validate Choice → Use validate-verification.md  
   ↓
3. Collect Rollouts → See rollout-collection/
   ↓
4. Prepare Data → See datasets/prepare-for-training.md
   ↓
5. Train → See integration/
```

---

## Advanced Topics

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`code-square;1.5em;sd-mr-1` Custom Patterns
:link: custom-patterns-cookbook
:link-type: doc

Copy-paste verification patterns for building custom resource servers. Six patterns with complete code examples.
+++
{bdg-secondary}`Advanced`
:::

:::{grid-item-card} {octicon}`versions;1.5em;sd-mr-1` Multi-Objective
:link: multi-objective-scoring
:link-type: doc

Combine multiple reward signals or balance conflicting objectives in custom verification.
+++
{bdg-secondary}`Advanced`
:::

::::

---

## Next Steps

:::{button-ref} server-selection
:color: primary
:outline:
:ref-type: doc

Pick a Server →
:::

:::{button-ref} validate-verification
:color: secondary
:outline:
:ref-type: doc

Validate Your Choice →
:::

```{toctree}
:hidden:
:maxdepth: 1

server-selection
validate-verification
custom-patterns-cookbook
multi-objective-scoring
```
