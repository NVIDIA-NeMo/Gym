(configure-rollout-sampling)=

# Configure Sampling Parameters

Adjust temperature, top-p, and repeat settings to generate rollouts optimized for your training objective.

---

## How It Works

Different training objectives benefit from different sampling strategies:

```{list-table}
:header-rows: 1
:widths: 20 25 25 30

* - Objective
  - Temperature
  - num_repeats
  - Goal
* - **RL**
  - 0.7–1.0
  - 4–16
  - Diverse exploration for policy optimization
* - **SFT**
  - 0.0–0.3
  - 1–2
  - High-quality responses to imitate
* - **DPO**
  - 0.5–0.8
  - 2–4
  - Variance for preference pairs
```

---

## Basic Parameters

```bash
ng_collect_rollouts \
    +agent_name=your_agent \
    +input_jsonl_fpath=prompts.jsonl \
    +output_jsonl_fpath=rollouts.jsonl \
    +num_repeats=4 \
    +num_samples_in_parallel=8 \
    +responses_create_params.temperature=0.7 \
    +responses_create_params.top_p=0.9
```

### Parameter Reference

```{list-table}
:header-rows: 1
:widths: 35 15 50

* - Parameter
  - Default
  - Description
* - `+num_repeats`
  - 1
  - Rollouts per prompt (use higher for RL/DPO)
* - `+num_samples_in_parallel`
  - unlimited
  - Concurrent requests (tune based on backend capacity)
* - `+limit`
  - all
  - Max prompts to process (useful for testing)
* - `+responses_create_params.temperature`
  - model default
  - Sampling randomness (0.0 = deterministic)
* - `+responses_create_params.top_p`
  - model default
  - Nucleus sampling threshold
* - `+responses_create_params.max_output_tokens`
  - model default
  - Maximum response length
```

---

## RL Sampling

For on-policy RL, you want **diverse rollouts** that explore the action space:

```bash
ng_collect_rollouts \
    +agent_name=math_agent \
    +input_jsonl_fpath=math_prompts.jsonl \
    +output_jsonl_fpath=rl_rollouts.jsonl \
    +num_repeats=8 \
    +responses_create_params.temperature=0.8 \
    +responses_create_params.top_p=0.95
```

**Why these settings:**
- **High repeats**: Multiple attempts per prompt for reward variance
- **Higher temperature**: Encourages exploration of different strategies
- **High top_p**: Allows diverse token choices

---

## SFT Sampling

For supervised fine-tuning, you want **high-quality rollouts** to imitate:

```bash
ng_collect_rollouts \
    +agent_name=math_agent \
    +input_jsonl_fpath=math_prompts.jsonl \
    +output_jsonl_fpath=sft_rollouts.jsonl \
    +num_repeats=1 \
    +responses_create_params.temperature=0.0
```

**Post-processing**: Filter by reward score before training:

```python
# Filter to high-reward rollouts only
high_quality = [r for r in rollouts if r["reward"] >= 0.8]
```

**Why these settings:**
- **Low repeats**: You only need one good response per prompt
- **Temperature 0**: Deterministic, best-effort responses

---

## DPO Sampling

For preference learning, you need **paired rollouts** (chosen vs rejected):

```bash
ng_collect_rollouts \
    +agent_name=math_agent \
    +input_jsonl_fpath=math_prompts.jsonl \
    +output_jsonl_fpath=dpo_rollouts.jsonl \
    +num_repeats=4 \
    +responses_create_params.temperature=0.6
```

**Post-processing**: Create preference pairs:

```python
# Group by prompt, pair high/low reward
from itertools import groupby

pairs = []
for prompt_id, group in groupby(rollouts, key=lambda r: r["prompt_id"]):
    sorted_group = sorted(group, key=lambda r: r["reward"], reverse=True)
    if len(sorted_group) >= 2:
        pairs.append({
            "prompt": sorted_group[0]["input"],
            "chosen": sorted_group[0]["output"],
            "rejected": sorted_group[-1]["output"],
        })
```

**Why these settings:**
- **Multiple repeats**: Need variance to create meaningful pairs
- **Moderate temperature**: Balance between quality and diversity

