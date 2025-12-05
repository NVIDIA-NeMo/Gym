(tutorial-train-with-trl)=

# Train with TRL

In [Collecting Rollouts](../get-started/rollout-collection.md), you generated scored interactions between your agent and environment. Now you'll use those rollouts to actually train a model using Hugging Face's TRL library.

:::{card}

**Goal**: Train your first model using Gym rollouts and TRL.

^^^

**In this tutorial, you will**:

1. Format rollouts for SFT training
2. Configure TRL's SFTTrainer
3. Run a training loop
4. Validate the trained model

:::

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← Get Started
:::

---

## Before You Begin

Make sure you have:

- ✅ Completed [Collecting Rollouts](../get-started/rollout-collection.md)
- ✅ Rollouts file at `results/rollouts.jsonl`
- ✅ TRL installed (`pip install trl`)
- ✅ GPU with sufficient memory for your base model

**What you'll build**: A fine-tuned model checkpoint that incorporates the successful behaviors from your rollouts.

---

## 1. Inspect Your Rollouts

<!-- SME: Show what the rollout data looks like and what fields are relevant for training -->

First, examine the rollouts you collected:

```bash
# TODO: Command to inspect rollout structure
```

Each rollout contains:

- **input**: The original prompt/conversation
- **output**: The model's response (including tool calls)
- **reward**: Verification score (0.0–1.0)

**✅ Success Check**: You should see <!-- SME: What should they see? -->

---

## 2. Filter High-Quality Rollouts

<!-- SME: Show how to filter rollouts by reward score for SFT -->

For SFT training, you typically want only successful rollouts:

```bash
# TODO: Command or script to filter rollouts with reward >= threshold
```

```{list-table} Filter Parameters
:header-rows: 1
:widths: 30 15 55

* - Parameter
  - Type
  - Description
* - `reward_threshold`
  - `float`
  - Minimum reward score to include (e.g., `0.8`)
* - <!-- SME: Add other relevant parameters -->
  - 
  - 
```

**✅ Success Check**: You should have <!-- SME: Expected output -->

---

## 3. Format for TRL

<!-- SME: Show the conversion from Gym rollout format to TRL's expected format -->

TRL expects data in a specific format. Convert your filtered rollouts:

```python
# TODO: Python snippet showing format conversion
# from gym_rollout_format to trl_training_format
```

The key fields TRL needs:

- **prompt**: <!-- SME: Describe -->
- **completion**: <!-- SME: Describe -->
- <!-- SME: Other fields? -->

**✅ Success Check**: Your formatted data should <!-- SME: validation criteria -->

---

## 4. Configure Training

<!-- SME: Show TRL SFTTrainer configuration -->

Create your training configuration:

```python
from trl import SFTTrainer, SFTConfig

# TODO: Training configuration
config = SFTConfig(
    # SME: Fill in recommended defaults
    output_dir="./trained_model",
    # ...
)
```

```{list-table} Key Configuration Options
:header-rows: 1
:widths: 30 15 55

* - Parameter
  - Default
  - Description
* - `output_dir`
  - (required)
  - Where to save checkpoints
* - <!-- SME: Add key parameters -->
  - 
  - 
```

---

## 5. Run Training

<!-- SME: Show how to launch training -->

Start the training loop:

```python
# TODO: Training execution code
trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=formatted_data,
    # ...
)
trainer.train()
```

**✅ Success Check**: You should see training progress:

```text
# TODO: Example training output
```

---

## 6. Validate with Gym

<!-- SME: Show how to test the trained model back in Gym -->

Test your trained model by running it through Gym:

```bash
# TODO: Command to run Gym with the new model checkpoint
```

Compare the results:

- **Before training**: <!-- SME: baseline metrics -->
- **After training**: <!-- SME: expected improvement -->

**✅ Success Check**: Your trained model should <!-- SME: success criteria -->

---

## Training Objectives Reference

::::{tab-set}

:::{tab-item} SFT

**When to use**: You have high-quality rollouts and want to imitate successful behavior.

```python
# TODO: SFT-specific configuration
```

:::

:::{tab-item} DPO

**When to use**: You have paired rollouts (good vs. bad) for the same prompt.

```python
# TODO: DPO-specific configuration
```

:::

:::{tab-item} GRPO

**When to use**: You have multiple rollouts per prompt with varying rewards.

```python
# TODO: GRPO-specific configuration
```

:::

::::

---

## Troubleshooting

<!-- SME: Add common issues and solutions -->

:::{dropdown} Out of memory during training
<!-- SME: Solutions -->
:::

:::{dropdown} Training loss not decreasing
<!-- SME: Solutions -->
:::

:::{dropdown} Model performance worse after training
<!-- SME: Solutions -->
:::

---

## What's Next?

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Iterate on Training Data
:link: /get-started/rollout-collection
:link-type: doc

Collect more rollouts with your improved model to continue the training loop.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Scale Training
:link: train-with-nemo-rl
:link-type: doc

For larger models or multi-node training, try NeMo RL.
:::

::::

