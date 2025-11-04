(training-integration)=

# RL Framework Integration

Connect your training data to reinforcement learning frameworks. This section provides integration guides for NeMo-RL, VeRL, OpenRLHF, TRL, and custom frameworks.

Your data is prepared—now integrate with your chosen RL training framework.

## When You Need This

Use this section when you need to:

* **Connect to RL frameworks** - Integrate rollouts with NeMo-RL, VeRL, OpenRLHF, TRL, or others
* **Configure training pipelines** - Set up end-to-end workflows from rollouts to trained models
* **Compare frameworks** - Understand requirements and tradeoffs between different RL frameworks
* **Implement custom integration** - Connect to proprietary or custom training systems

:::{tip}
**New to RL training?** Start with {doc}`framework-comparison` to understand which framework fits your needs, then follow the framework-specific guide.
:::

## Supported RL Frameworks

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` NeMo-RL
:link: nemo-rl
:link-type: doc

**NVIDIA's RL framework** integrated with NeMo ecosystem. Enterprise support, optimized for NVIDIA infrastructure.
+++
{bdg-secondary}`nvidia` {bdg-secondary}`enterprise` {bdg-secondary}`ppo` {bdg-secondary}`dpo`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` VeRL
:link: verl
:link-type: doc

**High-performance RL framework** with advanced algorithms and efficient implementations for large-scale training.
+++
{bdg-secondary}`performance` {bdg-secondary}`scalable` {bdg-secondary}`ppo`
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` OpenRLHF
:link: openrlhf
:link-type: doc

**Open-source RLHF framework** supporting PPO, DPO, and other preference learning algorithms with flexible configurations.
+++
{bdg-secondary}`open-source` {bdg-secondary}`flexible` {bdg-secondary}`rlhf`
:::

:::{grid-item-card} {octicon}`hubot;1.5em;sd-mr-1` TRL (HuggingFace)
:link: trl
:link-type: doc

**Transformer Reinforcement Learning** from Hugging Face. Easy integration with Transformers library and model hub.
+++
{bdg-secondary}`huggingface` {bdg-secondary}`transformers` {bdg-secondary}`dpo`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Custom Frameworks
:link: custom-frameworks
:link-type: doc

**Integration patterns** for custom or proprietary RL training systems. Learn data format requirements and handoff patterns.
+++
{bdg-secondary}`custom` {bdg-secondary}`patterns` {bdg-secondary}`flexible`
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Framework Comparison
:link: framework-comparison
:link-type: doc

**Reference** comparing requirements, features, and trade-offs to help you choose the right framework.
+++
{bdg-secondary}`reference` {bdg-secondary}`comparison` {bdg-secondary}`decision-guide`
:::

::::


## Quick Framework Selection

Choose based on your environment and requirements:

```{list-table}
:header-rows: 1
:widths: 30 40 30

* - If You Have...
  - Recommended Framework
  - Why
* - NVIDIA infrastructure + support contract
  - {doc}`NeMo-RL <nemo-rl>`
  - Enterprise support, optimized for NVIDIA
* - Need maximum performance at scale
  - {doc}`VeRL <verl>`
  - High-throughput, efficient implementations
* - Want flexibility and open-source
  - {doc}`OpenRLHF <openrlhf>`
  - Flexible configurations, active community
* - Already using HuggingFace ecosystem
  - {doc}`TRL <trl>`
  - Easy Transformers integration
* - Custom/proprietary training system
  - {doc}`Custom Integration <custom-frameworks>`
  - Flexible patterns for any framework
```

See {doc}`framework-comparison` for detailed comparison.


## Integration Pipeline

```
Training-Ready Datasets
    ↓
[1. Choose Framework]    ← NeMo-RL, VeRL, OpenRLHF, TRL, custom
    ↓
[2. Configure Pipeline]  ← Training config, model config, data paths
    ↓
[3. Launch Training]     ← Framework-specific training command
    ↓
Trained Model Checkpoint
```

**Previous**: {doc}`../datasets/index` for data preparation


## Common Integration Patterns

### Pattern 1: Direct Data Handoff
```python
# NeMo Gym → Framework data format
# Example: NeMo-RL
train_data = load_rollouts('sft_data.jsonl')
trainer = NeMoRLTrainer(config=training_config)
trainer.train(train_data)
```

**Use for**: Simple pipelines, single-step integration

### Pattern 2: Continuous Generation
```python
# Generate → Train loop
while training:
    new_rollouts = generate_rollouts(current_policy)
    trainer.update(new_rollouts)
```

**Use for**: Online RL, iterative improvement

### Pattern 3: Staged Pipeline
```python
# Multi-stage workflow
# 1. Generate rollouts
generate_batch(size=10000)

# 2. Filter and prepare
filtered = filter_quality(rollouts)
formatted = prepare_for_training(filtered)

# 3. Train
train_with_framework(formatted)
```

**Use for**: Production workflows, quality control

See framework-specific guides for implementations.


## Data Format Requirements

Each framework expects specific data formats:

```{list-table}
:header-rows: 1
:widths: 25 40 35

* - Framework
  - Expected Format
  - Key Fields
* - **NeMo-RL**
  - JSONL with conversation history
  - `messages`, `reward`, `metadata`
* - **VeRL**
  - Trajectory format with states
  - `states`, `actions`, `rewards`
* - **OpenRLHF**
  - Flexible (supports multiple)
  - Configurable in framework config
* - **TRL**
  - HuggingFace dataset format
  - `prompt`, `chosen`, `rejected` (DPO)
```

See {doc}`../datasets/format-specification` and framework guides for details.


## End-to-End Example

```bash
# 1. Generate rollouts
ng_collect_rollouts \
  +agent_name=my_agent \
  +input_jsonl_fpath=tasks.jsonl \
  +output_jsonl_fpath=rollouts.jsonl

# 2. Filter and prepare
python filter_rollouts.py \
  --input rollouts.jsonl \
  --output filtered.jsonl \
  --min_reward 0.7

python prepare_for_nemo_rl.py \
  --input filtered.jsonl \
  --output training_data.jsonl

# 3. Launch training (NeMo-RL example)
python train_nemo_rl.py \
  --data_path training_data.jsonl \
  --config nemo_rl_config.yaml
```

See framework-specific guides for complete workflows.


## Related Topics

### Data Pipeline

* {doc}`../rollout-collection/index` - Generate training rollouts
* {doc}`../data-quality/index` - Filter and curate data
* {doc}`../datasets/index` - Format for framework compatibility

### Configuration

* **Configuration Management** *(coming soon)* - System configuration for training pipelines


## Next Steps

Not sure which framework to use?

:::{button-ref} framework-comparison
:color: primary
:outline:
:ref-type: doc

Start with Framework Comparison →
:::
