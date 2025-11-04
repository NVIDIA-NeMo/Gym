(training-datasets)=

# Dataset Management for Training

Organize, validate, and prepare training datasets in formats compatible with RL frameworks. Learn dataset structure, format validation, and conversion for SFT, DPO, and RL training.

Your rollouts are curated—now prepare them in the correct format for your training framework.


## When You Need This

Use this section when you need to:

* **Prepare data for training** - Convert rollouts to SFT, DPO, or RL formats
* **Validate dataset format** - Check compatibility with training frameworks
* **Organize datasets** - Structure files for versioning and reproducibility
* **Understand format specifications** - Learn rollout JSON schema and requirements

:::{seealso}
For training framework integration, see {doc}`../integration/index` after preparing your datasets.
:::


## Guides and References

::::{grid} 1 1 1 2
:gutter: 3

:::{grid-item-card} {octicon}`package-dependencies;1.5em;sd-mr-1` Prepare for Training
:link: prepare-for-training
:link-type: doc

**How-to guide** for converting rollouts to SFT, DPO, and RL training formats compatible with popular frameworks.
+++
{bdg-secondary}`how-to` {bdg-secondary}`sft` {bdg-secondary}`dpo` {bdg-secondary}`rl`
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Validate Format
:link: validate-format
:link-type: doc

**How-to guide** for validating dataset format using `ng_prepare_data` and checking framework compatibility.
+++
{bdg-secondary}`how-to` {bdg-secondary}`validation` {bdg-secondary}`ng_prepare_data`
:::

:::{grid-item-card} {octicon}`file-code;1.5em;sd-mr-1` Format Specification
:link: format-specification
:link-type: doc

**Reference** for rollout JSON schema, field definitions, and format requirements for different training types.
+++
{bdg-secondary}`reference` {bdg-secondary}`schema` {bdg-secondary}`spec`
:::

::::


## Dataset Formats by Training Type

```{list-table}
:header-rows: 1
:widths: 25 40 35

* - Training Type
  - Output Format
  - Key Requirements
* - **SFT (Supervised Fine-Tuning)**
  - Conversation history
  - High-quality examples (reward ≥ 0.8)
* - **DPO (Direct Preference Optimization)**
  - Preference pairs (chosen/rejected)
  - Quality difference ≥ 0.1
* - **RL (Reinforcement Learning)**
  - Rollouts with rewards
  - Diverse quality range
```

See {doc}`prepare-for-training` for conversion examples.


## Dataset Pipeline

```
Curated Rollouts (JSONL)
    ↓
[1. Choose Format]       ← SFT, DPO, or RL?
    ↓
[2. Convert Format]      ← prepare-for-training guide
    ↓
[3. Validate]            ← ng_prepare_data command
    ↓
Training-Ready Data      → to RL framework integration
```

**Previous**: {doc}`../data-quality/index` for curation  
**Next**: {doc}`../integration/index` for framework integration


## Common Dataset Operations

### Convert to SFT Format
```python
# Extract conversation history
sft_data = [{
    "messages": rollout['output'],
    "reward": rollout['reward']
} for rollout in rollouts if rollout['reward'] >= 0.8]
```

### Create DPO Pairs
```python
# Compare multiple rollouts per task
pairs = [{
    "prompt": task['input'],
    "chosen": high_reward_rollout['output'],
    "rejected": low_reward_rollout['output'],
    "quality_difference": high_reward - low_reward
} for task, rollouts in grouped_rollouts]
```

### Prepare for RL
```python
# Include full rollout with rewards
rl_data = [{
    "responses_create_params": rollout['responses_create_params'],
    "output": rollout['output'],
    "reward": rollout['reward']
} for rollout in rollouts]
```

See {doc}`prepare-for-training` for complete implementations.


## Validation Workflow

Use `ng_prepare_data` to validate and analyze datasets:

```bash
ng_prepare_data \
  +mode=train_preparation \
  +config_paths=[your_config.yaml] \
  +output_dirpath=./training_data
```

**Checks performed**:
- Format compliance (Responses API schema)
- Aggregate statistics (lengths, turns, diversity)
- Missing or invalid fields
- Compatibility with training frameworks

See {doc}`validate-format` for detailed usage.


## Dataset Organization Best Practices

```
training_data/
├── v1.0/                       # Version your datasets
│   ├── sft_data.jsonl
│   ├── dpo_pairs.jsonl
│   └── statistics.json
├── v1.1/
│   ├── sft_data.jsonl         # Improved curation
│   └── statistics.json
└── README.md                   # Document changes
```

**Benefits**: Reproducibility, version tracking, easy rollback


## Related Topics

### Data Pipeline

* {doc}`../data-quality/index` - Filter and curate before formatting
* {doc}`../rollout-collection/index` - Generate raw rollouts
* {doc}`../integration/index` - Use formatted data with frameworks

### Framework-Specific Formats

* {doc}`../integration/nemo-rl` - NeMo-RL format requirements
* {doc}`../integration/verl` - VeRL format requirements
* {doc}`../integration/openrlhf` - OpenRLHF format requirements
* {doc}`../integration/trl` - TRL format requirements


## Next Steps

:::{button-ref} prepare-for-training
:color: primary
:outline:
:ref-type: doc

Start with Prepare for Training →
:::

:::{tip}
**Not sure which format you need?** Start with {doc}`../integration/framework-comparison` to understand your framework's requirements, then return here for conversion guidance.
:::

```{toctree}
:hidden:
:maxdepth: 1

prepare-for-training
validate-format
format-specification
```

