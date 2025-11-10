(training-integration)=

# Training Framework Integration

Detailed integration guides for using NeMo Gym rollouts with popular RL training frameworks.

## Available Integration Guides

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` NeMo RL (Complete Guide)
:link: nemo-rl
:link-type: doc

NVIDIA's post-training framework with GRPO, SFT, and DPO support. Includes complete data transformation scripts and configuration examples.
+++
{bdg-primary}`Official` {bdg-secondary}`GRPO` {bdg-secondary}`SFT` {bdg-secondary}`DPO`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Quick Reference
:link: quick-reference
:link-type: doc

Fast lookup for transformations, configs, and commands. Perfect cheat sheet for repeat integrations.
+++
{bdg-info}`Cheat Sheet`
:::

::::

## Coming Soon

Additional integration guides for:
- VeRL
- OpenRLHF  
- TRL (HuggingFace)
- Custom frameworks

See {doc}`../handoff-to-training` for framework overview and basic examples.

---

## Integration Quick Reference

```{list-table}
:header-rows: 1
:widths: 30 30 40

* - Framework
  - Algorithms
  - Status
* - **NeMo RL**
  - GRPO, SFT, DPO, RM
  - {doc}`Full Guide Available <nemo-rl/index>`
* - **VeRL**
  - PPO, GRPO
  - Overview only
* - **OpenRLHF**
  - PPO, DPO
  - Overview only
* - **TRL**
  - DPO, PPO
  - Overview only
```

---

## Choosing a Framework

```{list-table}
:header-rows: 1
:widths: 25 35 40

* - Use Case
  - Recommended Framework
  - Why
* - **NVIDIA ecosystem**
  - NeMo RL
  - Native Megatron Core support, vLLM integration, multi-node scaling
* - **HuggingFace models**
  - TRL or NeMo RL
  - Both support HF models; TRL simpler for small models
* - **Research / flexibility**
  - VeRL or OpenRLHF
  - Open source, active development, community support
* - **Custom requirements**
  - Any (all support JSONL)
  - Use framework best suited to your infrastructure
```

---

## Common Integration Patterns

All frameworks follow similar patterns when integrating with NeMo Gym:

### 1. Data Transformation

```python
# Load NeMo Gym rollouts
rollouts = [json.loads(line) for line in open('rollouts.jsonl')]

# Transform to framework format
training_data = transform_rollouts(rollouts, target_format="framework_name")

# Write framework-compatible JSONL
write_jsonl(training_data, 'training_data.jsonl')
```

### 2. Configuration

```yaml
# Framework config (YAML or Python)
data:
  train_path: "training_data.jsonl"
  input_key: "input"      # Field names vary by framework
  output_key: "output"

model:
  name: "Qwen/Qwen2.5-1.5B"
  
training:
  algorithm: "grpo"  # or sft, dpo, etc.
```

### 3. Training Launch

```bash
# Most frameworks use similar command structure
framework_train --config config.yaml
```

See {doc}`nemo-rl/index` for a complete end-to-end example.

---

## Field Mapping Guide

Different frameworks expect different field names. Here's how NeMo Gym fields map:

```{list-table}
:header-rows: 1
:widths: 30 30 40

* - NeMo Gym Field
  - Common Framework Names
  - Description
* - `responses_create_params.input`
  - `input`, `prompt`, `messages`, `context`
  - Original user prompt
* - `output`
  - `output`, `response`, `completion`, `assistant_message`
  - Agent's response
* - `reward`
  - `reward`, `score`, `quality`, `preference`
  - Verification score
```

**Example transformations**:

```python
# For framework expecting "prompt" and "completion"
transformed = {
    "prompt": extract_input(rollout["responses_create_params"]["input"]),
    "completion": extract_output(rollout["output"]),
    "score": rollout["reward"]
}

# For framework expecting "messages" format
transformed = {
    "messages": convert_to_messages(rollout)
}
```

---

## Data Quality Validation

Before training with any framework, validate your transformed data:

```python
import json
from collections import Counter

# Load transformed data
data = [json.loads(line) for line in open('training_data.jsonl')]

print(f"Total examples: {len(data)}")

# Check required fields
required_fields = ["input", "output"]  # Adjust for your framework
for i, item in enumerate(data[:10]):
    missing = [f for f in required_fields if f not in item]
    if missing:
        print(f"❌ Example {i} missing fields: {missing}")

# Check for empty values
empty_inputs = sum(1 for item in data if not item.get("input"))
empty_outputs = sum(1 for item in data if not item.get("output"))
print(f"Empty inputs: {empty_inputs}, Empty outputs: {empty_outputs}")

# Distribution checks
if "reward" in data[0]:
    rewards = [item["reward"] for item in data]
    print(f"Reward range: [{min(rewards):.3f}, {max(rewards):.3f}]")
    print(f"Mean reward: {sum(rewards)/len(rewards):.3f}")
```

---

## Framework-Specific Resources

### NeMo RL
- **Repository**: [github.com/NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL)
- **Integration**: {doc}`nemo-rl/index` (complete guide)
- **Best for**: Multi-node training, Megatron models, NVIDIA GPUs

### VeRL
- **Repository**: [github.com/volcengine/verl](https://github.com/volcengine/verl)
- **Integration**: Overview only (see VeRL section above)
- **Best for**: High-performance PPO, research flexibility

### OpenRLHF
- **Repository**: [github.com/OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- **Integration**: Overview only (see OpenRLHF section above)
- **Best for**: Multiple algorithm support, flexible deployment

### TRL
- **Repository**: [github.com/huggingface/trl](https://github.com/huggingface/trl)
- **Documentation**: [huggingface.co/docs/trl](https://huggingface.co/docs/trl/)
- **Integration**: Overview only (see TRL section above)
- **Best for**: HuggingFace ecosystem, quick prototyping

---

## Contributing Integration Guides

Want to add an integration guide for another framework?

1. Follow the structure in {doc}`nemo-rl/index`:
   - Overview and prerequisites
   - Data transformation with scripts
   - Configuration examples
   - Training launch commands
   - Troubleshooting section

2. Include:
   - Complete transformation scripts (not pseudocode)
   - Real configuration files
   - Common error patterns and solutions
   - Performance optimization tips

3. Submit a pull request with:
   - New guide in `docs/training/integration/`
   - Update to this index page
   - Example rollouts → transformed data

See [CONTRIBUTING.md](https://github.com/NVIDIA-NeMo/Gym/blob/main/CONTRIBUTING.md) for guidelines.

---

## Related Topics

- {doc}`../datasets/prepare-for-training` - Data preparation by algorithm
- {doc}`../rollout-collection/index` - Generate training data
- {doc}`../data-quality/index` - Filter and curate rollouts

```{toctree}
:hidden:
:maxdepth: 2

nemo-rl/index
quick-reference
```
