(training-nemo-rl-grpo-gym-configuration)=

# Gym Configuration

Before running GRPO training, you need to configure how NeMo RL connects to NeMo Gym. The training config file contains Gym-specific parameters that control data loading, environment interaction, and validation.

:::{card}

**Goal**: Understand the Gym configuration parameters for RL training.

**Time**: ~10 minutes (read)

^^^

**In this section, you will learn**:

1. How to configure data paths for training and validation
2. How to enable and configure NeMo Gym in NeMo RL

:::

:::{button-ref} training-nemo-rl-grpo-about-workplace-assistant
:color: secondary
:outline:
:ref-type: ref

← Previous: About Workplace Assistant
:::

## Prerequisites

- Read {doc}`About Workplace Assistant <about-workplace-assistant>` to understand the training environment

---

## Configuration File Location

The full training configuration file is located at:

```
examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml
```

---

## Gym Configuration Sections

There are two Gym-specific sections in the NeMo RL training config: `data` and `env`.

### Data Section

```yaml
data:
  train_jsonl_fpath: 3rdparty/Gym-workspace/Gym/data/workplace_assistant/train.jsonl
  validation_jsonl_fpath: 3rdparty/Gym-workspace/Gym/data/workplace_assistant/validation.jsonl
```

| Parameter | Description |
|-----------|-------------|
| `train_jsonl_fpath` | Path to training dataset (prepared in {doc}`Setup <setup>`) |
| `validation_jsonl_fpath` | Path to validation dataset |

### Environment Section

```yaml
env:
  should_use_nemo_gym: true
  nemo_gym:
    config_paths:
    - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
    - resources_servers/workplace_assistant/configs/workplace_assistant.yaml
    workplace_assistant_simple_agent:
      responses_api_agents:
        simple_agent:
          max_steps: 6
```

| Parameter | Description |
|-----------|-------------|
| `should_use_nemo_gym` | Set to `true` to enable Gym |
| `nemo_gym` | Everything under this key is a Gym config |
| `nemo_gym.config_paths` | Gym config files: vLLM model config and Workplace Assistant agent/resources config |
| `max_steps` | Maximum tool-calling steps per task (6 for Workplace Assistant) |

:::{important}
The `vllm_model_for_training.yaml` config is required for NeMo RL training integration.
:::

---

## Next Steps

With the Gym configuration understood, learn about the GRPO training parameters:

:::{button-ref} training-nemo-rl-grpo-nemo-rl-configuration
:color: primary
:ref-type: ref

Continue to NeMo RL Configuration →
:::
