(training-nemo-rl-grpo-gym-configuration)=

# Gym Configuration
:::{tip}
The full NeMo RL training configuration file is located at: `examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml`
:::

There are two Gym-specific sections within the NeMo RL training config under the `data` and `env` parameters, shown below.

```yaml
data:
  train_jsonl_fpath: 3rdparty/Gym-workspace/Gym/data/workplace_assistant/train.jsonl
  validation_jsonl_fpath: 3rdparty/Gym-workspace/Gym/data/workplace_assistant/validation.jsonl

env:
  should_use_nemo_gym: true
  nemo_gym:
    config_paths:
    - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
    - resources_servers/workplace_assistant/configs/workplace_assistant.yaml
    workplace_assistant_simple_agent:
      responses_api_agents:
        simple_agent:
          max_steps: 6  # Maximum tool-calling steps per task
```

Under the `data` section, we provide NeMo RL with the filepaths to our train and validation datasets that will be prepared in the {doc}`Setup Instructions <setup>`.

The `env` section contains the majority of the configuration for Gym during NeMo RL training:
- `should_use_nemo_gym: true`: Enable Gym
- `nemo_gym`: Everything under this key is exactly a Gym config. Specifically, we use two configs:
  - `responses_api_models/vllm_model/configs/vllm_model_for_training.yaml`: This model is necessary for NeMo RL training!
  - `resources_servers/workplace_assistant/configs/workplace_assistant.yaml`: This config contains the Workplace Assistant agent and resources servers.
  - We explicitly set the number of `max_steps` in the top level config, allowing the model 6 steps to accomplish a given task.
