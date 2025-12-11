(training-nemo-rl-grpo-gym-configuration)=

# Gym Configuration

BRIAN TODO

The training configuration file is located at:
`examples/penguin/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml`

#### Environment Configuration

```yaml
env:
  should_use_penguin: true
  penguin:
    config_paths:
    - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
    - resources_servers/workplace_assistant/configs/workplace_assistant.yaml
    workplace_assistant_simple_agent:
      responses_api_agents:
        simple_agent:
          max_steps: 6  # Maximum tool-calling steps per task
```
