(training-nemo-rl-grpo-nemo-rl-configuration)=

# NeMo RL Configuration
:::{tip}
The full NeMo RL training configuration file is located at: `examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml`
:::


## Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model_name` | nvidia/NVIDIA-Nemotron-Nano-9B-v2 | Base model |
| `max_total_sequence_length` | 32768 | Maximum context length |
| `precision` | bfloat16 | Training precision |
| `tensor_model_parallel_size` | 8 | Tensor parallelism across GPUs |

## GRPO Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_prompts_per_step` | 4 | Number of prompts per training step |
| `num_generations_per_prompt` | 4 | Rollouts generated per prompt |
| `max_num_steps` | 10 | Total training steps |
| `use_leave_one_out_baseline` | true | Variance reduction technique |
| `normalize_rewards` | true | Normalize rewards across batch |

## Optimizer Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| `optimizer` | Adam | Optimizer type |
| `lr` | 5.0e-6 | Learning rate |
| `min_lr` | 5.0e-7 | Minimum learning rate |
| `weight_decay` | 0.01 | Weight decay |
| `adam_beta1` / `adam_beta2` | 0.9 / 0.999 | Adam hyperparameters |
| `clip_grad` | 1.0 | Gradient clipping threshold |
