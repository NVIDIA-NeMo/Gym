(gym-rl-framework-integration-success-criteria)=

# Gym + RL framework integration success criteria

:::{tip}
The success criteria for Gym + RL framework integration may change over time as we discover various problems with new integrations!
:::

1. You have the same form factor as detailed in the previous page.
2. You are able to run any arbitrary training environment from Gym via training run configuration.
3. You have trained on the [DAPO17k math training environment](https://github.com/NVIDIA-NeMo/Gym/blob/299e8c04f4a3bbf0f6069139092225f2fe3aa70f/resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml) for > 1k steps using [Qwen 3 4B Instruct 2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) and reached 85%+ on the AIME24 validation set provided with the training environment.
4. You have trained on the [Workplace assistant training environment](https://github.com/NVIDIA-NeMo/Gym/tree/299e8c04f4a3bbf0f6069139092225f2fe3aa70f/resources_servers/workplace_assistant) for > 100 steps using [Qwen 3 4B Instruct 2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) and observed validation set improvements.
