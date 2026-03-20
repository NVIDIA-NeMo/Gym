# NeMo Gym

**[Requirements](#-requirements)** • **[Quick Start](#-quick-start)** • **[Available Environments](#-available-environments)** • **[Documentation & Resources](#-documentation--resources)** • **[Community & Support](#-community--support)** • **[Citations](#-citations)**

NeMo Gym is a library for building reinforcement learning (RL) training environments for large language models (LLMs). It provides infrastructure to develop environments, scale rollout collection, and integrate seamlessly with your preferred training framework.

## 🏆 Why NeMo Gym?

- Scaffolding and patterns to accelerate environment development: multi-step, multi-turn, and user modeling scenarios
- Contribute environments without expert knowledge of the entire RL training loop
- Test environments and throughput end-to-end, independent of the RL training loop
- Interoperable with existing environments, systems, and RL training frameworks
- Growing collection of training environments and datasets for Reinforcement Learning from Verifiable Reward (RLVR)

> [!IMPORTANT]
> NeMo Gym is currently in early development. You should expect evolving APIs, incomplete documentation, and occasional bugs. We welcome contributions and feedback - for any changes, please open an issue first to kick off discussion!

## 🔗 Ecosystem

NeMo Gym is part of [NVIDIA NeMo](https://docs.nvidia.com/nemo/gym/latest/about/ecosystem.html#related-nemo-libraries), NVIDIA's GPU-accelerated platform for building and training generative AI models. NeMo Gym integrates with a growing number of RL training frameworks and environment libraries; see the [Ecosystem](https://docs.nvidia.com/nemo/gym/latest/about/ecosystem.html) page for full details and tutorials.

**Training Frameworks:** [NeMo RL](https://docs.nvidia.com/nemo/gym/latest/training-tutorials/nemo-rl-grpo/index.html) • [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/python/agent_func_nemogym_executor.py) • [Unsloth](https://docs.nvidia.com/nemo/gym/latest/training-tutorials/unsloth-training.html) • [more →](https://docs.nvidia.com/nemo/gym/latest/about/ecosystem.html#training-framework-integrations)

**Environment Libraries:** [Reasoning Gym](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/reasoning_gym) • [Aviary](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/aviary) • [more →](https://docs.nvidia.com/nemo/gym/latest/about/ecosystem.html#environment-library-integrations)

## 📋 Requirements

NeMo Gym is designed to run on standard development machines:

| Hardware Requirements | Software Requirements |
| --------------------- | --------------------- |
| **GPU**: Not required for NeMo Gym library operation<br>• GPU may be needed for specific resources servers or model inference (see individual server documentation) | **Operating System**:<br>• Linux (Ubuntu 20.04+, or equivalent)<br>• macOS (11.0+ for x86_64, 12.0+ for Apple Silicon)<br>• Windows (via WSL2) |
| **CPU**: Any modern x86_64 or ARM64 processor (e.g., Intel, AMD, Apple Silicon) | **Python**: 3.12 or higher |
| **RAM**: Minimum 8 GB (16 GB+ recommended for larger environments) | **Git**: For cloning the repository |
| **Storage**: Minimum 5 GB free disk space for installation and basic usage | **Internet Connection**: Required for downloading dependencies and API access |

**Additional Requirements**

- **API Keys**: OpenAI API key with available credits (for the quickstart examples)
  - Other model providers supported (Azure OpenAI, self-hosted models via vLLM)
- **Ray**: Automatically installed as a dependency (no separate setup required)

## 🚀 Quick Start

Install NeMo Gym, start the servers, and collect your first verified rollouts for RL training.

### Setup
```bash
# Clone the repository
git clone git@github.com:NVIDIA-NeMo/Gym.git
cd Gym

# Install UV (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install NeMo Gym
uv sync --extra dev --group docs
```

### Configure Your API Key
Create an `env.yaml` file that contains your OpenAI API key and the [policy model](https://docs.nvidia.com/nemo/gym/latest/about/concepts/key-terminology.html#term-Policy-Model) you want to use. Replace `your-openai-api-key` with your actual key. This file helps keep your secrets out of version control while still making them available to NeMo Gym.

```bash
echo "policy_base_url: https://api.openai.com/v1
policy_api_key: your-openai-api-key
policy_model_name: gpt-4.1-2025-04-14" > env.yaml
```

> [!NOTE]
> We use GPT-4.1 in this quickstart because it provides low latency (no reasoning step) and works reliably out-of-the-box. NeMo Gym is **not limited to OpenAI models**—you can use self-hosted models via vLLM or any OpenAI-compatible inference server. See the [documentation](https://docs.nvidia.com/nemo/gym/latest/get-started/detailed-setup.html) for details.

### Start Servers

**Terminal 1 (start servers)**:
```bash
# Start servers (this will keep running)
config_paths="resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"
```

**Terminal 2 (interact with agent)**:
```bash
# In a NEW terminal, activate environment
source .venv/bin/activate

# Interact with your agent
python responses_api_agents/simple_agent/client.py
```

### Collect Rollouts

**Terminal 2** (keep servers running in Terminal 1):
```bash
# Create a simple dataset with one query
echo '{"responses_create_params":{"input":[{"role":"developer","content":"You are a helpful assistant."},{"role":"user","content":"What is the weather in Seattle?"}]}}' > weather_query.jsonl

# Collect verified rollouts
ng_collect_rollouts \
    +agent_name=example_single_tool_call_simple_agent \
    +input_jsonl_fpath=weather_query.jsonl \
    +output_jsonl_fpath=weather_rollouts.jsonl

# View the result
cat weather_rollouts.jsonl | python -m json.tool
```
This generates training data with verification scores!

### Clean Up Servers

**Terminal 1** with the running servers: Ctrl+C to stop the ng_run process.

### Next Steps

Now that you can generate rollouts, choose your path:

- **Start training** — Train models using NeMo Gym with your preferred RL framework. See the [Training Tutorials](https://docs.nvidia.com/nemo/gym/latest/training-tutorials/index.html).

- **Use an existing environment** — Browse the [Available Environments](#-available-environments) below to find an environment that matches your goals.

- **Build a custom environment** — Implement or integrate existing tools and define task verification logic. Get started with the [Creating a Training Environment](https://docs.nvidia.com/nemo/gym/latest/environment-tutorials/creating-training-environment.html) tutorial.


## 📦 Available Environments

NeMo Gym includes a curated collection of environments for training and evaluation across multiple domains:

### Example Environment Patterns

Purpose: Demonstrate NeMo Gym patterns and concepts.

<!-- START_EXAMPLE_ONLY_SERVERS_TABLE -->
| Name | Demonstrates | Config | README |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------ | ------------------------------------ |
<!-- END_EXAMPLE_ONLY_SERVERS_TABLE -->

### Environments for Training & Evaluation

Purpose: Training-ready environments with curated datasets.

> [!TIP]
> Each resources server includes example data, configuration files, and tests. See each server's README for details.

<!-- START_TRAINING_SERVERS_TABLE -->
| Resources Server | Domain | Description | Value | Train | Validation | License | Config | Dataset |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------ | ------------------------------------------------- |
<!-- END_TRAINING_SERVERS_TABLE -->

## 📖 Documentation & Resources

- **[Documentation](https://docs.nvidia.com/nemo/gym/latest/index.html)** - Technical reference docs
- **[Training Tutorials](https://docs.nvidia.com/nemo/gym/latest/training-tutorials/index.html)** - Train with NeMo Gym environments
- **[API Reference](https://docs.nvidia.com/nemo/gym/latest/apidocs/index.html)** - Complete class and function reference
 

## 🤝 Community & Support

We'd love your contributions! Here's how to get involved:

- **[Report Issues](https://github.com/NVIDIA-NeMo/Gym/issues)** - Bug reports and feature requests
- **[Contributing Guide](https://docs.nvidia.com/nemo/gym/latest/contribute/index.html)** - How to contribute code, docs, new environments, or training framework integrations

## 📚 Citations

If you use NeMo Gym in your research, please cite it using the following BibTeX entry:

```bibtex
@misc{nemo-gym,
  title = {NeMo Gym: An Open Source Library for Scaling Reinforcement Learning Environments for LLM},
  howpublished = {\url{https://github.com/NVIDIA-NeMo/Gym}},
  author={NVIDIA},
  year = {2025},
  note = {GitHub repository},
}
```
