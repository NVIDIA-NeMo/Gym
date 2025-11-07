# NeMo Gym

NeMo Gym enables scalable data collection for reinforcement learning of AI agents. It provides both the infrastructure to systematically capture agent interactions and a collection of high-quality RL environments, making it easy to generate training data for reinforcement learning workflows using the framework of your choice.

> [!IMPORTANT]
> NeMo Gym is currently in early development. While NVIDIA is using it for training Nemotron models, you should expect evolving APIs, incomplete documentation, and occasional bugs. We welcome contributions and feedback! For any changes, please open an issue first to coordinate with the team and ensure alignment with product direction.


## 🏆 Why NeMo Gym?

- **Fast Data Generation** - Less boilerplate, more innovation: get from prompt to training rollouts quickly
- **Flexible Environment Integration** - Connect to any environment: custom APIs, MCP-compatible tools, external services, or leverage our curated resources
- **Smart Orchestration** - Async architecture automatically coordinates model-resource calls for high-throughput training workloads
- **Configuration-Driven** - Swap models, resources, and environments via YAML without touching code
- **Standardized Interfaces** - Consistent patterns for models, environments, and agents across different systems


## 🚀 Quick Start

### New to NeMo Gym?
Follow our **[Tutorial Series](docs/tutorials/README.md)** for a progressive learning experience:
- **Setup & Core Concepts**: Understand Models, Resources, Agents and run your first interaction
- **Rollout Generation**: Capture agent interactions for RL, SFT, and DPO training  
- **Custom Environments**: Build your own tools, verification systems, and training scenarios
- **Production Deployment**: Configuration, testing, scaling, and advanced agent patterns

### Quick Installation
```bash
git clone git@github.com:NVIDIA-NeMo/Gym.git
cd Gym

# Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra dev --group docs
```

### Run Your First Agent
Start with **[Understanding Concepts](docs/tutorials/01-concepts.md)**, then follow **[Setup & Installation](docs/tutorials/02-setup.md)** for hands-on implementation.

**TLDR**:
```bash
# Configure API access
echo "policy_base_url: https://api.openai.com/v1
policy_api_key: your-openai-api-key
policy_model_name: gpt-4.1-2025-04-14" > env.yaml

# Start servers and run agent
config_paths="resources_servers/example_simple_weather/configs/simple_weather.yaml,responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"

python responses_api_agents/simple_agent/client.py
```

## 📖 Documentation

- **[Tutorials](docs/tutorials/README.md)** - Progressive learning path
- **[Contributing](https://github.com/NVIDIA-NeMo/Gym/blob/main/CONTRIBUTING.md)** - Developer setup, testing, and contribution guidelines
- **[API Documentation](https://github.com/NVIDIA-NeMo/Gym/tree/main/docs)** - Technical reference and API specifications
 

## 🤝 Community & Support

We'd love your contributions! Here's how to get involved:

- **[Report Issues](https://github.com/NVIDIA-NeMo/Gym/issues)** - Bug reports and feature requests
<!-- TODO: Add link [Discussions](https://github.com/NVIDIA-NeMo/Gym/discussions) -->
- **Discussions (Coming soon!)** - Community Q&A and ideas
- **[Contributing Guide](https://github.com/NVIDIA-NeMo/Gym/blob/main/CONTRIBUTING.md)** - How to contribute code, docs, or new environments

## 📦 Available Resource Servers

NeMo Gym includes a curated collection of resource servers for training and evaluation across multiple domains:

> [!TIP]
> Each resource server includes example data, configuration files, and tests. See each server's README for details.

### Table 1: Example Resource Servers

Purpose: Demonstrate NeMo Gym patterns and concepts.

<!-- START_EXAMPLE_ONLY_SERVERS_TABLE -->
| Name             | Demonstrates                         | Config                                                                                                       | README                                                                    |
| ---------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| Simple Weather   | Basic single-step tool calling       | <a href='resources_servers/example_simple_weather/configs/simple_weather.yaml'>simple_weather.yaml</a>       | <a href='resources_servers/example_simple_weather/README.md'>README</a>   |
| Stateful Counter | Session state management (in-memory) | <a href='resources_servers/example_stateful_counter/configs/stateful_counter.yaml'>stateful_counter.yaml</a> | <a href='resources_servers/example_stateful_counter/README.md'>README</a> |
<!-- END_EXAMPLE_ONLY_SERVERS_TABLE -->

### Table 2: Resource Servers for Training

Purpose: Training-ready environments with curated datasets.

<!-- START_TRAINING_SERVERS_TABLE -->
| Domain                | Resource Server       | Train | Validation | Verified                                                                                    | Config                                                                                                                                         | License                                                   |
| --------------------- | --------------------- | ----- | ---------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| coding                | Comp Coding           | ✓     | ✓          | <a href='https://wandb.ai/nvidia/bxyu-nemo-gym-rl-integration-20250926/runs/54uzarwq'>✓</a> | <a href='resources_servers/comp_coding/configs/comp_coding.yaml'>comp_coding.yaml</a>                                                          | Apache 2.0                                                |
| math                  | Library Judge Math    | ✓     | ✓          | <a href='https://wandb.ai/nvidia/bxyu-nemo-gym-rl-integration/runs/5d2a98ix'>✓</a>          | <a href='resources_servers/library_judge_math/configs/bytedtsinghua_dapo17k.yaml'>bytedtsinghua_dapo17k.yaml</a>                               | Apache 2.0                                                |
| agent                 | Google Search         | ✓     | -          | -                                                                                           | <a href='resources_servers/google_search/configs/google_search.yaml'>google_search.yaml</a>                                                    | Apache 2.0                                                |
| agent                 | Multiverse Math Hard  | ✓     | -          | -                                                                                           | <a href='resources_servers/multiverse_math_hard/configs/multiverse_math_hard.yaml'>multiverse_math_hard.yaml</a>                               | Apache 2.0                                                |
| agent                 | Workbench             | ✓     | ✓          | -                                                                                           | <a href='resources_servers/workbench/configs/workbench.yaml'>workbench.yaml</a>                                                                | Apache 2.0                                                |
| coding                | Mini Swe Resource     | ✓     | ✓          | -                                                                                           | <a href='resources_servers/mini_swe_resource/configs/mini_swe_resource.yaml'>mini_swe_resource.yaml</a>                                        | MIT                                                       |
| instruction_following | Instruction Following | ✓     | -          | -                                                                                           | <a href='resources_servers/instruction_following/configs/instruction_following.yaml'>instruction_following.yaml</a>                            | Apache 2.0                                                |
| instruction_following | Multineedle           | ✓     | ✓          | -                                                                                           | <a href='resources_servers/multineedle/configs/multineedle.yaml'>multineedle.yaml</a>                                                          | Apache 2.0                                                |
| instruction_following | Structured Outputs    | ✓     | ✓          | -                                                                                           | <a href='resources_servers/structured_outputs/configs/structured_outputs_json.yaml'>structured_outputs_json.yaml</a>                           | Apache 2.0                                                |
| knowledge             | Equivalence Llm Judge | -     | -          | -                                                                                           | <a href='resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml'>equivalence_llm_judge.yaml</a>                            | -                                                         |
| knowledge             | Mcqa                  | ✓     | -          | -                                                                                           | <a href='resources_servers/mcqa/configs/mcqa.yaml'>mcqa.yaml</a>                                                                               | Apache 2.0                                                |
| math                  | Library Judge Math    | -     | ✓          | -                                                                                           | <a href='resources_servers/library_judge_math/configs/dapo17k_trajectory_collection.yaml'>dapo17k_trajectory_collection.yaml</a>               | -                                                         |
| math                  | Library Judge Math    | ✓     | ✓          | -                                                                                           | <a href='resources_servers/library_judge_math/configs/dapo17k.yaml'>dapo17k.yaml</a>                                                           | Apache 2.0                                                |
| math                  | Library Judge Math    | ✓     | ✓          | -                                                                                           | <a href='resources_servers/library_judge_math/configs/dapo17k_filtered_qwen330ba3binstruct.yaml'>dapo17k_filtered_qwen330ba3binstruct.yaml</a> | Apache 2.0                                                |
| math                  | Library Judge Math    | ✓     | ✓          | -                                                                                           | <a href='resources_servers/library_judge_math/configs/library_judge_math.yaml'>library_judge_math.yaml</a>                                     | Creative Commons Attribution 4.0 International            |
| math                  | Library Judge Math    | ✓     | ✓          | -                                                                                           | <a href='resources_servers/library_judge_math/configs/math_stack_overflow.yaml'>math_stack_overflow.yaml</a>                                   | Creative Commons Attribution-ShareAlike 4.0 International |
| math                  | Python Math Exec      | ✓     | -          | -                                                                                           | <a href='resources_servers/python_math_exec/configs/python_math_exec.yaml'>python_math_exec.yaml</a>                                           | Apache 2.0                                                |
<!-- END_TRAINING_SERVERS_TABLE -->
