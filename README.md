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
config_paths="resources_servers/simple_weather/configs/simple_weather.yaml,responses_api_models/openai_model/configs/openai_model.yaml"
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

<!-- START_RESOURCE_TABLE -->
| Domain                | Resource Server Name  | Config Path                                                                                                                                                         | License                                                   | Usage                      |
| --------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | -------------------------- |
| agent                 | Google Search         | <a href='resources_servers/google_search/configs/google_search.yaml'>resources_servers/google_search/configs/google_search.yaml</a>                                 | Apache 2.0                                                | Train, Example             |
| agent                 | Multiverse Math Hard  | <a href='resources_servers/multiverse_math_hard/configs/multiverse_math_hard.yaml'>resources_servers/multiverse_math_hard/configs/multiverse_math_hard.yaml</a>     | Apache 2.0                                                | Train, Example             |
| agent                 | Simple Weather        | <a href='resources_servers/simple_weather/configs/simple_weather.yaml'>resources_servers/simple_weather/configs/simple_weather.yaml</a>                             | None                                                      | Example                    |
| agent                 | Stateful Counter      | <a href='resources_servers/stateful_counter/configs/stateful_counter.yaml'>resources_servers/stateful_counter/configs/stateful_counter.yaml</a>                     | Apache 2.0                                                | Train, Validation, Example |
| agent                 | Workbench             | <a href='resources_servers/workbench/configs/workbench.yaml'>resources_servers/workbench/configs/workbench.yaml</a>                                                 | Apache 2.0                                                | Train, Validation, Example |
| coding                | Comp Coding           | <a href='resources_servers/comp_coding/configs/comp_coding.yaml'>resources_servers/comp_coding/configs/comp_coding.yaml</a>                                         | Apache 2.0                                                | Train, Validation, Example |
| instruction_following | Instruction Following | <a href='resources_servers/instruction_following/configs/instruction_following.yaml'>resources_servers/instruction_following/configs/instruction_following.yaml</a> | Apache 2.0                                                | Train, Example             |
| instruction_following | Multineedle           | <a href='resources_servers/multineedle/configs/multineedle.yaml'>resources_servers/multineedle/configs/multineedle.yaml</a>                                         | Apache 2.0                                                | Train, Validation, Example |
| knowledge             | Equivalence Llm Judge | <a href='resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml'>resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml</a> | None                                                      | Example, Example           |
| knowledge             | Mcqa                  | <a href='resources_servers/mcqa/configs/mcqa.yaml'>resources_servers/mcqa/configs/mcqa.yaml</a>                                                                     | Apache 2.0                                                | Train, Example, Example    |
| math                  | Library Judge Math    | <a href='resources_servers/library_judge_math/configs/bytedtsinghua_dapo17k.yaml'>resources_servers/library_judge_math/configs/bytedtsinghua_dapo17k.yaml</a>       | Apache 2.0                                                | Train, Validation          |
| math                  | Library Judge Math    | <a href='resources_servers/library_judge_math/configs/dapo17k.yaml'>resources_servers/library_judge_math/configs/dapo17k.yaml</a>                                   | Apache 2.0                                                | Train, Validation          |
| math                  | Library Judge Math    | <a href='resources_servers/library_judge_math/configs/library_judge_math.yaml'>resources_servers/library_judge_math/configs/library_judge_math.yaml</a>             | Creative Commons Attribution 4.0 International            | Train, Validation, Example |
| math                  | Library Judge Math    | <a href='resources_servers/library_judge_math/configs/math_stack_overflow.yaml'>resources_servers/library_judge_math/configs/math_stack_overflow.yaml</a>           | Creative Commons Attribution-ShareAlike 4.0 International | Train, Validation          |
| math                  | Python Math Exec      | <a href='resources_servers/python_math_exec/configs/python_math_exec.yaml'>resources_servers/python_math_exec/configs/python_math_exec.yaml</a>                     | Apache 2.0                                                | Train, Example             |
<!-- END_RESOURCE_TABLE -->

> [!TIP]
> Each resource server includes example data, configuration files, and tests. See each server's README for details.