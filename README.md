# NeMo Gym

NeMo Gym is a framework for building reinforcement learning environments to train large language models. 
> *Part of the [NVIDIA NeMo](https://www.nvidia.com/en-us/ai-data-science/products/nemo/) software suite for managing the AI agent lifecycle.*


> [!IMPORTANT]
> NeMo Gym is currently in early development. You should expect evolving APIs, incomplete documentation, and occasional bugs. We welcome contributions and feedback - for any changes, please open an issue first to kick off discussion!


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

### Table 1: Example Resource Servers

Purpose: Demonstrate NeMo Gym patterns and concepts.

<!-- START_EXAMPLE_ONLY_SERVERS_TABLE -->
| Name             | Demonstrates                         | Config                                                                                                       | README                                                                    |
| ---------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| Multi Step       | Instruction_Following example        | <a href='resources_servers/example_multi_step/configs/example_multi_step.yaml'>example_multi_step.yaml</a>   | <a href='resources_servers/example_multi_step/README.md'>README</a>       |
| Simple Weather   | Basic single-step tool calling       | <a href='resources_servers/example_simple_weather/configs/simple_weather.yaml'>simple_weather.yaml</a>       | <a href='resources_servers/example_simple_weather/README.md'>README</a>   |
| Stateful Counter | Session state management (in-memory) | <a href='resources_servers/example_stateful_counter/configs/stateful_counter.yaml'>stateful_counter.yaml</a> | <a href='resources_servers/example_stateful_counter/README.md'>README</a> |
<!-- END_EXAMPLE_ONLY_SERVERS_TABLE -->

### Table 2: Resource Servers for Training

Purpose: Training-ready environments with curated datasets.

> [!TIP]
> Each resource server includes example data, configuration files, and tests. See each server's README for details.

<!-- START_TRAINING_SERVERS_TABLE -->
| Dataset                                                                                                                                      | Domain                | Resource Server            | Description                                                                                          | Value                                                                    | Train | Validation | License                                        |
| -------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ----- | ---------- | ---------------------------------------------- |
| <a href='resources_servers/google_search/configs/google_search.yaml'>Nemotron-RL-knowledge-web_search-mcqa</a>                               | agent                 | Google Search              | Multi-choice question answering problems with search tools integrated                                | Improve knowledge-related benchmarks with search tools                   | ✓     | -          | Apache 2.0                                     |
| <a href='resources_servers/math_advanced_calculations/configs/math_advanced_calculations.yaml'>Nemotron-RL-math-advanced_calculations</a>    | agent                 | Math Advanced Calculations | An instruction following math environment with counter-intuitive calculators                         | Improve instruction following capabilities in specific math environments | ✓     | -          | Apache 2.0                                     |
| <a href='resources_servers/workplace_assistant/configs/workplace_assistant.yaml'>Nemotron-RL-agent-workplace_assistant</a>                   | agent                 | Workplace Assistant        | Workplace assistant multi-step tool-using environment                                                | Improve multi-step tool use capability                                   | ✓     | ✓          | Apache 2.0                                     |
| <a href='resources_servers/mini_swe_agent/configs/mini_swe_agent.yaml'>Nemotron-RL-coding-mini_swe_agent</a>                                 | coding                | Mini Swe Agent             | A software development with mini-swe-agent orchestration                                             | Improve software development capabilities, like SWE-bench                | ✓     | ✓          | MIT                                            |
| <a href='resources_servers/instruction_following/configs/instruction_following.yaml'>Nemotron-RL-instruction_following</a>                   | instruction_following | Instruction Following      | Instruction following datasets targeting IFEval and IFBench style instruction following capabilities | Improve IFEval and IFBench                                               | ✓     | -          | Apache 2.0                                     |
| <a href='resources_servers/structured_outputs/configs/structured_outputs_json.yaml'>Nemotron-RL-instruction_following-structured_outputs</a> | instruction_following | Structured Outputs         | Check if responses are following structured output requirements in prompts                           | Improve instruction following capabilities                               | ✓     | ✓          | Apache 2.0                                     |
| <a href='resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml'>Nemotron-RL-knowledge-openQA</a>                        | knowledge             | Equivalence Llm Judge      | Short answer questions with LLM-as-a-judge                                                           | Improve knowledge-related benchmarks like GPQA / HLE                     | -     | -          | -                                              |
| <a href='resources_servers/mcqa/configs/mcqa.yaml'>Nemotron-RL-knowledge-mcqa</a>                                                            | knowledge             | Mcqa                       | Multi-choice question answering problems                                                             | Improve benchmarks like MMLU / GPQA / HLE                                | ✓     | -          | Apache 2.0                                     |
| <a href='resources_servers/math_with_judge/configs/math_with_judge.yaml'>Nemotron-RL-math-OpenMathReasoning</a>                              | math                  | Math With Judge            | Math dataset with math-verify and LLM-as-a-judge                                                     | Improve math capabilities including AIME 24 / 25                         | ✓     | ✓          | Creative Commons Attribution 4.0 International |
<!-- END_TRAINING_SERVERS_TABLE -->


## Citations

If you use NeMo Gym in your research, please cite it using the following BibTeX entry:

@misc{nemo-gym,
title = {NeMo Gym: An Open Source Framework for Scaling Reinforcement Learning Environments for LLM},
howpublished = {\url{https://github.com/NVIDIA-NeMo/Gym}},
year = {2025},
note = {GitHub repository},
}