# NeMo Gym

NeMo Gym is a framework for building reinforcement learning environments to train large language models. 

NeMo Gym is a component of the [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/), NVIDIA’s GPU-accelerated platform for building and training generative AI models.


## 🏆 Why NeMo Gym?

- Scaffolding and patterns to accelerate environment development: multi-step, multi-turn, and user modeling scenarios
- Contribute environments without expert knowledge of the entire RL training loop
- Test environment and throughput end-to-end independent of the RL training loop
- Interoperable with existing environments, systems and RL training frameworks
- Growing collection of training environments and datasets to enable Reinforcement Learning from Verifiable Reward (RLVR)

> [!IMPORTANT]
> NeMo Gym is currently in early development. You should expect evolving APIs, incomplete documentation, and occasional bugs. We welcome contributions and feedback - for any changes, please open an issue first to kick off discussion!

## 📋 Requirements

### Hardware Requirements

NeMo Gym is designed to run on standard development machines:

- **GPU**: Not required for NeMo Gym framework operation
  - GPU may be needed for specific resource servers or model inference (see individual server documentation)
- **CPU**: Any modern x86_64 or ARM64 processor (e.g., Intel, AMD, Apple Silicon)
- **RAM**: Minimum 8 GB (16 GB+ recommended for larger environments)
- **Storage**: Minimum 5 GB free disk space for installation and basic usage

### Software Requirements

- **Operating System**: 
  - Linux (Ubuntu 20.04+, or equivalent)
  - macOS (11.0+ for x86_64, 12.0+ for Apple Silicon)
  - Windows (via WSL2)
- **Python**: 3.12 or higher
- **Git**: For cloning the repository
- **Internet Connection**: Required for downloading dependencies and API access

### Additional Requirements

- **API Keys**: OpenAI API key with available credits (for the quickstart examples)
  - Other model providers supported (Azure OpenAI, self-hosted models via vLLM)
- **Ray**: Automatically installed as a dependency (no separate setup required)

## 🚀 Quick Start

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
> We use GPT-4.1 in this quickstart because it provides low latency (no reasoning step) and works reliably out-of-the-box. NeMo Gym is **not limited to OpenAI models**—you can use self-hosted models via vLLM or any OpenAI-compatible inference server. See the [documentation](https://docs.nvidia.com/nemo/gym/latest/get-started/setup-installation.html) for details.

### Start Servers

**Terminal 1 (start servers)**:
```bash
# Start servers (this will keep running)
config_paths="resources_servers/example_simple_weather/configs/simple_weather.yaml,\
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
    +agent_name=simple_weather_simple_agent \
    +input_jsonl_fpath=weather_query.jsonl \
    +output_jsonl_fpath=weather_rollouts.jsonl

# View the result
cat weather_rollouts.jsonl | python -m json.tool
```
This generates training data with verification scores!

### Clean Up Servers

**Terminal 1** with the running servers: Ctrl+C to stop the ng_run process.


## 📖 Documentation

- **[Documentation](https://docs.nvidia.com/nemo/gym/latest/index.html)** - Technical reference docs
- **[Tutorials](https://docs.nvidia.com/nemo/gym/latest/tutorials/index.html)** - Hands-on tutorials and practical examples
 

## 🤝 Community & Support

We'd love your contributions! Here's how to get involved:

- **[Report Issues](https://github.com/NVIDIA-NeMo/Gym/issues)** - Bug reports and feature requests
- **[Contributing Guide](https://github.com/NVIDIA-NeMo/Gym/blob/main/CONTRIBUTING.md)** - How to contribute code, docs, or new environments

## 📚 Citations

If you use NeMo Gym in your research, please cite it using the following BibTeX entry:

```bibtex
@misc{nemo-gym,
  title = {NeMo Gym: An Open Source Framework for Scaling Reinforcement Learning Environments for LLM},
  howpublished = {\url{https://github.com/NVIDIA-NeMo/Gym}},
  year = {2025},
  note = {GitHub repository},
}
```

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
| Resource Server            | Domain                | Dataset                                                                                                                                                        | Description                                                                                          | Value                                                                    | Config                                                                                                    | Train | Validation | License                                        |
| -------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- | ----- | ---------- | ---------------------------------------------- |
| Google Search              | agent                 | <a href='https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-web_search-mcqa'>Nemotron-RL-knowledge-web_search-mcqa</a>                               | Multi-choice question answering problems with search tools integrated                                | Improve knowledge-related benchmarks with search tools                   | <a href='resources_servers/google_search/configs/google_search.yaml'>config</a>                           | ✓     | -          | Apache 2.0                                     |
| Math Advanced Calculations | agent                 | <a href='https://huggingface.co/datasets/nvidia/Nemotron-RL-math-advanced_calculations'>Nemotron-RL-math-advanced_calculations</a>                             | An instruction following math environment with counter-intuitive calculators                         | Improve instruction following capabilities in specific math environments | <a href='resources_servers/math_advanced_calculations/configs/math_advanced_calculations.yaml'>config</a> | ✓     | -          | Apache 2.0                                     |
| Workplace Assistant        | agent                 | <a href='https://huggingface.co/datasets/nvidia/Nemotron-RL-agent-workplace_assistant'>Nemotron-RL-agent-workplace_assistant</a>                               | Workplace assistant multi-step tool-using environment                                                | Improve multi-step tool use capability                                   | <a href='resources_servers/workplace_assistant/configs/workplace_assistant.yaml'>config</a>               | ✓     | ✓          | Apache 2.0                                     |
| Mini Swe Agent             | coding                | <a href='https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified'>SWE-bench_Verified</a>                                                              | A software development with mini-swe-agent orchestration                                             | Improve software development capabilities, like SWE-bench                | <a href='resources_servers/mini_swe_agent/configs/mini_swe_agent.yaml'>config</a>                         | ✓     | ✓          | MIT                                            |
| Instruction Following      | instruction_following | <a href='https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following'>Nemotron-RL-instruction_following</a>                                       | Instruction following datasets targeting IFEval and IFBench style instruction following capabilities | Improve IFEval and IFBench                                               | <a href='resources_servers/instruction_following/configs/instruction_following.yaml'>config</a>           | ✓     | -          | Apache 2.0                                     |
| Structured Outputs         | instruction_following | <a href='https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following-structured_outputs'>Nemotron-RL-instruction_following-structured_outputs</a> | Check if responses are following structured output requirements in prompts                           | Improve instruction following capabilities                               | <a href='resources_servers/structured_outputs/configs/structured_outputs_json.yaml'>config</a>            | ✓     | ✓          | Apache 2.0                                     |
| Equivalence Llm Judge      | knowledge             | <a href='https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-openQA'>Nemotron-RL-knowledge-openQA</a>                                                 | Short answer questions with LLM-as-a-judge                                                           | Improve knowledge-related benchmarks like GPQA / HLE                     | <a href='resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml'>config</a>           | ✓     | -          | Apache 2.0                                     |
| Mcqa                       | knowledge             | <a href='https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-mcqa'>Nemotron-RL-knowledge-mcqa</a>                                                     | Multi-choice question answering problems                                                             | Improve benchmarks like MMLU / GPQA / HLE                                | <a href='resources_servers/mcqa/configs/mcqa.yaml'>config</a>                                             | ✓     | -          | Apache 2.0                                     |
| Math With Judge            | math                  | <a href='https://huggingface.co/datasets/nvidia/Nemotron-RL-math-OpenMathReasoning'>Nemotron-RL-math-OpenMathReasoning</a>                                     | Math dataset with math-verify and LLM-as-a-judge                                                     | Improve math capabilities including AIME 24 / 25                         | <a href='resources_servers/math_with_judge/configs/math_with_judge.yaml'>config</a>                       | ✓     | ✓          | Creative Commons Attribution 4.0 International |
<!-- END_TRAINING_SERVERS_TABLE -->
