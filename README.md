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

## 🚀 Quick Start

### Setup
```bash
git clone git@github.com:NVIDIA-NeMo/Gym.git
cd Gym

# Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra dev --group docs

# Configure your model API access
echo "policy_base_url: https://api.openai.com/v1
policy_api_key: your-openai-api-key
policy_model_name: gpt-4.1-2025-04-14" > env.yaml
```

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
- **[Get Started](https://docs.nvidia.com/nemo/gym/latest/get-started/index.html)** - Installation through your first verified agent
- **[Tutorials](https://docs.nvidia.com/nemo/gym/latest/tutorials/index.html)** - Hands-on tutorials and practical examples
- **[Multi-Step Patterns](https://docs.nvidia.com/nemo/gym/latest/tutorials/development/multi-step-patterns.html)** - Essential guide to session-based state management
 

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
