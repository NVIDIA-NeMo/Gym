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

**Training Frameworks:** [NeMo RL](https://docs.nvidia.com/nemo/gym/latest/training-tutorials/nemo-rl-grpo/index.html) • [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/python/agent_func_nemogym_executor.py) • [TRL](https://docs.nvidia.com/nemo/gym/latest/training-tutorials/trl.html) • [Unsloth](https://docs.nvidia.com/nemo/gym/latest/training-tutorials/unsloth-training.html) • [more →](https://docs.nvidia.com/nemo/gym/latest/about/ecosystem.html#training-framework-integrations)

**Environment Libraries:** [Reasoning Gym](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/reasoning_gym) • [Aviary](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/aviary) • [more →](https://docs.nvidia.com/nemo/gym/latest/about/ecosystem.html#environment-library-integrations)

## 📋 Requirements

NeMo Gym is designed to run on standard development machines:

| Hardware Requirements | Software Requirements |
| --------------------- | --------------------- |
| **GPU**: Not required for NeMo Gym library operation<br>• GPU may be needed for specific resource servers or model inference (see individual server documentation) | **Operating System**:<br>• Linux (Ubuntu 20.04+, or equivalent)<br>• macOS (11.0+ for x86_64, 12.0+ for Apple Silicon)<br>• Windows (via WSL2) |
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
| Name               | Demonstrates                         | Config                                                                                                                             | README                                                                      |
| ------------------ | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Multi Step         | Multi-step tool calling              | <a href='resources_servers/example_multi_step/configs/example_multi_step.yaml'>example_multi_step.yaml</a>                         | <a href='resources_servers/example_multi_step/README.md'>README</a>         |
| Session State Mgmt | Session state management (in-memory) | <a href='resources_servers/example_session_state_mgmt/configs/example_session_state_mgmt.yaml'>example_session_state_mgmt.yaml</a> | <a href='resources_servers/example_session_state_mgmt/README.md'>README</a> |
| Single Tool Call   | Basic single-step tool calling       | <a href='resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml'>example_single_tool_call.yaml</a>       | <a href='resources_servers/example_single_tool_call/README.md'>README</a>   |
<!-- END_EXAMPLE_ONLY_SERVERS_TABLE -->

### Environments for Training & Evaluation

Purpose: Training-ready environments with curated datasets.

> [!TIP]
> Each resource server includes example data, configuration files, and tests. See each server's README for details.

<!-- START_TRAINING_SERVERS_TABLE -->
| Resource Server            | Config                                                                                                                                      | Domain                | Dataset                                                                                                                                                        | Description                                                                                                                        | Value                                                                                                                         | Train | Validation | License                                                   |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ----- | ---------- | --------------------------------------------------------- |
| Arc Agi                    | <a href='resources_servers/arc_agi/configs/arc_agi.yaml'>arc_agi.yaml</a>                                                                   | knowledge             | -                                                                                                                                                              | -                                                                                                                                  | -                                                                                                                             | -     | ✓          | -                                                         |
| Aviary                     | <a href='resources_servers/aviary/configs/aviary.yaml'>aviary.yaml</a>                                                                      | math                  | -                                                                                                                                                              | -                                                                                                                                  | -                                                                                                                             | ✓     | ✓          | Apache 2.0                                                |
| Aviary                     | <a href='resources_servers/aviary/configs/bixbench_aviary.yaml'>bixbench_aviary.yaml</a>                                                    | coding                | -                                                                                                                                                              | -                                                                                                                                  | -                                                                                                                             | -     | -          | -                                                         |
| Aviary                     | <a href='resources_servers/aviary/configs/gsm8k_aviary.yaml'>gsm8k_aviary.yaml</a>                                                          | math                  | -                                                                                                                                                              | -                                                                                                                                  | -                                                                                                                             | ✓     | ✓          | Apache 2.0                                                |
| Aviary                     | <a href='resources_servers/aviary/configs/hotpotqa_aviary.yaml'>hotpotqa_aviary.yaml</a>                                                    | agent                 | -                                                                                                                                                              | -                                                                                                                                  | -                                                                                                                             | ✓     | ✓          | Apache 2.0                                                |
| Calendar                   | <a href='resources_servers/calendar/configs/calendar.yaml'>calendar.yaml</a>                                                                | agent                 | <a href='https://huggingface.co/datasets/nvidia/Nemotron-RL-agent-calendar_scheduling'>Nemotron-RL-agent-calendar_scheduling</a>                               | -                                                                                                                                  | -                                                                                                                             | ✓     | ✓          | Apache 2.0                                                |
| Code Gen                   | <a href='resources_servers/code_gen/configs/code_gen.yaml'>code_gen.yaml</a>                                                                | coding                | <a href='https://huggingface.co/datasets/nvidia/nemotron-RL-coding-competitive_coding'>nemotron-RL-coding-competitive_coding</a>                               | -                                                                                                                                  | -                                                                                                                             | ✓     | ✓          | Apache 2.0                                                |
| Equivalence Llm Judge      | <a href='resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml'>equivalence_llm_judge.yaml</a>                         | knowledge             | -                                                                                                                                                              | Short answer questions with LLM-as-a-judge                                                                                         | Improve knowledge-related benchmarks like GPQA / HLE                                                                          | -     | -          | -                                                         |
| Equivalence Llm Judge      | <a href='resources_servers/equivalence_llm_judge/configs/lc.yaml'>lc.yaml</a>                                                               | knowledge             | -                                                                                                                                                              | -                                                                                                                                  | -                                                                                                                             | -     | -          | -                                                         |
| Equivalence Llm Judge      | <a href='resources_servers/equivalence_llm_judge/configs/lc_judge.yaml'>lc_judge.yaml</a>                                                   | knowledge             | -                                                                                                                                                              | -                                                                                                                                  | -                                                                                                                             | -     | -          | -                                                         |
| Equivalence Llm Judge      | <a href='resources_servers/equivalence_llm_judge/configs/nl2bash-equivalency.yaml'>nl2bash-equivalency.yaml</a>                             | agent                 | -                                                                                                                                                              | Short bash command generation questions with LLM-as-a-judge                                                                        | Improve foundational bash and IF capabilities                                                                                 | ✓     | ✓          | GNU General Public License v3.0                           |
| Google Search              | <a href='resources_servers/google_search/configs/google_search.yaml'>google_search.yaml</a>                                                 | agent                 | <a href='https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-web_search-mcqa'>Nemotron-RL-knowledge-web_search-mcqa</a>                               | Multi-choice question answering problems with search tools integrated                                                              | Improve knowledge-related benchmarks with search tools                                                                        | ✓     | -          | Apache 2.0                                                |
| Instruction Following      | <a href='resources_servers/instruction_following/configs/instruction_following.yaml'>instruction_following.yaml</a>                         | instruction_following | <a href='https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following'>Nemotron-RL-instruction_following</a>                                       | Instruction following datasets targeting IFEval and IFBench style instruction following capabilities                               | Improve IFEval and IFBench                                                                                                    | ✓     | -          | Apache 2.0                                                |
| Math Advanced Calculations | <a href='resources_servers/math_advanced_calculations/configs/math_advanced_calculations.yaml'>math_advanced_calculations.yaml</a>          | agent                 | <a href='https://huggingface.co/datasets/nvidia/Nemotron-RL-math-advanced_calculations'>Nemotron-RL-math-advanced_calculations</a>                             | An instruction following math environment with counter-intuitive calculators                                                       | Improve instruction following capabilities in specific math environments                                                      | ✓     | -          | Apache 2.0                                                |
| Math Formal Lean           | <a href='resources_servers/math_formal_lean/configs/math_formal_lean.yaml'>math_formal_lean.yaml</a>                                        | math                  | -                                                                                                                                                              | Lean4 formal proof verification environment                                                                                        | Improve formal theorem proving capabilities                                                                                   | ✓     | -          | MIT                                                       |
| Math Formal Lean           | <a href='resources_servers/math_formal_lean/configs/math_formal_lean_multi_turn.yaml'>math_formal_lean_multi_turn.yaml</a>                  | math                  | -                                                                                                                                                              | Lean4 formal proof verification environment with multi-turn self-correction                                                        | Improve formal theorem proving capabilities                                                                                   | ✓     | -          | MIT                                                       |
| Math Formal Lean           | <a href='resources_servers/math_formal_lean/configs/nemotron_clean_easy.yaml'>nemotron_clean_easy.yaml</a>                                  | math                  | -                                                                                                                                                              | Lean4 formal proof verification environment                                                                                        | Improve formal theorem proving capabilities                                                                                   | ✓     | -          | Apache 2.0                                                |
| Math Formal Lean           | <a href='resources_servers/math_formal_lean/configs/nemotron_first_try_hard.yaml'>nemotron_first_try_hard.yaml</a>                          | math                  | -                                                                                                                                                              | Lean4 formal proof verification environment                                                                                        | Improve formal theorem proving capabilities                                                                                   | ✓     | -          | Apache 2.0                                                |
| Math Formal Lean           | <a href='resources_servers/math_formal_lean/configs/nemotron_medium_500.yaml'>nemotron_medium_500.yaml</a>                                  | math                  | -                                                                                                                                                              | Lean4 formal proof verification environment                                                                                        | Improve formal theorem proving capabilities                                                                                   | ✓     | -          | Apache 2.0                                                |
| Math Formal Lean           | <a href='resources_servers/math_formal_lean/configs/nemotron_very_easy.yaml'>nemotron_very_easy.yaml</a>                                    | math                  | -                                                                                                                                                              | Lean4 formal proof verification environment                                                                                        | Improve formal theorem proving capabilities                                                                                   | ✓     | -          | Apache 2.0                                                |
| Math With Code             | <a href='resources_servers/math_with_code/configs/math_with_code.yaml'>math_with_code.yaml</a>                                              | math                  | -                                                                                                                                                              | -                                                                                                                                  | -                                                                                                                             | ✓     | -          | Apache 2.0                                                |
| Math With Judge            | <a href='resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml'>bytedtsinghua_dapo17k.yaml</a>                               | math                  | -                                                                                                                                                              | -                                                                                                                                  | -                                                                                                                             | ✓     | ✓          | Apache 2.0                                                |
| Math With Judge            | <a href='resources_servers/math_with_judge/configs/dapo17k.yaml'>dapo17k.yaml</a>                                                           | math                  | -                                                                                                                                                              | -                                                                                                                                  | -                                                                                                                             | ✓     | ✓          | Apache 2.0                                                |
| Math With Judge            | <a href='resources_servers/math_with_judge/configs/dapo17k_filtered_qwen330ba3binstruct.yaml'>dapo17k_filtered_qwen330ba3binstruct.yaml</a> | math                  | -                                                                                                                                                              | -                                                                                                                                  | -                                                                                                                             | ✓     | ✓          | Apache 2.0                                                |
| Math With Judge            | <a href='resources_servers/math_with_judge/configs/dapo17k_trajectory_collection.yaml'>dapo17k_trajectory_collection.yaml</a>               | math                  | -                                                                                                                                                              | -                                                                                                                                  | -                                                                                                                             | -     | ✓          | -                                                         |
| Math With Judge            | <a href='resources_servers/math_with_judge/configs/math_stack_overflow.yaml'>math_stack_overflow.yaml</a>                                   | math                  | <a href='https://huggingface.co/datasets/nvidia/Nemotron-RL-math-stack_overflow'>Nemotron-RL-math-stack_overflow</a>                                           | -                                                                                                                                  | -                                                                                                                             | ✓     | ✓          | Creative Commons Attribution-ShareAlike 4.0 International |
| Math With Judge            | <a href='resources_servers/math_with_judge/configs/math_with_judge.yaml'>math_with_judge.yaml</a>                                           | math                  | <a href='https://huggingface.co/datasets/nvidia/Nemotron-RL-math-OpenMathReasoning'>Nemotron-RL-math-OpenMathReasoning</a>                                     | Math dataset with math-verify and LLM-as-a-judge                                                                                   | Improve math capabilities including AIME 24 / 25                                                                              | ✓     | ✓          | Creative Commons Attribution 4.0 International            |
| Math With Judge            | <a href='resources_servers/math_with_judge/configs/math_with_local_judge.yaml'>math_with_local_judge.yaml</a>                               | math                  | -                                                                                                                                                              | -                                                                                                                                  | -                                                                                                                             | -     | -          | -                                                         |
| Mcqa                       | <a href='resources_servers/mcqa/configs/mcqa.yaml'>mcqa.yaml</a>                                                                            | knowledge             | <a href='https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-mcqa'>Nemotron-RL-knowledge-mcqa</a>                                                     | Multi-choice question answering problems                                                                                           | Improve benchmarks like MMLU / GPQA / HLE                                                                                     | ✓     | -          | Apache 2.0                                                |
| Mini Swe Agent             | <a href='resources_servers/mini_swe_agent/configs/mini_swe_agent.yaml'>mini_swe_agent.yaml</a>                                              | coding                | <a href='https://huggingface.co/datasets/SWE-Gym/SWE-Gym'>SWE-Gym</a>                                                                                          | A software development with mini-swe-agent orchestration                                                                           | Improve software development capabilities, like SWE-bench                                                                     | ✓     | ✓          | MIT                                                       |
| Multichallenge             | <a href='resources_servers/multichallenge/configs/multichallenge.yaml'>multichallenge.yaml</a>                                              | knowledge             | -                                                                                                                                                              | MultiChallenge benchmark evaluation with LLM judge                                                                                 | -                                                                                                                             | ✓     | -          | TBD                                                       |
| Multichallenge             | <a href='resources_servers/multichallenge/configs/multichallenge_nrl.yaml'>multichallenge_nrl.yaml</a>                                      | knowledge             | -                                                                                                                                                              | MultiChallenge benchmark evaluation with LLM judge                                                                                 | -                                                                                                                             | ✓     | -          | TBD                                                       |
| Newton Bench               | <a href='resources_servers/newton_bench/configs/newton_bench.yaml'>newton_bench.yaml</a>                                                    | math                  | -                                                                                                                                                              | -                                                                                                                                  | -                                                                                                                             | ✓     | -          | Apache 2.0                                                |
| Ns Tools                   | <a href='resources_servers/ns_tools/configs/ns_tools.yaml'>ns_tools.yaml</a>                                                                | agent                 | -                                                                                                                                                              | NeMo Skills tool execution with math verification                                                                                  | -                                                                                                                             | -     | -          | -                                                         |
| Reasoning Gym              | <a href='resources_servers/reasoning_gym/configs/reasoning_gym.yaml'>reasoning_gym.yaml</a>                                                 | knowledge             | -                                                                                                                                                              | -                                                                                                                                  | -                                                                                                                             | ✓     | -          | Apache 2.0                                                |
| Reasoning Gym              | <a href='resources_servers/reasoning_gym/configs/resources_only.yaml'>resources_only.yaml</a>                                               | knowledge             | -                                                                                                                                                              | -                                                                                                                                  | -                                                                                                                             | -     | -          | -                                                         |
| Spider2 Lite               | <a href='resources_servers/spider2_lite/configs/spider2_lite.yaml'>spider2_lite.yaml</a>                                                    | coding                | -                                                                                                                                                              | Text-to-SQL with execution-based evaluation on Spider 2.0-Lite (135 SQLite tasks). Binary reward based on result-set equivalence.
 | Improve text-to-SQL capabilities for real-world enterprise queries using execution-based binary reward without an LLM judge.
 | -     | ✓          | -                                                         |
| Structured Outputs         | <a href='resources_servers/structured_outputs/configs/structured_outputs_json.yaml'>structured_outputs_json.yaml</a>                        | instruction_following | <a href='https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following-structured_outputs'>Nemotron-RL-instruction_following-structured_outputs</a> | Check if responses are following structured output requirements in prompts                                                         | Improve instruction following capabilities                                                                                    | ✓     | ✓          | Apache 2.0                                                |
| Swerl Gen                  | <a href='resources_servers/swerl_gen/configs/swerl_gen.yaml'>swerl_gen.yaml</a>                                                             | coding                | -                                                                                                                                                              | Running sandboxed evaluation for SWE-style tasks (either patch generation or reproduction test generation)                         | Improve SWE capabilities useful for benchmarks like SWE-bench                                                                 | ✓     | ✓          | Apache 2.0                                                |
| Swerl Llm Judge            | <a href='resources_servers/swerl_llm_judge/configs/swerl_llm_judge.yaml'>swerl_llm_judge.yaml</a>                                           | coding                | -                                                                                                                                                              | SWE-style multiple-choice LLM-judge tasks scored via <solution>...</solution> choice.                                              | Improve SWE capabilities useful for benchmarks like SWE-bench                                                                 | ✓     | ✓          | MIT                                                       |
| Terminus Judge             | <a href='resources_servers/terminus_judge/configs/terminus_judge.yaml'>terminus_judge.yaml</a>                                              | agent                 | -                                                                                                                                                              | single-step terminal based task                                                                                                    | Improve on terminal-style tasks                                                                                               | ✓     | ✓          | Apache 2.0                                                |
| Text To Sql                | <a href='resources_servers/text_to_sql/configs/text_to_sql.yaml'>text_to_sql.yaml</a>                                                       | coding                | -                                                                                                                                                              | Text-to-SQL generation with LLM-as-a-judge equivalence checking                                                                    | Improve text-to-SQL capabilities across multiple dialects                                                                     | -     | -          | -                                                         |
| Workplace Assistant        | <a href='resources_servers/workplace_assistant/configs/workplace_assistant.yaml'>workplace_assistant.yaml</a>                               | agent                 | <a href='https://huggingface.co/datasets/nvidia/Nemotron-RL-agent-workplace_assistant'>Nemotron-RL-agent-workplace_assistant</a>                               | Workplace assistant multi-step tool-using environment                                                                              | Improve multi-step tool use capability                                                                                        | ✓     | ✓          | Apache 2.0                                                |
| Xlam Fc                    | <a href='resources_servers/xlam_fc/configs/xlam_fc.yaml'>xlam_fc.yaml</a>                                                                   | agent                 | -                                                                                                                                                              | -                                                                                                                                  | -                                                                                                                             | ✓     | ✓          | Apache 2.0                                                |
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