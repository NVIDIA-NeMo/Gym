(environment-aviary)=

# Aviary

This tutorial guides you through integrating with [Future-House/aviary](https://github.com/Future-House/aviary), a gymnasium for defining custom language agent RL environments.

:::{card}

**Goal**: Run Aviary environments in NeMo Gym for training and inference.

^^^

**In this tutorial, you will**:

1. Install Aviary dependencies
2. Configure and launch servers
3. Collect rollouts with a pre-built environment

:::

## What is Aviary?

Aviary is a framework for building custom RL environments with tool use and multi-step reasoning. Environments built in Aviary can run through NeMo Gym for training and inference. This integration includes environments for math (GSM8K), multi-hop question answering (HotPotQA), and scientific notebook execution (BixBench). The upstream Aviary library includes additional environments. Refer to the [Aviary GitHub repository](https://github.com/Future-House/aviary) for a complete list.

:::{tip}
**Key terminology**:
- **Resources server**: A NeMo Gym component that manages environment instances and provides the `/seed_session`, `/step`, and `/verify` endpoints.
- **Rollout**: A complete episode of agent-environment interaction, from initial observation to terminal state.

Unlike typical NeMo Gym environments, Aviary environments handle tool execution internally. The resources server wraps the Aviary environment to expose it through the NeMo Gym API.
:::

---

## Before You Start

Ensure you have:

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) package manager installed
- A CUDA-capable GPU with sufficient VRAM for your chosen model (minimum 8 GB for Qwen3-4B)
- The NeMo Gym repository cloned locally

:::{note}
BixBench environments require Docker for notebook execution.
:::

### Install Dependencies

Install the Aviary package from the NeMo Gym repository root:

```bash
uv venv
source .venv/bin/activate
uv sync
uv add "fhaviary[gsm8k,hotpotqa,notebook,llm]>=0.28.0"
```

:::{note}
The package is `fhaviary` (Future House Aviary), not `aviary-gymnasium`. The extras `[gsm8k,hotpotqa,notebook,llm]` enable the specific environment integrations.
:::

### Available Environments

The integration includes several pre-built Aviary environments:

| Environment | File | Description |
|-------------|------|-------------|
| **GSM8K** | `gsm8k_app.py` | Grade school math problems with calculator tool |
| **HotPotQA** | `hotpotqa_app.py` | Multi-hop question answering |
| **BixBench** | `notebook_app.py` | Jupyter notebook execution for scientific tasks |
| **Client/Proxy** | `client_app.py` | Generic interface to remote Aviary dataset servers |

---

## Tutorial Steps

1. **Configure Model Server**

   Create `env.yaml` at the repository root:

   ```yaml
   policy_base_url: "http://localhost:8000/v1"
   policy_api_key: "dummy"
   policy_model_name: "Qwen/Qwen3-4B-Instruct-2507"
   ```

   :::{note}
   For production deployments, use environment variables instead of hardcoding credentials. The `policy_api_key` field accepts a placeholder when authentication is not required.
   :::

2. **Start Model Server**

   Start a vLLM server with tool-calling support:

   ```bash
   uv add vllm
   vllm serve Qwen/Qwen3-4B-Instruct-2507 \
     --max-model-len 32768 \
     --enable-auto-tool-choice \
     --tool-call-parser hermes
   ```

   :::{tip}
   The model must support tool calling. The `--enable-auto-tool-choice` and `--tool-call-parser` flags are required for Aviary environments. Different models may require different parsers. Refer to the [vLLM documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#tool-calling-in-the-chat-completion-api) for supported configurations.
   :::

3. **Launch NeMo Gym Servers**

   Start the GSM8K Aviary resources server:

   ```bash
   ng_run "+config_paths=[resources_servers/aviary/configs/gsm8k_aviary.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
   ```

   **Expected output** (ports may vary based on configuration):

   ```text
   INFO:     Starting servers...
   INFO:     gsm8k_aviary_resources_server running on http://0.0.0.0:8001
   INFO:     gsm8k_aviary_agent running on http://0.0.0.0:8002
   INFO:     Press Ctrl+C to stop
   ```

4. **Collect Rollouts**

   In another terminal:

   ```bash
   ng_collect_rollouts \
       +agent_name=gsm8k_aviary_agent \
       +input_jsonl_fpath=resources_servers/aviary/data/example.jsonl \
       +output_jsonl_fpath=resources_servers/aviary/data/example_rollouts.jsonl
   ```

   | Option | Description |
   |--------|-------------|
   | `+limit=N` | Process first N examples only |
   | `+num_repeats=K` | Run each example K times for variance |
   | `+num_samples_in_parallel=P` | Limit concurrent requests to P |

---

## Reference

- [Aviary GitHub](https://github.com/Future-House/aviary) - Official Aviary repository
- [Aviary Paper](https://arxiv.org/abs/2412.21154) - Training language agents on challenging scientific tasks
- `resources_servers/aviary/` - NeMo Gym resources server implementations
- `responses_api_agents/aviary_agent/` - NeMo Gym Aviary agent integration

---

## Next Steps

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Multi-Step Environments
:link: /environment-tutorials/multi-step
:link-type: doc
Build sequential tool-calling workflows.
:::

:::{grid-item-card} {octicon}`law;1.5em;sd-mr-1` LLM-as-a-Judge
:link: /environment-tutorials/llm-as-judge
:link-type: doc
Use LLMs for flexible verification.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Train with NeMo RL
:link: training-nemo-rl-grpo-index
:link-type: ref
Start training models on your environment.
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Verifiers Integration
:link: verifiers
:link-type: doc
Explore 600+ environments from Prime Intellect.
:::

::::
