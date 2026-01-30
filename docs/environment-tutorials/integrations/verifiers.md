(environment-verifiers)=

# Verifiers

This tutorial guides you through integrating with [PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers), enabling environments from Prime Intellect's Environments Hub to run in NeMo Gym.

:::{card}

**Goal**: Run Verifiers environments in NeMo Gym for training and inference.

^^^

**In this tutorial, you will**:

1. Install Verifiers dependencies
2. Create a dataset from an environment
3. Configure and launch servers
4. Collect rollouts

:::

## What is Verifiers?

Verifiers is a framework that provides 600+ environments across reasoning, math, and agent tasks through the Environments Hub. Environments built for Environments Hub can be deployed through NeMo Gym for training with NeMo RL.

:::{tip}
**Key terminology**:
- **Environments Hub**: Prime Intellect's registry of pre-built RL environments accessible via the `prime` CLI tool.
- **Rollout**: A complete episode of agent-environment interaction, from initial observation to terminal state.

Unlike typical NeMo Gym environments, Verifiers environments handle state management, verification, and tool execution internally without requiring a separate resource server.
:::

:::{note}
**Multi-turn environments**: Currently require disabling `enforce_monotonicity` in training configuration until token propagation is fully patched.
:::

---

## Before You Start

Ensure you have:

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) package manager installed
- A CUDA-capable GPU with sufficient VRAM for your chosen model (minimum 8 GB for Qwen3-4B)
- The NeMo Gym repository cloned locally

### Install Dependencies

Install Verifiers and the Prime CLI from the NeMo Gym repository root:

```bash
uv venv
source .venv/bin/activate
uv sync
uv add verifiers
uv tool install prime
```

Install an environment from the Environments Hub:

```bash
prime env install primeintellect/acereason-math
```

:::{note}
Browse available environments at the [Prime Intellect Environments Hub](https://app.primeintellect.ai/dashboard/environments). Each environment can be installed using `prime env install <env-id>`.
:::

### Available Environments

The Environments Hub includes 600+ environments. Example categories:

| Category | Example Environment | Description |
|----------|---------------------|-------------|
| **Math Reasoning** | `primeintellect/acereason-math` | Mathematical problem solving |
| **Code Generation** | `primeintellect/code-*` | Programming challenges |
| **Agent Tasks** | `primeintellect/agent-*` | Multi-step reasoning tasks |

---

## Tutorial Steps

1. **Create Dataset**

   Generate example tasks from your installed environment:

   ```bash
   python3 responses_api_agents/verifiers_agent/scripts/create_dataset.py \
     --env-id primeintellect/acereason-math \
     --size 5 \
     --output responses_api_agents/verifiers_agent/data/acereason-math-example.jsonl
   ```

2. **Update Agent Requirements**

   Add the environment to `responses_api_agents/verifiers_agent/requirements.txt`:

   ```txt
   -e nemo-gym[dev] @ ../../
   verifiers>=0.1.9
   --extra-index-url https://hub.primeintellect.ai/primeintellect/simple/
   acereason-math
   ```

   :::{tip}
   The `--extra-index-url` line is required to access Prime Intellect's package registry. Replace `acereason-math` with your chosen environment package name.
   :::

3. **Configure Model Server**

   Create `env.yaml` at the repository root:

   ```yaml
   policy_base_url: "http://localhost:8000/v1"
   policy_api_key: "dummy"
   policy_model_name: "Qwen/Qwen3-4B-Instruct-2507"
   ```

   :::{note}
   For production deployments, use environment variables instead of hardcoding credentials. The `policy_api_key` field accepts a placeholder when authentication is not required.
   :::

4. **Start Model Server**

   Start a vLLM server with tool-calling support:

   ```bash
   uv add vllm
   vllm serve Qwen/Qwen3-4B-Instruct-2507 \
     --max-model-len 32768 \
     --reasoning-parser qwen3 \
     --enable-auto-tool-choice \
     --tool-call-parser hermes
   ```

   :::{tip}
   The `--reasoning-parser qwen3` flag enables Qwen3's extended thinking mode. Different models may require different parsers. Refer to the [vLLM documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#tool-calling-in-the-chat-completion-api) for supported configurations.
   :::

5. **Launch NeMo Gym Servers**

   Start the Verifiers resources server:

   ```bash
   ng_run "+config_paths=[responses_api_agents/verifiers_agent/configs/acereason-math.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
   ```

6. **Collect Rollouts**

   In another terminal:

   ```bash
   ng_collect_rollouts \
       +agent_name=verifiers_agent \
       +input_jsonl_fpath=responses_api_agents/verifiers_agent/data/acereason-math-example.jsonl \
       +output_jsonl_fpath=responses_api_agents/verifiers_agent/data/acereason-math-example-rollouts.jsonl \
       +limit=5
   ```

   | Option | Description |
   |--------|-------------|
   | `+limit=N` | Process first N examples only |
   | `+num_repeats=K` | Run each example K times for variance |
   | `+num_samples_in_parallel=P` | Limit concurrent requests to P |

---

## Reference

- [Prime Intellect Environments Hub](https://app.primeintellect.ai/dashboard/environments) - Browse 600+ available environments
- [Verifiers GitHub](https://github.com/PrimeIntellect-ai/verifiers) - Official Verifiers repository
- `responses_api_agents/verifiers_agent/` - NeMo Gym agent integration

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

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Aviary Integration
:link: aviary
:link-type: doc
Scientific and reasoning agent environments.
:::

::::
