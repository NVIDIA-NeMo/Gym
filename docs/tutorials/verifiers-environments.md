(using-verifiers-environments)=

# Using Verifiers Environments

Learn how to run environments from Prime Intellect's [Environments Hub](https://app.primeintellect.ai/dashboard/environments) within NeMo Gym.

::::{grid} 2
:gutter: 3

:::{grid-item-card} {octicon}`clock;1em;` **Time**
20 minutes
:::

:::{grid-item-card} {octicon}`bookmark;1em;` **Prerequisites**

- Completed {doc}`../get-started/detailed-setup`

:::

::::

---

## What is this integration?

The verifiers integration enables NeMo Gym to use pre-built environments from Prime Intellect's Environments Hub. Unlike typical NeMo Gym environments that require a separate resource server, verifiers environments handle state management, verification, and tool execution internally.

**Key differences:**
- No resource server needed - verification logic is built into the environment
- Uses an agent server that wraps verifiers environments

**Available environments include:**
- `primeintellect/acereason-math` - Mathematical reasoning with chain-of-thought
- `kalomaze/alphabet-sort` - Multi-turn alphabetical sorting
- `primeintellect/ascii-tree` - ASCII tree generation
- And [many more (600+) on Environments Hub](https://app.primeintellect.ai/dashboard/environments)

:::{note}
**Multi-turn environments:** Currently require disabling `enforce_monotonicity` in training configuration until token propagation is fully patched.
:::

---

## 1. Install Dependencies

Install verifiers and prime tools in the main Gym environment:

```bash
# From the Gym repository root
uv venv
source .venv/bin/activate
uv sync
uv add verifiers
uv add tool prime
```

Install the acereason-math environment:

```bash
prime env install primeintellect/acereason-math
```

:::{tip}
The agent server's virtual environment is automatically created by `ng_run` - you don't need to build it manually.
:::

---

## 2. Create Example Dataset

Generate example tasks using the provided helper script:

```bash
python3 responses_api_agents/verifiers_agent/scripts/create_dataset.py \
  --env-id primeintellect/acereason-math \
  --size 5 \
  --output responses_api_agents/verifiers_agent/data/acereason-math-example.jsonl
```

This creates a dataset with 5 example tasks from the environment.

---

## 3. Update Agent Requirements

Add the environment package to `responses_api_agents/verifiers_agent/requirements.txt`:

```txt
-e nemo-gym[dev] @ ../../
verifiers>=0.1.9
--extra-index-url https://hub.primeintellect.ai/primeintellect/simple/
acereason-math
```

---

## 4. Configure Model Server

Set up your model server configuration in `env.yaml` at the repository root:

```yaml
policy_base_url: "http://localhost:8000/v1"
policy_api_key: "dummy"
policy_model_name: "Qwen/Qwen3-4B-Instruct-2507"
```

---

## 5. Start the Model Server

Serve your model with vLLM. Ensure the context length exceeds your generation length:

```bash
uv add vllm
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

:::{tip}
The `--reasoning-parser qwen3` flag enables the model to generate chain-of-thought reasoning tokens, which are required for acereason-math.
:::

---

## 6. Launch NeMo Gym Servers

Start the verifiers agent and model server:

```bash
ng_run "+config_paths=[responses_api_agents/verifiers_agent/configs/verifiers_acereason-math.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

This launches:
- **Verifiers agent** on port 5000 - wraps the acereason-math environment
- **vLLM model server proxy** - connects to your running vLLM server

---

## 7. Collect Rollouts

Generate rollouts using the example dataset:

```bash
ng_collect_rollouts \
    +agent_name=verifiers_agent \
    +input_jsonl_fpath=responses_api_agents/verifiers_agent/data/acereason-math-example.jsonl \
    +output_jsonl_fpath=responses_api_agents/verifiers_agent/data/acereason-math-example-rollouts.jsonl \
    +limit=5
```

View a rollout in the terminal:

```bash
tail -n 1 responses_api_agents/verifiers_agent/data/acereason-math-example-rollouts.jsonl | jq | less
```

---

## Understanding the Configuration

The verifiers agent configuration (`configs/verifiers_acereason-math.yaml`) specifies:

```yaml
verifiers_agent:
  responses_api_agents:
    verifiers_agent:
      entrypoint: app.py
      model_server:
        type: responses_api_models
        name: policy_model
      model_name: ""
      vf_env_id: acereason-math          # The verifiers environment ID
      vf_env_args: {}                    # Environment-specific arguments
      group_size: 1                      # Rollouts per example
      max_concurrent_generation: -1      # Unlimited concurrency
      max_concurrent_scoring: -1
      max_tokens: 8192                   # Max generation length
      temperature: 1.0
      top_p: 1.0
```

**Key parameters:**
- `vf_env_id`: The environment identifier from Environments Hub
- `vf_env_args`: Optional environment-specific configuration
- `max_tokens`: Must be less than model server's `max_model_len`

---

## Adding New Environments

To use a different environment from Environments Hub:

### 1. Install the environment package

```bash
prime env install primeintellect/ascii-tree
```

### 2. Generate example data

Use the provided helper script:

```bash
python3 responses_api_agents/verifiers_agent/scripts/create_dataset.py \
  --env-id primeintellect/ascii-tree \
  --size 5 \
  --output responses_api_agents/verifiers_agent/data/ascii-tree-example.jsonl
```

### 3. Update requirements.txt

Add the environment package to `responses_api_agents/verifiers_agent/requirements.txt`:

```txt
-e nemo-gym[dev] @ ../../
verifiers>=0.1.9
--extra-index-url https://hub.primeintellect.ai/primeintellect/simple/
ascii-tree
```

### 4. Create a new config file

Copy an existing config and update the `vf_env_id`:

```bash
cp responses_api_agents/verifiers_agent/configs/verifiers_acereason-math.yaml \
   responses_api_agents/verifiers_agent/configs/verifiers_ascii-tree.yaml
# Edit the file to change vf_env_id to "ascii-tree"
```

### 5. Collect rollouts

```bash
# ng_run will automatically build the agent's venv and install requirements
ng_run "+config_paths=[responses_api_agents/verifiers_agent/configs/verifiers_ascii-tree.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

ng_collect_rollouts \
    +agent_name=verifiers_agent \
    +input_jsonl_fpath=responses_api_agents/verifiers_agent/data/ascii-tree-example.jsonl \
    +output_jsonl_fpath=responses_api_agents/verifiers_agent/data/ascii-tree-example-rollouts.jsonl \
    +limit=5
```

---

## Training with Verifiers Environments

Training works the same as with standard NeMo Gym environments. Use the generated datasets with your preferred training framework:

- {doc}`nemo-rl-grpo/index` for GRPO training
- {doc}`unsloth-training` for efficient single-GPU training
- {doc}`offline-training-w-rollouts` for SFT/DPO

For multi-environment training, create separate agent instances for each environment and blend the datasets.

---

## Reference

- [Prime Intellect Environments Hub](https://app.primeintellect.ai/dashboard/environments) - Browse available environments
- [Verifiers GitHub](https://github.com/PrimeIntellect-ai/verifiers) - Verifiers library documentation
- {doc}`creating-resource-server` - For building custom NeMo Gym environments
- {doc}`../contribute/environments/new-environment` - Environment contribution guide
