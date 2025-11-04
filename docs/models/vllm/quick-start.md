(models-vllm-quick-start)=

# Quick Start

Get vLLM running with NeMo Gym in under 5 minutes.

---

## Before You Start

Choose your path based on whether you need to start a vLLM server or already have one running.

::::{tab-set}

:::{tab-item} I need to start a vLLM server
**Step 1: Install vLLM**

```bash
# Create a virtual environment
uv venv --python 3.12 --seed
source .venv/bin/activate

# Install vLLM with dependencies
uv pip install hf_transfer datasets vllm --torch-backend=auto
```

**Step 2: Download a model**

```bash
# Example: Download Qwen3-30B-A3B (supports tool calling)
HF_HOME=.cache/ \
HF_HUB_ENABLE_HF_TRANSFER=1 \
    hf download Qwen/Qwen3-30B-A3B
```

**Step 3: Start vLLM server**

```bash
HF_HOME=.cache/ \
HOME=. \
vllm serve \
    Qwen/Qwen3-30B-A3B \
    --dtype auto \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --host 0.0.0.0 \
    --port 10240
```

```{important}
**Do NOT use vLLM's reasoning parser** (e.g., `--reasoning-parser qwen3`). NeMo Gym's vLLM adapter handles reasoning token parsing to maintain compatibility with the Responses API format.
```

**✅ Success check**: Visit `http://localhost:10240/health` - you should see a health status response.

**Step 4: Configure NeMo Gym**

Create `env.yaml` in your NeMo Gym repository:

```yaml
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: Qwen/Qwen3-30B-A3B
```

**Step 5: Start NeMo Gym servers**

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/simple_weather/configs/simple_weather.yaml"

ng_run "+config_paths=[${config_paths}]"
```

**✅ Success check**: You should see multiple servers starting, including the head server on port 11000.

:::

:::{tab-item} I have a vLLM server running
**Step 1: Get your vLLM endpoint details**

You need:
- vLLM server URL (e.g., `http://your-server:8000/v1`)
- Model name loaded in vLLM (e.g., `meta-llama/Llama-3.1-8B-Instruct`)
- API key (use `EMPTY` if authentication not configured)

**Step 2: Configure NeMo Gym**

Create or update `env.yaml` in your repository root:

```yaml
policy_base_url: http://your-vllm-server:8000/v1
policy_api_key: EMPTY  # or your API key
policy_model_name: meta-llama/Llama-3.1-8B-Instruct
```

**Step 3: Start NeMo Gym servers**

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/simple_weather/configs/simple_weather.yaml"

ng_run "+config_paths=[${config_paths}]"
```

**✅ Success check**: Verify servers start without connection errors.

**Step 4: Test the integration**

```bash
ng_test +entrypoint=responses_api_models/vllm_model
```

**✅ Success check**: All tests should pass.

:::

::::

---

## What's Next?

Now that vLLM is configured, you can:

- **[Collect rollouts](../../get-started/collecting-rollouts.md)** for training data generation
- **[Configure additional parameters](configuration.md)** like token IDs for training
- **[Set up load balancing](optimization.md)** for production throughput
- **[Switch from OpenAI](optimization.md#switch-from-openai-to-vllm)** if migrating from API-based models

:::{tip}
**New to NeMo Gym?** Complete the [Setup and Installation](../../get-started/setup-installation.md) tutorial first to understand the full workflow.
:::

