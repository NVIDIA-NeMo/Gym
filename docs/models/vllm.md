(models-vllm)=

# vLLM - Self-Hosted Open Models

Configure and deploy vLLM for high-throughput inference with self-hosted open models in NeMo Gym.

---

## Overview

vLLM is a fast and memory-efficient inference engine for large language models. NeMo Gym's vLLM model server adapter enables you to deploy open-source models on your own infrastructure with optimized throughput for large-scale agent training.

**Why use vLLM with NeMo Gym?**

- **High throughput**: Optimized batch processing for generating thousands of rollouts concurrently
- **Full control**: Deploy any open-source model on your infrastructure
- **Cost-effective**: No per-token API costs for large-scale training workloads
- **Training-ready**: Built-in support for token IDs and log probabilities required by RL frameworks
- **Reasoning support**: Automatic parsing and extraction of reasoning tokens using `<think>` tags

**When to use vLLM**:

- Training with open-source models (Llama, Qwen, Mistral, etc.)
- Large-scale rollout collection requiring high throughput
- Cost-sensitive projects with predictable infrastructure costs
- Research requiring full model control and reproducibility

:::{seealso}
**Not sure which model serving method to use?** See the [Models Overview](index.md#choosing-a-model-serving-method) for a comparison.
:::

---

## Quick Start

Get vLLM running with NeMo Gym in under 5 minutes.

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

:::{important}
**Do NOT use vLLM's reasoning parser** (e.g., `--reasoning-parser qwen3`). NeMo Gym's vLLM adapter handles reasoning token parsing to maintain compatibility with the Responses API format.
:::

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

:::{tip}
**New to NeMo Gym?** Complete the [Setup and Installation](../get-started/setup-installation.md) tutorial first to understand the full workflow.
:::

---

## Configuration Reference

All configuration options for the vLLM model server adapter.

### Configuration File Structure

```yaml
# responses_api_models/vllm_model/configs/vllm_model.yaml
policy_model:
  responses_api_models:
    vllm_model:
      entrypoint: app.py
      base_url: ${policy_base_url}
      api_key: ${policy_api_key}
      model: ${policy_model_name}
      return_token_id_information: false
      uses_reasoning_parser: true
      replace_developer_role_with_system: false
```

### Configuration Parameters

```{list-table}
:header-rows: 1
:widths: 20 15 15 50

* - Parameter
  - Type
  - Default
  - Description
* - `base_url`
  - `str` or `list[str]`
  - Required
  - vLLM server endpoint(s). Single URL: `http://localhost:8000/v1`. List for load balancing: `["http://server1:8000/v1", "http://server2:8000/v1"]`
* - `api_key`
  - `str`
  - Required
  - Authentication key. Use `EMPTY` if vLLM server has no authentication configured.
* - `model`
  - `str`
  - Required
  - Model identifier matching what's loaded in vLLM (e.g., `meta-llama/Llama-3.1-8B-Instruct`).
* - `return_token_id_information`
  - `bool`
  - `false`
  - Include token IDs and log probabilities in responses. Set to `true` for RL training with token-level information.
* - `uses_reasoning_parser`
  - `bool`
  - `true`
  - Enable automatic parsing of reasoning tokens using `<think>` tags. Converts between Responses API reasoning format and Chat Completions content.
* - `replace_developer_role_with_system`
  - `bool`
  - `false`
  - Convert `developer` role messages to `system` role for models that don't support the developer role.
```

### Environment Variables

Define these in `env.yaml`:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Variable
  - Description
* - `policy_base_url`
  - vLLM server endpoint URL including `/v1` path
* - `policy_api_key`
  - API key for vLLM server authentication
* - `policy_model_name`
  - Model identifier loaded in vLLM server
```

:::{dropdown} Example: Complete configuration for training mode

```yaml
# env.yaml
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: Qwen/Qwen3-30B-A3B

# responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
policy_model:
  responses_api_models:
    vllm_model:
      entrypoint: app.py
      base_url: ${policy_base_url}
      api_key: ${policy_api_key}
      model: ${policy_model_name}
      return_token_id_information: true  # Enable for training
      uses_reasoning_parser: true
```

This configuration includes token IDs and log probabilities required by RL training frameworks.
:::

---

## Common Tasks

### Collect Rollouts for Training

Generate training data using your vLLM model:

```bash
ng_collect_rollouts \
  +agent_name=simple_weather_simple_agent \
  +input_jsonl_fpath=resources_servers/simple_weather/data/example.jsonl \
  +output_jsonl_fpath=results/vllm_rollouts.jsonl \
  +limit=100 \
  +concurrency=50
```

**What this does**:
- Processes 100 examples from the input dataset
- Runs 50 concurrent agent interactions for high throughput
- Saves rollouts with verification scores to `results/vllm_rollouts.jsonl`

:::{seealso}
See [Collecting Rollouts](../get-started/collecting-rollouts.md) for a complete walkthrough.
:::

### Use vLLM for Both Policy and Judge Models

::::{tab-set}

:::{tab-item} Scenario: Single vLLM server

Use the same vLLM model for both policy decisions and verification:

```yaml
# env.yaml
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: Qwen/Qwen3-30B-A3B

# Use same values for judge
judge_base_url: http://localhost:10240/v1
judge_api_key: EMPTY
judge_model_name: Qwen/Qwen3-30B-A3B
```

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
responses_api_models/vllm_model/configs/vllm_judge_model.yaml,\
resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml"

ng_run "+config_paths=[${config_paths}]"
```

:::

:::{tab-item} Scenario: Separate models

Use different vLLM models for policy and judge:

```yaml
# env.yaml
# Policy model
policy_base_url: http://policy-server:8000/v1
policy_api_key: EMPTY
policy_model_name: meta-llama/Llama-3.1-70B-Instruct

# Judge model
judge_base_url: http://judge-server:8000/v1
judge_api_key: EMPTY
judge_model_name: Qwen/Qwen3-30B-A3B
```

This enables using a larger model for policy and a smaller/faster model for verification.

:::

::::

:::{seealso}
See [Separate Policy and Judge Models](../tutorials/separate-policy-and-judge-models.md) for advanced scenarios.
:::

### Enable Load Balancing Across Multiple vLLM Servers

Distribute inference requests across multiple vLLM instances for higher throughput:

```yaml
# env.yaml - Define as comma-separated list
policy_base_url: http://vllm-1:8000/v1,http://vllm-2:8000/v1,http://vllm-3:8000/v1
policy_api_key: EMPTY
policy_model_name: meta-llama/Llama-3.1-8B-Instruct
```

Or use YAML list syntax in the config file:

```yaml
# Custom config file
policy_model:
  responses_api_models:
    vllm_model:
      entrypoint: app.py
      base_url:
        - http://vllm-1:8000/v1
        - http://vllm-2:8000/v1
        - http://vllm-3:8000/v1
      api_key: ${policy_api_key}
      model: ${policy_model_name}
```

**How it works**: NeMo Gym uses session-based routing - each client session is assigned to one vLLM server and all requests from that session route to the same server. New sessions are distributed round-robin across available servers.

:::{tip}
Load balancing is most effective when:
- Running high concurrency rollout collection (`+concurrency=100+`)
- Each vLLM server has similar capacity
- All servers have the same model loaded
:::

### Switch from OpenAI to vLLM

Already using OpenAI? Switch to vLLM by changing one configuration file:

**Before** (OpenAI):
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/simple_weather/configs/simple_weather.yaml"
```

**After** (vLLM):
```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/simple_weather/configs/simple_weather.yaml"
```

Update `env.yaml` to point to your vLLM server, and everything else remains the same.

---

## Advanced Topics

:::{dropdown} How reasoning token parsing works

NeMo Gym's vLLM adapter handles conversion between two formats:

**Responses API format** (what agents use):
```python
{
  "type": "reasoning",
  "summary": [{"text": "thinking about the problem...", "type": "summary_text"}]
}
```

**Chat Completions format** (what vLLM uses):
```python
{
  "role": "assistant",
  "content": "<think>thinking about the problem...</think>actual response"
}
```

The adapter automatically:
1. Wraps reasoning items in `<think>` tags when sending to vLLM
2. Extracts reasoning from `<think>` tags in vLLM responses
3. Converts between formats transparently

**Why this matters**: Some models (like Qwen3) generate reasoning tokens during inference. The adapter ensures these tokens are properly captured in the Responses format expected by NeMo Gym's training data format.

:::{important}
Always set `uses_reasoning_parser: true` in your vLLM config (this is the default). Do NOT use vLLM's built-in reasoning parser arguments.
:::

:::

:::{dropdown} Understanding token ID information for training

When `return_token_id_information: true`, the vLLM adapter adds these fields to responses:

```python
{
  "prompt_token_ids": [128000, 791, 1217, ...],      # Tokens in the prompt
  "generation_token_ids": [3, 791, 574, ...],        # Tokens generated by model
  "generation_log_probs": [-0.01, -0.23, -0.15, ...]  # Log probability per token
}
```

**Use cases**:
- **PPO training**: Requires log probabilities to compute policy gradients
- **DPO training**: Uses token-level information for direct preference optimization
- **Analysis**: Debug model behavior by examining token-level decisions

**How it works**: The adapter calls vLLM's `/tokenize` endpoint to get prompt token IDs and extracts generation token IDs from vLLM's logprobs output.

:::{note}
Token ID information adds minimal overhead but increases response payload size. Only enable for training, not for production inference-only workloads.
:::

:::

:::{dropdown} Developer role message handling

Some models don't support the `developer` role used in OpenAI's API. Set `replace_developer_role_with_system: true` to automatically convert:

```yaml
policy_model:
  responses_api_models:
    vllm_model:
      replace_developer_role_with_system: true
```

**Before conversion**:
```python
{"role": "developer", "content": "You are a helpful assistant."}
```

**After conversion**:
```python
{"role": "system", "content": "You are a helpful assistant."}
```

This is model-specific - most open models use `system` role and don't recognize `developer`.
:::

:::{dropdown} Session-based routing for load balancing

When using multiple vLLM servers, NeMo Gym routes requests based on session IDs:

```python
# First request from client A → routes to vLLM server 1
# Second request from client A → routes to vLLM server 1 (same session)
# First request from client B → routes to vLLM server 2 (round-robin)
# Second request from client B → routes to vLLM server 2 (same session)
```

**Why session-based?** Ensures conversation context remains on the same server, important for:
- KV cache efficiency (vLLM can reuse cached tokens)
- Consistent behavior within a single agent interaction
- Proper handling of multi-turn conversations

**Load distribution**: New sessions are assigned round-robin across servers, providing good load distribution for high-concurrency workloads with many independent rollouts.
:::

---

## Troubleshooting

### Connection Issues

**Symptom**: `Connection refused` or `Cannot connect to vLLM server`

**Check these**:

1. **vLLM server is running**:
   ```bash
   curl http://localhost:10240/health
   ```
   Should return `{"status": "ok"}` or similar.

2. **Correct URL in configuration**:
   ```yaml
   # Correct - includes /v1 path
   policy_base_url: http://localhost:10240/v1
   
   # Incorrect - missing /v1
   policy_base_url: http://localhost:10240
   ```

3. **Network connectivity**:
   ```bash
   # Test basic connectivity
   curl http://your-vllm-server:8000/v1/models
   ```

4. **Firewall rules**: Ensure the port is accessible from the machine running NeMo Gym.

### Model Not Found

**Symptom**: `Model not found` or `Invalid model identifier`

**Solution**:

1. **Verify model loaded in vLLM**:
   ```bash
   curl http://localhost:10240/v1/models
   ```
   Check that your model name appears in the response.

2. **Match model identifier exactly**:
   ```yaml
   # If vLLM shows: meta-llama/Llama-3.1-8B-Instruct
   # Use exactly that:
   policy_model_name: meta-llama/Llama-3.1-8B-Instruct
   ```

3. **Check vLLM logs**: Look for model loading errors or warnings.

### Context Length Exceeded

**Symptom**: Requests fail with "context length exceeded" errors

**How NeMo Gym handles this**: The vLLM adapter automatically catches context length errors and returns an empty response, allowing rollout collection to continue.

**To fix**:
- Use models with larger context windows
- Reduce conversation history length in your resource server
- Configure vLLM with `--max-model-len` to match your needs

### Reasoning Tokens Not Appearing

**Symptom**: Reasoning items missing from responses even though model generates them

**Check**:

1. **Reasoning parser enabled**:
   ```yaml
   uses_reasoning_parser: true  # Should be true (default)
   ```

2. **vLLM reasoning parser NOT used**:
   ```bash
   # Incorrect - do not use
   vllm serve model --reasoning-parser qwen3
   
   # Correct - no reasoning parser argument
   vllm serve model --tool-call-parser hermes
   ```

3. **Model generates reasoning**: Not all models produce reasoning tokens. Check model documentation.

### Slow Inference Performance

**Symptom**: Low throughput or high latency

**Optimize**:

1. **vLLM server parameters**:
   ```bash
   vllm serve model \
       --tensor-parallel-size 4 \          # Use multiple GPUs
       --gpu-memory-utilization 0.95 \     # Maximize GPU usage
       --max-num-seqs 256 \                # Increase batch size
       --enable-prefix-caching              # Cache common prefixes
   ```

2. **Load balancing**:
   ```yaml
   base_url:
     - http://vllm-1:8000/v1
     - http://vllm-2:8000/v1
     - http://vllm-3:8000/v1  # More servers = higher throughput
   ```

3. **NeMo Gym concurrency**:
   ```bash
   ng_collect_rollouts +concurrency=100  # Increase concurrent requests
   ```

4. **Monitor vLLM metrics**: Check GPU utilization, batch sizes, and queue lengths.

---

## API Reference

### Supported Endpoints

The vLLM adapter exposes these endpoints:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Endpoint
  - Description
* - `/v1/responses`
  - Main inference endpoint using Responses API format
* - `/v1/chat/completions`
  - Direct Chat Completions API access (for debugging)
* - `/health`
  - Health check endpoint
```

### Example Request

```python
from nemo_gym.server_utils import ServerClient

server_client = ServerClient.load_from_global_config()

response = await server_client.post(
    server_name="policy_model",
    url_path="/v1/responses",
    json={
        "input": [{"role": "user", "content": "What's the weather in San Francisco?"}],
        "tools": [{
            "type": "function",
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }]
    }
)
```

### Tool Calling Support

vLLM adapter supports function calling through the `tools` parameter. Configure vLLM with appropriate tool call parser:

```bash
# For Hermes-format models (Qwen, Llama-3.1-70B-Instruct, etc.)
vllm serve model --enable-auto-tool-choice --tool-call-parser hermes

# For Mistral-format models
vllm serve model --enable-auto-tool-choice --tool-call-parser mistral
```

Refer to [vLLM tool calling documentation](https://docs.vllm.ai/) for supported models and parsers.

---

## Next Steps

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Collect Your First Rollouts
:link: ../get-started/collecting-rollouts
:link-type: doc

Generate training data at scale using your vLLM deployment.
:::

:::{grid-item-card} {octicon}`stack;1.5em;sd-mr-1` Configure Resource Servers
:link: ../how-to-faq
:link-type: doc

Add tools and verification logic for your specific use case.
:::

:::{grid-item-card} {octicon}`people;1.5em;sd-mr-1` Use Separate Judge Models
:link: ../tutorials/separate-policy-and-judge-models
:link-type: doc

Deploy different models for policy and verification.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Understand Model Abstractions
:link: ../about/concepts/core-abstractions
:link-type: doc

Learn how models fit into NeMo Gym's architecture.
:::

::::

:::{seealso}
**Related Documentation**:
- [Models Overview](index.md) - Compare all model serving methods
- [Configuration System](../about/concepts/configuration-system.md) - Deep dive into Hydra configuration
- [Offline Training with Rollouts](../tutorials/offline-training-w-rollouts.md) - Use rollouts with RL frameworks
:::