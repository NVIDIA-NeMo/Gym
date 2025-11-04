(models-vllm-configuration)=

# Configuration Reference

Complete reference for all vLLM adapter configuration options in NeMo Gym.

---

## Configuration File Structure

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

---

## Configuration Parameters

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
  - vLLM server endpoint(s). Single URL: `http://localhost:8000/v1`. List for load balancing: `["http://server1:8000/v1", "http://server2:8000/v1"]`. See [Optimization](optimization.md#load-balancing) for details.
* - `api_key`
  - `str`
  - Required
  - Authentication key. Use `EMPTY` if vLLM server has no authentication configured.
* - `model`
  - `str`
  - Required
  - Model identifier matching what's loaded in vLLM (e.g., `meta-llama/Llama-3.1-8B-Instruct`). Must match exactly.
* - `return_token_id_information`
  - `bool`
  - `false`
  - Include token IDs and log probabilities in responses for RL training. See training mode example below.
* - `uses_reasoning_parser`
  - `bool`
  - `true`
  - Enable automatic parsing of reasoning tokens using `<think>` tags. Converts between Responses API reasoning format and Chat Completions content. Keep enabled unless you have a specific reason to disable.
* - `replace_developer_role_with_system`
  - `bool`
  - `false`
  - Convert `developer` role messages to `system` role for models that don't support the developer role. Most open models need this set to `false`.
```

---

## Environment Variables

Define these in `env.yaml` at your repository root:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Variable
  - Description
* - `policy_base_url`
  - vLLM server endpoint URL including `/v1` path (e.g., `http://localhost:8000/v1`)
* - `policy_api_key`
  - API key for vLLM server authentication (use `EMPTY` if none)
* - `policy_model_name`
  - Model identifier loaded in vLLM server (must match exactly)
```

---

## Common Configuration Patterns

:::{dropdown} Training mode (with token IDs and log probabilities)

Enable token-level information required by RL training frameworks:

```yaml
# env.yaml
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: Qwen/Qwen3-30B-A3B

# Use vllm_model_for_training.yaml config
```

```yaml
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

**Use case**: When collecting rollouts for PPO, DPO, or other RL algorithms that require token-level information.

:::

:::{dropdown} Load balancing across multiple vLLM servers

Distribute requests for higher throughput:

```yaml
# env.yaml
policy_base_url: http://vllm-1:8000/v1,http://vllm-2:8000/v1,http://vllm-3:8000/v1
policy_api_key: EMPTY
policy_model_name: meta-llama/Llama-3.1-8B-Instruct
```

Or use YAML list syntax in a custom config file:

```yaml
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

See [Optimization Guide](optimization.md#load-balancing) for how session-based routing works.

:::

:::{dropdown} Judge model configuration

Configure a separate model for LLM-as-a-judge verification:

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

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
responses_api_models/vllm_model/configs/vllm_judge_model.yaml,\
resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml"

ng_run "+config_paths=[${config_paths}]"
```

See [Separate Policy and Judge Models](../../tutorials/separate-policy-and-judge-models.md) for advanced patterns.

:::

:::{dropdown} Developer role conversion

For models that don't support the `developer` role:

```yaml
policy_model:
  responses_api_models:
    vllm_model:
      replace_developer_role_with_system: true
```

This converts `{"role": "developer", "content": "..."}` to `{"role": "system", "content": "..."}` automatically.

:::

---

## API Endpoints

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

---

## Tool Calling Support

vLLM adapter supports function calling through the `tools` parameter. Configure vLLM with the appropriate tool call parser:

```bash
# For Hermes-format models (Qwen, Llama-3.1-70B-Instruct, etc.)
vllm serve model --enable-auto-tool-choice --tool-call-parser hermes

# For Mistral-format models
vllm serve model --enable-auto-tool-choice --tool-call-parser mistral
```

Refer to [vLLM tool calling documentation](https://docs.vllm.ai/) for supported models and parsers.

---

## Next Steps

- **[Optimize for production](optimization.md)** - Load balancing and performance tuning
- **[Troubleshoot issues](troubleshooting.md)** - Common problems and solutions
- **[Collect rollouts](../../get-started/collecting-rollouts.md)** - Generate training data

