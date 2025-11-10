(models-vllm-configuration)=

# Configuration Reference

Complete reference for all vLLM adapter configuration options in NeMo Gym.

---

## Configuration File Structure

The vLLM adapter uses a standard [Hydra configuration file](https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/) with environment variable substitution:

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

Configuration values resolve through **three layers** with increasing precedence.

::::{tab-set}

:::{tab-item} Layer 1: env.yaml

**Base values and secrets** (git-ignored)

```yaml
# env.yaml
policy_base_url: http://localhost:8000/v1
policy_api_key: EMPTY
policy_model_name: Qwen/Qwen3-30B-A3B
```

:::{note}
The default vLLM port is `8000`. Use the port where your vLLM server is running.
:::

**When to use**:
- Secrets and API keys
- Environment-specific values (dev/staging/prod URLs)
- Personal/local settings

:::

:::{tab-item} Layer 2: Config YAML
:selected:

**Structure with variable substitution** (version controlled)

```yaml
# responses_api_models/vllm_model/configs/vllm_model.yaml
policy_model:
  responses_api_models:
    vllm_model:
      base_url: ${policy_base_url}      # ← substitutes from Layer 1
      api_key: ${policy_api_key}
      model: ${policy_model_name}
      return_token_id_information: false
      uses_reasoning_parser: true
```

**This is the vLLM adapter configuration** - what parameters are available and their structure.

:::

:::{tab-item} Layer 3: CLI Overrides

**Runtime overrides** (highest precedence)

```bash
# Override at runtime without changing files
ng_run "+config_paths=[${config_paths}]" \
    +policy_model_name=meta-llama/Llama-3.1-8B-Instruct \
    +policy_model.responses_api_models.vllm_model.return_token_id_information=true
```

**When to use**:
- Quick experiments with different models
- CI/CD deployments with dynamic values
- One-off changes without editing files

**Syntax**: Use dotted path to nested values with `+` prefix

:::

::::

:::{seealso}
See [Configuration System](../../about/concepts/configuration-system.md) for complete details on precedence and composition.
:::

---

## Configuration Parameters

All available parameters for the vLLM adapter:

```{list-table}
:header-rows: 1
:widths: 20 15 15 50

* - Parameter
  - Type
  - Default
  - Description
* - `entrypoint`
  - `str`
  - `app.py`
  - Python module containing the vLLM model server. Standard NeMo Gym configuration (usually no need to change).
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
  - OpenAI-compatible Chat Completions API (for debugging or direct integration)
```

:::{dropdown} Example request using ServerClient

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

:::

---

## Tool Calling Support

vLLM adapter supports function calling through the `tools` parameter. Configure vLLM with the appropriate tool call parser:

```bash
# For Hermes-format models (Qwen, Llama-3.1-70B-Instruct, etc.)
vllm serve model --enable-auto-tool-choice --tool-call-parser hermes

# For Mistral-format models
vllm serve model --enable-auto-tool-choice --tool-call-parser mistral
```

Refer to [vLLM tool calling documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#tool-calling) for supported models and parsers.
