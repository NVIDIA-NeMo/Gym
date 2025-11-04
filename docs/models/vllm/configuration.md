(models-vllm-configuration)=

# Configuration Reference

Complete reference for all vLLM adapter configuration options in NeMo Gym.

---

## Configuration File Structure

The vLLM adapter uses a standard [Hydra configuration file](https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/) with environment variable substitution:

:::{dropdown} How configuration layers work

Configuration values resolve through **three layers** with increasing precedence:

**Layer 1: `env.yaml` (base values)**
```yaml
# env.yaml - git-ignored, contains secrets
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: Qwen/Qwen3-30B-A3B
```

**Layer 2: Config YAML (structure with variable substitution)**
```yaml
# responses_api_models/vllm_model/configs/vllm_model.yaml
policy_model:
  responses_api_models:
    vllm_model:
      base_url: ${policy_base_url}      # ← substitutes from env.yaml
      api_key: ${policy_api_key}
      model: ${policy_model_name}
      return_token_id_information: false
```

**Layer 3: Command-line overrides (highest precedence)**
```bash
# Override specific values at runtime
ng_run "+config_paths=[${config_paths}]" \
    +policy_model_name=meta-llama/Llama-3.1-8B-Instruct \
    +policy_model.responses_api_models.vllm_model.return_token_id_information=true
```

**Common use cases**:
- **env.yaml**: Secrets, environment-specific URLs (dev/staging/prod)
- **Config YAML**: Structure, defaults, relationships between components
- **CLI overrides**: Quick experiments, CI/CD deployments, one-off changes

See [Configuration System](../../about/concepts/configuration-system.md) for complete details on precedence and composition.

:::

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

All available parameters for the vLLM adapter:

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

Refer to [vLLM tool calling documentation](https://docs.vllm.ai/) for supported models and parsers.
