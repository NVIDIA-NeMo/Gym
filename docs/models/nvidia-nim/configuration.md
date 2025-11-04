(models-nvidia-nim-configuration)=

# Configuration Reference

Complete reference for all NVIDIA NIM configuration options in NeMo Gym.

---

## Configuration File Structure

NVIDIA NIM uses OpenAI-compatible endpoints, so you configure it using the OpenAI model adapter with your NIM endpoint:

```yaml
# responses_api_models/openai_model/configs/openai_model.yaml
policy_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      base_url: ${policy_base_url}
      api_key: ${policy_api_key}
      model: ${policy_model_name}
```

Configuration values resolve through **three layers** with increasing precedence.

::::{tab-set}

:::{tab-item} Layer 1: env.yaml

**Base values and secrets** (git-ignored)

```yaml
# env.yaml
policy_base_url: https://your-nim-endpoint.nvidia.com/v1
policy_api_key: your-nvidia-api-key
policy_model_name: meta/llama-3.1-8b-instruct
```

**When to use**:
- API keys and authentication credentials
- Environment-specific endpoints (dev/staging/prod)
- Personal or deployment-specific settings

:::

:::{tab-item} Layer 2: Config YAML
:selected:

**Structure with variable substitution** (version controlled)

```yaml
# responses_api_models/openai_model/configs/openai_model.yaml
policy_model:
  responses_api_models:
    openai_model:
      base_url: ${policy_base_url}      # ← substitutes from Layer 1
      api_key: ${policy_api_key}
      model: ${policy_model_name}
```

**Note**: NVIDIA NIM uses OpenAI-compatible endpoints, so you use the `openai_model` adapter.

:::

:::{tab-item} Layer 3: CLI Overrides

**Runtime overrides** (highest precedence)

```bash
# Override at runtime without changing files
ng_run "+config_paths=[${config_paths}]" \
    +policy_model_name=meta/llama-3.1-70b-instruct
```

**When to use**:
- Quick experiments with different NIM models
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

All available parameters for the NVIDIA NIM integration:

```{list-table}
:header-rows: 1
:widths: 20 15 65

* - Parameter
  - Type
  - Description
* - `base_url`
  - `str`
  - NVIDIA NIM endpoint URL including `/v1` path (e.g., `https://nim.example.com/v1`)
* - `api_key`
  - `str`
  - API key from your NIM deployment for authentication
* - `model`
  - `str`
  - Model identifier available in your NIM instance (e.g., `meta/llama-3.1-8b-instruct`)
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
  - NVIDIA NIM endpoint URL including `/v1` path
* - `policy_api_key`
  - API key for NIM authentication
* - `policy_model_name`
  - Model identifier deployed in NIM
```

---

## Available Models

NVIDIA NIM supports optimized versions of popular open models. Model availability depends on your NIM deployment.

**Common model families available in NIM**:

```{list-table}
:header-rows: 1
:widths: 40 60

* - Model Family
  - Example Identifiers
* - **Meta Llama**
  - `meta/llama-3.1-8b-instruct`, `meta/llama-3.1-70b-instruct`
* - **Mistral**
  - `mistralai/mixtral-8x7b-instruct-v0.1`, `mistralai/mistral-7b-instruct`
* - **NVIDIA Models**
  - Check NIM catalog for NVIDIA-optimized models
```

:::{dropdown} Check available models in your NIM instance

Verify which models are deployed and available:

```bash
curl -X GET "https://your-nim-endpoint.nvidia.com/v1/models" \
  -H "Authorization: Bearer your-api-key"
```

Or using Python:

```python
import openai

client = openai.OpenAI(
    api_key="your-api-key",
    base_url="https://your-nim-endpoint.nvidia.com/v1"
)

models = client.models.list()
for model in models.data:
    print(f"Available: {model.id}")
```

:::

:::{seealso}
**Full model catalog**: See [NVIDIA NIM documentation](https://developer.nvidia.com/nim) for complete model listings and optimization details.
:::

---

## API Endpoints

The NVIDIA NIM integration exposes standard OpenAI-compatible endpoints:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Endpoint
  - Description
* - `/v1/chat/completions`
  - Main inference endpoint using Chat Completions API format
* - `/v1/completions`
  - Legacy completions endpoint (if supported by NIM)
* - `/v1/models`
  - List available models in NIM deployment
* - `/health`
  - Health check endpoint
```

:::{dropdown} Example request using ServerClient

```python
from nemo_gym.server_utils import ServerClient

server_client = ServerClient.load_from_global_config()

response = await server_client.post(
    server_name="policy_model",
    url_path="/v1/chat/completions",
    json={
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [
            {"role": "user", "content": "What's the weather in San Francisco?"}
        ],
        "max_tokens": 100
    }
)
```

:::

