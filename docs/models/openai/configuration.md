(models-openai-configuration)=

# Configuration Reference

Complete reference for all OpenAI configuration options in NeMo Gym.

---

## Configuration File Structure

The OpenAI adapter uses a standard configuration file with environment variable substitution:

```yaml
# responses_api_models/openai_model/configs/openai_model.yaml
policy_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      openai_base_url: ${policy_base_url}
      openai_api_key: ${policy_api_key}
      openai_model: ${policy_model_name}
```

Configuration values resolve through **three layers** with increasing precedence.

::::{tab-set}

:::{tab-item} Layer 1: env.yaml

**Base values and secrets** (git-ignored)

```yaml
# env.yaml
policy_base_url: https://api.openai.com/v1
policy_api_key: sk-your-openai-api-key-here
policy_model_name: gpt-4-turbo
```

**When to use**:
- API keys and authentication credentials
- Environment-specific endpoints
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
      openai_base_url: ${policy_base_url}      # ← substitutes from Layer 1
      openai_api_key: ${policy_api_key}
      openai_model: ${policy_model_name}
```

**This is the OpenAI adapter configuration** - what parameters are available and their structure.

:::

:::{tab-item} Layer 3: CLI Overrides

**Runtime overrides** (highest precedence)

```bash
# Override at runtime without changing files
ng_run "+config_paths=[${config_paths}]" \
    +policy_model_name=gpt-3.5-turbo
```

**When to use**:
- Quick experiments with different OpenAI models
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

All available parameters for the OpenAI adapter:

```{list-table}
:header-rows: 1
:widths: 20 15 65

* - Parameter
  - Type
  - Description
* - `openai_base_url`
  - `str`
  - **Required**. OpenAI API endpoint URL (typically `https://api.openai.com/v1`)
* - `openai_api_key`
  - `str`
  - **Required**. Your OpenAI API key (starts with `sk-`)
* - `openai_model`
  - `str`
  - **Required**. OpenAI model identifier (e.g., `gpt-4-turbo`, `gpt-3.5-turbo`)
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
  - OpenAI API endpoint (typically `https://api.openai.com/v1`)
* - `policy_api_key`
  - Your OpenAI API key from platform.openai.com
* - `policy_model_name`
  - Model identifier to use for inference
```

---

## Available Models

OpenAI provides several model families with different capabilities and pricing.

### GPT-4 Family

High-intelligence models for complex reasoning and generation:

```{list-table}
:header-rows: 1
:widths: 30 25 45

* - Model
  - Context Length
  - Best For
* - `gpt-4-turbo`
  - 128K tokens
  - Latest GPT-4 with improved speed and lower cost
* - `gpt-4`
  - 8K tokens
  - Original GPT-4, highest capability
* - `gpt-4-32k`
  - 32K tokens
  - Extended context for long conversations
```

### GPT-3.5 Family

Fast and cost-effective models for most use cases:

```{list-table}
:header-rows: 1
:widths: 30 25 45

* - Model
  - Context Length
  - Best For
* - `gpt-3.5-turbo`
  - 16K tokens
  - Fast, cost-effective, great for testing
* - `gpt-3.5-turbo-16k`
  - 16K tokens
  - Longer context variant
```

:::{dropdown} Check model availability and pricing

Visit the [OpenAI Model Documentation](https://platform.openai.com/docs/models) for:
- Complete model list with capabilities
- Current pricing per token
- Model deprecation notices
- Recommended use cases

**Pricing considerations**:
- GPT-4 models: Higher cost, best quality
- GPT-3.5 models: 10-30x cheaper, excellent for most tasks
- Token pricing includes both input and output tokens

:::

:::{seealso}
**Full model catalog**: See [OpenAI Models](https://platform.openai.com/docs/models) for the latest models and detailed specifications.
:::

---

## API Endpoints

The OpenAI adapter exposes two OpenAI-compatible endpoints:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Endpoint
  - Description
* - `/v1/chat/completions`
  - Chat Completions API for conversational inference (messages format)
* - `/v1/responses`
  - OpenAI Responses API for structured multi-turn conversations with tool calling
```

:::{note}
The OpenAI adapter proxies requests to OpenAI's actual endpoints. It does **not** implement `/v1/completions` (legacy) or `/v1/models` (model listing) endpoints. Use `/v1/chat/completions` for most use cases.
:::

:::{dropdown} Example request using ServerClient

```python
from nemo_gym.server_utils import ServerClient

server_client = ServerClient.load_from_global_config()

response = await server_client.post(
    server_name="policy_model",
    url_path="/v1/chat/completions",
    json={
        "model": "gpt-4-turbo",
        "messages": [
            {"role": "user", "content": "What's the weather in San Francisco?"}
        ],
        "max_tokens": 100
    }
)
```

:::

