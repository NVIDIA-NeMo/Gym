(models-openrouter-configuration)=

# Configuration Reference

Complete reference for all OpenRouter configuration options in NeMo Gym.

---

## Configuration File Structure

OpenRouter uses OpenAI-compatible endpoints, so you configure it using the OpenAI model adapter with OpenRouter's base URL:

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
policy_base_url: https://openrouter.ai/api/v1
policy_api_key: sk-or-your-openrouter-api-key
policy_model_name: openai/gpt-4-turbo
```

**When to use**:
- API keys and authentication credentials
- Environment-specific settings
- Personal or deployment-specific configuration

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

**Note**: OpenRouter uses OpenAI-compatible endpoints, so you use the `openai_model` adapter.

:::

:::{tab-item} Layer 3: CLI Overrides

**Runtime overrides** (highest precedence)

```bash
# Override at runtime without changing files
ng_run "+config_paths=[${config_paths}]" \
    +policy_model_name=anthropic/claude-3-sonnet
```

**When to use**:
- Quick experiments with different OpenRouter models
- CI/CD deployments with dynamic values
- Comparing models from different providers

**Syntax**: Use dotted path to nested values with `+` prefix

:::

::::

:::{seealso}
See [Configuration System](../../about/concepts/configuration-system.md) for complete details on precedence and composition.
:::

---

## Configuration Parameters

All available parameters for OpenRouter integration:

```{list-table}
:header-rows: 1
:widths: 20 15 65

* - Parameter
  - Type
  - Description
* - `openai_base_url`
  - `str`
  - OpenRouter API endpoint (always `https://openrouter.ai/api/v1`)
* - `openai_api_key`
  - `str`
  - Your OpenRouter API key (starts with `sk-or-`)
* - `openai_model`
  - `str`
  - Model identifier in `provider/model-name` format
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
  - OpenRouter API endpoint (always `https://openrouter.ai/api/v1`)
* - `policy_api_key`
  - Your OpenRouter API key from openrouter.ai dashboard
* - `policy_model_name`
  - Model identifier in `provider/model-name` format
```

---

## Available Models

OpenRouter provides access to models from multiple providers. Model identifiers use the format `provider/model-name`.

### OpenAI Models

```{list-table}
:header-rows: 1
:widths: 50 50

* - Model
  - Identifier
* - GPT-4 Turbo
  - `openai/gpt-4-turbo`
* - GPT-4
  - `openai/gpt-4`
* - GPT-3.5 Turbo
  - `openai/gpt-3.5-turbo`
```

### Anthropic Models

```{list-table}
:header-rows: 1
:widths: 50 50

* - Model
  - Identifier
* - Claude 3 Opus
  - `anthropic/claude-3-opus`
* - Claude 3 Sonnet
  - `anthropic/claude-3-sonnet`
* - Claude 3 Haiku
  - `anthropic/claude-3-haiku`
```

### Google Models

```{list-table}
:header-rows: 1
:widths: 50 50

* - Model
  - Identifier
* - Gemini Pro
  - `google/gemini-pro`
* - Gemini Pro Vision
  - `google/gemini-pro-vision`
* - PaLM 2
  - `google/palm-2`
```

### Open Source Models

```{list-table}
:header-rows: 1
:widths: 50 50

* - Model
  - Identifier
* - Llama 3 70B
  - `meta-llama/llama-3-70b-instruct`
* - Mixtral 8x7B
  - `mistralai/mixtral-8x7b-instruct`
* - Qwen 2.5 72B
  - `qwen/qwen-2.5-72b-instruct`
```

:::{dropdown} View full model catalog

Visit [OpenRouter Models](https://openrouter.ai/models) for:
- Complete model list from all providers
- Real-time pricing per token
- Model capabilities and context lengths
- Provider availability status
- Performance benchmarks

**Model catalog features**:
- Search and filter by provider, price, context length
- Compare costs across providers
- Check current availability
- View model descriptions and use cases

:::

---

## Cost Optimization

OpenRouter enables cost optimization across multiple providers.

### Compare Provider Costs

OpenRouter displays transparent pricing for each model:

**Cost comparison strategies**:
- View per-request costs in OpenRouter dashboard
- Compare equivalent models across providers
- Switch to cheaper providers for testing/development
- Use expensive models only when quality justifies cost

**Example price comparison** (as of documentation date):
- GPT-3.5 Turbo: $0.50 / 1M input tokens
- Claude 3 Haiku: $0.25 / 1M input tokens
- Mixtral 8x7B: $0.24 / 1M input tokens

:::{tip}
**Pricing changes**: Check [OpenRouter models page](https://openrouter.ai/models) for current pricing - providers frequently update rates.
:::

### Model Selection Strategy

Choose models based on your use case and budget:

**Development/Testing**:
- Use cost-effective models (Claude Haiku, GPT-3.5, open models)
- Higher concurrency with cheaper models
- Iterate quickly without cost concern

**Production/Quality**:
- Use higher-quality models where needed (GPT-4, Claude Opus)
- Consider quality vs cost tradeoff
- A/B test different models to find optimal balance

**Fallback Strategy**:
- Primary: High-quality model (GPT-4, Claude Opus)
- Fallback: Cost-effective alternative (GPT-3.5, Claude Sonnet)
- OpenRouter can route requests based on availability

---

## Provider Fallback

OpenRouter provides automatic fallback when providers are unavailable.

### How Fallback Works

OpenRouter can automatically retry with alternative providers:

**Benefits**:
- Increased reliability when a provider has outages
- Automatic failover without code changes
- Continue rollout collection even if one provider fails

**Configuration**:
- Fallback is handled by OpenRouter automatically
- No NeMo Gym configuration required
- Monitor OpenRouter dashboard for fallback events

:::{seealso}
**Fallback configuration**: See [OpenRouter Documentation](https://openrouter.ai/docs) for advanced fallback and routing options.
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
The OpenAI adapter proxies requests to OpenRouter's OpenAI-compatible endpoints. It does **not** implement `/v1/completions` (legacy) or `/v1/models` (model listing) endpoints. Use `/v1/chat/completions` for most use cases.
:::

:::{dropdown} Example request using ServerClient

```python
from asyncio import run
from nemo_gym.server_utils import ServerClient

server_client = ServerClient.load_from_global_config()


async def main():
    response = await server_client.post(
        server_name="policy_model",
        url_path="/v1/chat/completions",
        json={
            "model": "openai/gpt-4-turbo",
            "messages": [
                {"role": "user", "content": "What's the weather in San Francisco?"}
            ],
            "max_tokens": 100
        }
    )
    print(await response.json())


if __name__ == "__main__":
    run(main())
```

```{note}
**Model parameter behavior**: The `model` parameter in the request JSON is **ignored**. The model specified in your configuration (`openai_model` in the YAML) always takes precedence. This ensures consistent model usage across all requests.
```

:::

---

## OpenRouter Dashboard

Monitor usage and costs through the OpenRouter dashboard.

**Dashboard features**:
- Real-time request logs
- Token usage tracking
- Cost breakdown by model and provider
- Spending history and trends
- API key management

**Access**: Visit [openrouter.ai/dashboard](https://openrouter.ai/dashboard) after signing in.

