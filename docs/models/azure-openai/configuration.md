(models-azure-openai-configuration)=

# Configuration Reference

Complete reference for all Azure OpenAI configuration options in NeMo Gym.

---

## Configuration File Structure

The Azure OpenAI adapter uses a standard configuration file with environment variable substitution:

```yaml
# responses_api_models/azure_openai_model/configs/azure_openai_model.yaml
policy_model:
  responses_api_models:
    azure_openai_model:
      entrypoint: app.py
      openai_base_url: ${policy_base_url}
      openai_api_key: ${policy_api_key}
      openai_model: ${policy_model_name}
      default_query:
        api-version: 2024-10-21
      num_concurrent_requests: 8
```

Configuration values resolve through **three layers** with increasing precedence.

::::{tab-set}

:::{tab-item} Layer 1: env.yaml

**Base values and secrets** (git-ignored)

```yaml
# env.yaml
policy_base_url: https://your-resource.openai.azure.com
policy_api_key: your-azure-api-key
policy_model_name: my-gpt-4-deployment
```

**When to use**:
- API keys and authentication credentials
- Environment-specific endpoints (dev/staging/prod resources)
- Personal or deployment-specific settings

:::

:::{tab-item} Layer 2: Config YAML
:selected:

**Structure with variable substitution** (version controlled)

```yaml
# responses_api_models/azure_openai_model/configs/azure_openai_model.yaml
policy_model:
  responses_api_models:
    azure_openai_model:
      openai_base_url: ${policy_base_url}      # ← substitutes from Layer 1
      openai_api_key: ${policy_api_key}
      openai_model: ${policy_model_name}
      default_query:
        api-version: 2024-10-21          # Azure-specific requirement (example - check Azure docs)
      num_concurrent_requests: 8
```

**Azure-specific**: Requires `api-version` parameter in `default_query`.

:::

:::{tab-item} Layer 3: CLI Overrides

**Runtime overrides** (highest precedence)

```bash
# Override at runtime without changing files
ng_run "+config_paths=[${config_paths}]" \
    +policy_model_name=my-gpt-35-deployment \
    +policy_model.responses_api_models.azure_openai_model.default_query.api-version=2024-11-06
```

**When to use**:
- Quick experiments with different Azure deployments
- CI/CD with dynamic values
- Testing different API versions

**Syntax**: Use dotted path to nested values with `+` prefix

:::

::::

:::{seealso}
See [Configuration System](../../about/concepts/configuration-system.md) for complete details on precedence and composition.
:::

---

## Configuration Parameters

All available parameters for the Azure OpenAI adapter:

```{list-table}
:header-rows: 1
:widths: 20 15 65

* - Parameter
  - Type
  - Description
* - `openai_base_url`
  - `str`
  - Azure OpenAI endpoint URL (format: `https://RESOURCE.openai.azure.com`)
* - `openai_api_key`
  - `str`
  - Azure OpenAI API key from Azure portal
* - `openai_model`
  - `str`
  - Your deployment name (not base model name like `gpt-4`)
* - `default_query.api-version`
  - `str`
  - Azure API version (format: `YYYY-MM-DD`, e.g., `2024-10-21`)
* - `num_concurrent_requests`
  - `int`
  - Maximum number of concurrent requests to Azure API (default: 8, controls rate limiting)
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
  - Azure OpenAI endpoint URL (e.g., `https://your-resource.openai.azure.com`)
* - `policy_api_key`
  - API key from Azure portal (Keys and Endpoint section)
* - `policy_model_name`
  - Your custom deployment name from Azure portal
```

---

## Azure Deployments

Azure OpenAI uses custom deployment names instead of base model names.

### Deployment Names vs Model Names

```{list-table}
:header-rows: 1
:widths: 50 50

* - ❌ Incorrect (base model name)
  - ✅ Correct (deployment name)
* - `policy_model_name: gpt-4`
  - `policy_model_name: my-gpt-4-deployment`
* - `policy_model_name: gpt-35-turbo`
  - `policy_model_name: prod-gpt35-turbo`
* - `policy_model_name: gpt-4-turbo`
  - `policy_model_name: dev-gpt4-turbo`
```

:::{tip}
**Find your deployment name**: Navigate to your Azure OpenAI resource → Model deployments → Copy the deployment name (not the model name).
:::

### Available Models in Azure

Azure OpenAI supports various OpenAI models. Availability varies by region.

**GPT-4 Family**:
- `gpt-4` (8K context)
- `gpt-4-32k` (32K context)
- `gpt-4-turbo` (128K context)

**GPT-3.5 Family**:
- `gpt-35-turbo` (16K context)
- `gpt-35-turbo-16k` (16K context)

:::{dropdown} Check model availability by region

Visit [Azure OpenAI model availability](https://learn.microsoft.com/azure/ai-services/openai/concepts/models) for:
- Complete model list
- Regional availability matrix
- Model version differences
- Deprecation schedules

**Regional considerations**:
- Not all models available in all regions
- Some regions have waitlists for specific models
- Check your region's model availability before deploying

:::

---

## API Versions

Azure OpenAI requires specifying an API version. Use the latest stable version.

### Current API Versions

```{list-table}
:header-rows: 1
:widths: 30 70

* - Version
  - Notes
* - `2024-10-21`
  - Latest stable version (as of documentation date)
* - `2024-08-01-preview`
  - Preview features
* - `2024-06-01`
  - Previous stable version
```

:::{tip}
**Check current versions**: Visit [Azure OpenAI API versions](https://learn.microsoft.com/azure/ai-services/openai/reference) for the latest stable and preview versions.
:::

### Setting API Version

**Method 1: CLI Override (recommended)**:
```bash
ng_run "+config_paths=[${config_paths}]" \
    +policy_model.responses_api_models.azure_openai_model.default_query.api-version=2024-10-21
```

**Method 2: Configuration File**:
```yaml
# responses_api_models/azure_openai_model/configs/azure_openai_model.yaml
policy_model:
  responses_api_models:
    azure_openai_model:
      default_query:
        api-version: 2024-10-21  # Check Azure docs for current stable version
```

---

## Regional Deployment

Azure OpenAI availability varies by region. Consider these factors when choosing a region:

**Data residency requirements**:
- Deploy in regions matching your data governance requirements
- EU regions for GDPR compliance
- US regions for US data residency
- Asia-Pacific regions for APAC data residency

**Model availability**:
- GPT-4 models: Limited regions
- GPT-3.5 models: More widely available
- Check [regional availability](https://learn.microsoft.com/azure/ai-services/openai/concepts/models#model-summary-table-and-region-availability)

**Latency considerations**:
- Deploy near your NeMo Gym infrastructure
- Cross-region latency can impact throughput
- Test latency before large-scale deployment

---

## API Endpoints

The Azure OpenAI adapter exposes Azure-compatible endpoints:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Endpoint
  - Description
* - `/v1/chat/completions`
  - OpenAI-compatible chat completions endpoint
* - `/v1/responses`
  - NeMo Gym responses API endpoint
```

:::{dropdown} Example request using ServerClient

```python
from nemo_gym.server_utils import ServerClient

server_client = ServerClient.load_from_global_config()

# The model parameter is not needed - uses deployment from config
response = await server_client.post(
    server_name="policy_model",
    url_path="/v1/chat/completions",
    json={
        "messages": [
            {"role": "user", "content": "What's the weather in San Francisco?"}
        ],
        "max_tokens": 100
    }
)
```

:::

---

## Azure Portal Setup

Access your Azure OpenAI configuration through the Azure portal.

### Finding Your Configuration

1. **Navigate to Azure OpenAI resource**:
   - [Azure portal](https://portal.azure.com)
   - Search for your Azure OpenAI resource
   - Open resource overview

2. **Get endpoint and key**:
   - Click "Keys and Endpoint" in left navigation
   - Copy "Endpoint" URL
   - Copy Key 1 or Key 2

3. **View deployments**:
   - Click "Model deployments"
   - See list of your deployed models
   - Copy deployment names for `env.yaml`

4. **Monitor usage**:
   - Click "Metrics" for usage metrics
   - "Cost Management" for billing
   - "Logs" for request logs

:::{seealso}
**Full Azure setup guide**: See [Azure OpenAI Documentation](https://learn.microsoft.com/azure/ai-services/openai/) for detailed Azure portal instructions.
:::

