(model-server-azure-openai)=
# Azure OpenAI Model Server

Connect NeMo Gym to Azure-hosted OpenAI models for enterprise RL training workflows. The server wraps the `AsyncAzureOpenAI` client and exposes OpenAI-compatible endpoints for generating rollouts.

**Goal**: Connect NeMo Gym to Azure OpenAI for enterprise deployments.

**Prerequisites**: Azure subscription with OpenAI Service access, deployed model

**Source**: `responses_api_models/azure_openai_model/`

:::{tip}
**Use Azure OpenAI when you need**: Enterprise compliance, data residency controls, or private network deployment.

**Use standard {doc}`openai` when**: You don't need Azure-specific features and want simpler setup.
:::

---

## Prerequisites

Before configuring the Azure OpenAI model server:

1. **Azure subscription** with OpenAI Service access approved
2. **Azure OpenAI resource** created in Azure portal
3. **Model deployment** configured — see [Azure: Create a deployment](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource)
4. **API credentials** from Azure portal:
   - Endpoint URL (e.g., `https://your-resource.openai.azure.com`)
   - API key
   - Deployment name (not model name)

---

## Configuration

### Step 1: Create Environment Variables

Create or update `env.yaml` in your NeMo Gym project directory (where you run `ng_run`):

```yaml
policy_base_url: https://your-resource.openai.azure.com
policy_api_key: your-azure-api-key
policy_model_name: your-deployment-name
```

```{important}
Use your **deployment name** (not the model name) for `policy_model_name`. Find this in Azure Portal → your resource → Model deployments.
```

```{tip}
For production deployments, use environment variables instead of storing secrets in files:
`export POLICY_API_KEY=your-azure-api-key`
```

### Step 2: Server Configuration

The server configuration is in `responses_api_models/azure_openai_model/configs/azure_openai_model.yaml`:

```yaml
policy_model:
  responses_api_models:
    azure_openai_model:
      entrypoint: app.py
      openai_base_url: ${policy_base_url}
      openai_api_key: ${policy_api_key}
      openai_model: ${policy_model_name}
      default_query:
        api-version: ???  # Required: Set via command line
      num_concurrent_requests: 8
```

### Configuration Reference

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `openai_base_url` | `str` | Yes | Azure OpenAI endpoint URL |
| `openai_api_key` | `str` | Yes | Azure API key |
| `openai_model` | `str` | Yes | Deployment name in Azure |
| `default_query.api-version` | `str` | Yes | Azure API version (e.g., `2024-10-21`) |
| `num_concurrent_requests` | `int` | No | Max concurrent requests (default: `8`) |
| `host` | `str` | No | Server host (auto-assigned if not set) |
| `port` | `int` | No | Server port (auto-assigned if not set) |

**Source**: `responses_api_models/azure_openai_model/app.py:36-41`

---

## Usage

### Running the Server

Start the server with an explicit API version:

```bash
config_paths="responses_api_models/azure_openai_model/configs/azure_openai_model.yaml, \
resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml"

ng_run "+config_paths=[${config_paths}]" \
    +policy_model.responses_api_models.azure_openai_model.default_query.api-version=2024-10-21
```

```{note}
The `api-version` must be set at runtime. Check [Azure OpenAI API versions](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#api-specs) for the latest supported versions.
```

### Collecting Rollouts

After starting the server, collect rollouts from your environment:

```bash
ng_collect_rollouts \
  +agent_name=equivalence_llm_judge_simple_agent \
  +input_jsonl_fpath=resources_servers/equivalence_llm_judge/data/example.jsonl \
  +output_jsonl_fpath=results/example_rollouts.jsonl \
  +limit=5
```

### Running Tests

Validate your configuration with the built-in test suite:

```bash
ng_test +entrypoint=responses_api_models/azure_openai_model
```

---

## API Endpoints

The server exposes two OpenAI-compatible endpoints:

| Endpoint | Purpose | Request Format |
|----------|---------|----------------|
| `POST /v1/responses` | Responses API for agentic workflows | `{"input": "your prompt"}` |
| `POST /v1/chat/completions` | Standard Chat Completions API | `{"messages": [{"role": "user", "content": "..."}]}` |

**Source**: `nemo_gym/base_responses_api_model.py:42-44`

### Concurrency

Requests are rate-limited using an asyncio semaphore. The `num_concurrent_requests` parameter (default: `8`) controls maximum parallel requests to Azure. When the limit is reached, additional requests queue until a slot becomes available.

```{tip}
Set `num_concurrent_requests` based on your Azure deployment's TPM (tokens per minute) quota. Start with `8` and increase if you have higher quotas.
```

**Source**: `responses_api_models/azure_openai_model/app.py:54`

---

## Troubleshooting

### Authentication Errors

```text
Error: Invalid API key or endpoint
```

**Cause**: Incorrect credentials in `env.yaml`.

**Fix**:
1. Verify `policy_base_url` matches your Azure portal endpoint exactly
2. Regenerate API key in Azure portal if needed
3. Check for trailing slashes or typos in the URL

### API Version Errors

```text
Error: API version not supported
```

**Cause**: Invalid or unsupported `api-version` parameter.

**Fix**: Use a supported version from your Azure deployment. Check Azure portal → your resource → Overview for supported versions.

### Deployment Not Found

```text
Error: The deployment 'gpt-4' does not exist
```

**Cause**: Using model name instead of deployment name.

**Fix**: Use the **deployment name** you created in Azure portal, not the base model name (e.g., `my-gpt4-deployment` not `gpt-4`).

### Connection Timeout

```text
Error: Connection timeout
```

**Cause**: Network connectivity or Azure service issues.

**Fix**:
1. Verify network access to Azure endpoint
2. Check Azure service health status
3. Increase timeout if needed in your network configuration

---

## Related Topics

- [OpenAI Model Server](openai.md) — Direct OpenAI API integration
- [vLLM Model Server](vllm.md) — Self-hosted model serving
- [Model Server Overview](index.md) — Compare model server options

---

**Source**: `responses_api_models/azure_openai_model/`
