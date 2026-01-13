(model-server-azure-openai)=
# Azure OpenAI Model Server

```{note}
This page is a stub. Content is being developed. See [GitHub Issue #194](https://github.com/NVIDIA-NeMo/Gym/issues/194) for details.
```

The Azure OpenAI model server (`responses_api_models/azure_openai_model/`) connects NeMo Gym to Azure-hosted OpenAI models for enterprise deployments.

---

## When to Use Azure OpenAI

Use Azure OpenAI when you need:
- Enterprise compliance requirements
- Data residency controls
- Azure subscription integration
- Private network deployment

## Configuration

Configure in `responses_api_models/azure_openai_model/configs/azure_openai_model.yaml`:

```yaml
policy_model:
  responses_api_models:
    azure_openai_model:
      entrypoint: app.py
      openai_base_url: ${policy_base_url}
      openai_api_key: ${policy_api_key}
      openai_model: ${policy_model_name}
      default_query:
        api-version: "2024-10-21"
      num_concurrent_requests: 8
```

Set credentials in `env.yaml`:

```yaml
policy_base_url: https://your-resource.openai.azure.com
policy_api_key: your-azure-api-key
policy_model_name: gpt-4-deployment
```

## Key Configuration Options

| Parameter | Description |
|-----------|-------------|
| `openai_base_url` | Azure OpenAI endpoint URL |
| `openai_api_key` | Azure API key |
| `openai_model` | Deployed model name in Azure |
| `default_query.api-version` | Azure API version (e.g., `2024-10-21`) |
| `num_concurrent_requests` | Max concurrent requests (default: 8) |

## Azure Setup Requirements

1. Azure subscription with OpenAI access
2. Azure OpenAI resource created
3. Model deployed to your resource
4. API key from Azure portal

## Running the Server

Set the API version and start the server:

```bash
config_paths="responses_api_models/azure_openai_model/configs/azure_openai_model.yaml, \
resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml"

ng_run "+config_paths=[${config_paths}]" \
    +policy_model.responses_api_models.azure_openai_model.default_query.api-version=2024-10-21
```

## Collecting Rollouts

```bash
ng_collect_rollouts \
  +agent_name=equivalence_llm_judge_simple_agent \
  +input_jsonl_fpath=resources_servers/equivalence_llm_judge/data/example.jsonl \
  +output_jsonl_fpath=results/example_rollouts.jsonl \
  +limit=5
```

## API Endpoints

The Azure OpenAI model server exposes two endpoints:

- `/v1/responses` - Responses API for agentic workflows
- `/v1/chat/completions` - Standard Chat Completions API

## Troubleshooting

### Authentication Errors

```text
Error: Invalid API key or endpoint
```

Verify your `policy_api_key` and `policy_base_url` in `env.yaml` match your Azure portal credentials.

### API Version Errors

```text
Error: API version not supported
```

Check supported API versions in Azure portal. Common versions: `2024-10-21`, `2024-02-15-preview`.

### Deployment Not Found

```text
Error: Deployment not found
```

Ensure `policy_model_name` matches the deployment name (not the model name) in your Azure OpenAI resource.
