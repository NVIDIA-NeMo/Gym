(model-server-openai)=
# OpenAI Model Server

```{note}
This page is a stub. Content is being developed. See [GitHub Issue #194](https://github.com/NVIDIA-NeMo/Gym/issues/194) for details.
```

The OpenAI model server (`responses_api_models/openai_model/`) connects NeMo Gym to OpenAI's API, providing access to GPT models with native function calling support.

---

## When to Use OpenAI

Use OpenAI when you need:
- Quick prototyping and development
- Access to latest GPT models
- Native Responses API support
- Reliable function calling

## Configuration

Configure in `responses_api_models/openai_model/configs/openai_model.yaml`:

```yaml
policy_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      openai_base_url: ${policy_base_url}
      openai_api_key: ${policy_api_key}
      openai_model: ${policy_model_name}
```

Set credentials in `env.yaml`:

```yaml
policy_base_url: https://api.openai.com/v1
policy_api_key: sk-your-api-key
policy_model_name: gpt-4.1-2025-04-14
```

## Supported Models

Any OpenAI model with function calling support:

| Model | Function Calling | Recommended Use |
|-------|------------------|-----------------|
| `gpt-4.1-2025-04-14` | ✅ | Low-latency prototyping |
| `gpt-4o` | ✅ | General purpose |
| `gpt-4o-mini` | ✅ | Cost-effective testing |
| `gpt-4-turbo` | ✅ | Complex reasoning |

## API Key Setup

1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Ensure billing is configured with available credits
3. Store securely in `env.yaml` (gitignored)

## Rate Limits

<!-- TODO: Document rate limit handling -->

## Cost Optimization

<!-- TODO: Document cost optimization strategies -->

## Troubleshooting

### Authentication Errors

```text
Error code: 401 - Incorrect API key provided
```

Verify your API key in `env.yaml` matches your OpenAI account.

### Quota Errors

```text
Error code: 429 - You exceeded your current quota
```

Add credits at [platform.openai.com/account/billing](https://platform.openai.com/account/billing).
