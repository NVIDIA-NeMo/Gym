(model-server-openai)=
# OpenAI Model Server

```{warning}
This article has not been reviewed by a developer SME. Content may change.
```

The OpenAI model server (`responses_api_models/openai_model/`) connects NeMo Gym to OpenAI's API, providing access to GPT models with native function calling support.

---

## When to Use OpenAI vs vLLM

| Factor | OpenAI | vLLM |
|--------|--------|------|
| **Setup time** | Minutes (API key only) | Hours (GPU + model download) |
| **Cost** | Pay per token | GPU infrastructure |
| **Training integration** | ❌ No token IDs | ✅ Full token tracking |
| **Data privacy** | Cloud processing | On-premise |
| **Latest models** | ✅ Immediate access | Depends on open weights |
| **Rate limits** | Yes (varies by tier) | No (self-hosted) |

**Use OpenAI when:**

- Quick prototyping and development
- Testing environment design before GPU investment
- Baseline comparisons with frontier models
- Budget allows pay-per-token pricing

**Use vLLM when:**

- Training with policy gradient methods (GRPO, PPO)
- Data privacy requirements
- High-volume rollout collection
- Custom or fine-tuned models

---

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
policy_model_name: gpt-4o-mini
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `openai_base_url` | `str` | Required | OpenAI API endpoint |
| `openai_api_key` | `str` | Required | API key for authentication |
| `openai_model` | `str` | Required | Model identifier |

---

## Supported Models

Any OpenAI model with function calling support:

| Model | Function Calling | Recommended Use |
|-------|------------------|-----------------|
| `gpt-4o` | ✅ | General purpose, best balance |
| `gpt-4o-mini` | ✅ | Cost-effective testing and prototyping |
| `gpt-4-turbo` | ✅ | Complex reasoning |
| `o1` / `o1-mini` | ✅ | Advanced reasoning tasks |
| `o3-mini` | ✅ | Fast reasoning with tool use |

---

## API Key Setup

1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Ensure billing is configured with available credits
3. Store securely in `env.yaml` (gitignored)

:::{important}
Never commit API keys to version control. The `env.yaml` file should be in your `.gitignore`.
:::

---

## Limitations

OpenAI's API has important limitations for RL training:

| Limitation | Impact | Workaround |
|------------|--------|------------|
| **No token IDs** | Cannot compute policy gradients | Use vLLM for training |
| **No log probabilities** | Cannot compute advantages | Use vLLM for training |
| **Rate limits** | Throttled at high volume | Implement backoff |
| **Cost at scale** | Expensive for large rollout collection | Use for prototyping only |

:::{note}
OpenAI is best suited for **prototyping and evaluation**. For RL training that requires token-level information, use {doc}`vllm`.
:::

---

## Troubleshooting

::::{dropdown} Authentication Errors
:icon: alert

```text
Error code: 401 - Incorrect API key provided
```

Verify your API key in `env.yaml` matches your OpenAI account.

::::

::::{dropdown} Quota Errors
:icon: alert

```text
Error code: 429 - You exceeded your current quota
```

Add credits at [platform.openai.com/account/billing](https://platform.openai.com/account/billing).

::::

::::{dropdown} Rate Limit Errors
:icon: alert

```text
Error code: 429 - Rate limit reached
```

Reduce `num_samples_in_parallel` in rollout collection or implement exponential backoff.

::::

---

## See Also

- {doc}`vllm` — Self-hosted inference with training support
- {doc}`azure-openai` — Enterprise Azure deployment
