(model-server-openai)=
# OpenAI Model Server

The OpenAI model server connects NeMo Gym to [OpenAI's API](https://platform.openai.com/docs/api-reference), enabling GPT models with native function calling for agentic workflows.

**Goal**: Connect NeMo Gym to OpenAI's API for inference.

**Prerequisites**: OpenAI API key ([platform.openai.com](https://platform.openai.com/api-keys))

**Source**: `responses_api_models/openai_model/`

:::{tip}
OpenAI is ideal for **prototyping** and **baseline comparisons**. For RL training that requires token-level information, use {doc}`vllm`.
:::

---

## Quick Start

Verify your OpenAI connection works:

```python
import openai

client = openai.OpenAI(
    api_key="sk-your-api-key",  # pragma: allowlist secret
    base_url="https://api.openai.com/v1"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)
```

If this returns a response, you're ready to configure NeMo Gym.

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

:::::{tab-set}

::::{tab-item} Use OpenAI
- **Prototyping**: Quick experiments without GPU setup
- **Baseline comparisons**: Test against frontier models
- **Environment design**: Validate workflows before infrastructure investment
- **Pay-per-use**: Budget allows per-token pricing
::::

::::{tab-item} Use vLLM
- **RL Training**: Policy gradient methods (GRPO, PPO) require token IDs
- **Data privacy**: On-premise deployment for sensitive data
- **High volume**: Large-scale rollout collection (many model responses)
- **Custom models**: Fine-tuned or open-weight models
::::

:::::

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

Any OpenAI model with [function calling support](https://platform.openai.com/docs/guides/function-calling):

| Model | Function Calling | Recommended Use |
|-------|------------------|-----------------|
| `gpt-4o` | ✅ | General purpose, best balance |
| `gpt-4o-mini` | ✅ | Cost-effective testing and prototyping |
| `gpt-4-turbo` | ✅ | Complex reasoning |
| `o1` / `o1-mini` | ✅ | Advanced reasoning tasks |
| `o3-mini` | ✅ | Fast reasoning with tool use |

:::{note}
Model availability and capabilities change. Check [OpenAI's model documentation](https://platform.openai.com/docs/models) for current information.
:::

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

## Production Considerations

:::{dropdown} Rate Limit Handling
:icon: clock

OpenAI enforces rate limits that vary by tier. For high-volume workloads:

- Implement exponential backoff in your rollout collection
- Monitor `429` responses and adjust `num_samples_in_parallel`
- Consider upgrading your OpenAI usage tier for higher limits

:::

:::{dropdown} API Key Security
:icon: shield

Best practices for production:

- Store keys in environment variables or secret managers (not in code)
- Use project-scoped keys when possible (limits blast radius)
- Rotate keys periodically
- Monitor usage via OpenAI dashboard for anomalies

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

- {doc}`vllm` — Self-hosted inference with full training support
- {doc}`azure-openai` — Enterprise Azure deployment with managed endpoints
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference) 🔗
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) 🔗
