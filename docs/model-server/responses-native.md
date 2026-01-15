(model-server-responses-native)=

# Responses-Native vs Conversion

This page explains the difference between responses-native and conversion-based model server backends.

## TL;DR

- **Responses-native (OpenAI)**: Requests pass directly to `/v1/responses` — lowest latency, simplest path.
- **Conversion (vLLM/Azure)**: Requests are converted to Chat Completions format — supports self-hosted models and training.
- **Choose responses-native** for prototyping and evaluation with GPT models.
- **Choose conversion** for training workflows that require token IDs.

---

## How Responses-Native Differs from Conversion

NeMo Gym model servers expose two endpoints: `/v1/chat/completions` and `/v1/responses`. The difference is in how each backend handles `/v1/responses` requests:

:::::{tab-set}

::::{tab-item} Responses-Native (OpenAI)

```text
Agent Server                   Model Server                    Backend
     │                              │                             │
     │──POST /v1/responses─────────►│                             │
     │                              │──POST /v1/responses─────────►│
     │                              │◄──NeMoGymResponse────────────│
     │◄──NeMoGymResponse────────────│                             │
```

**No conversion required** — the request passes directly to the backend native `/v1/responses` endpoint.

::::

::::{tab-item} Conversion (vLLM/Azure)

```text
Agent Server                   Model Server                    Backend
     │                              │                             │
     │──POST /v1/responses─────────►│                             │
     │                              │ ┌─────────────────────────┐ │
     │                              │ │ VLLMConverter           │ │
     │                              │ │ Responses → ChatCompl.  │ │
     │                              │ └─────────────────────────┘ │
     │                              │──POST /v1/chat/completions──►│
     │                              │◄──ChatCompletion─────────────│
     │                              │ ┌─────────────────────────┐ │
     │                              │ │ VLLMConverter           │ │
     │                              │ │ ChatCompl. → Responses  │ │
     │                              │ └─────────────────────────┘ │
     │◄──NeMoGymResponse────────────│                             │
```

**Conversion required** — vLLM and Azure OpenAI support Chat Completions natively, not Responses. The model server converts:

1. Responses API request → Chat Completions request
2. Chat Completions response → Responses API response

::::

:::::

---

## Supported Backends

| Backend                       | Responses-Native | Notes                                            |
| ----------------------------- | ---------------- | ------------------------------------------------ |
| [OpenAI](openai)              | ✅ Yes           | Direct `/v1/responses` pass-through              |
| [vLLM](vllm)                  | ❌ No            | Uses `VLLMConverter` for format translation      |
| [Azure OpenAI](azure-openai)  | ❌ No            | Uses `VLLMConverter` (Azure API uses Chat Completions) |

:::{note}
OpenAI's API is the sole provider with native Responses API support. vLLM and Azure OpenAI use the Chat Completions API format.
:::

---

## Benefits of Responses-Native

**Lower latency**: No conversion overhead. Requests pass directly to the backend.

**Simpler debugging**: Request/response formats match between NeMo Gym and the backend. No translation artifacts.

**Native tool calling**: OpenAI's Responses API provides first-class support for structured tool calls, conversation state, and reasoning items—without format conversion.

**Full feature support**: All Responses API features (reasoning, tool choice, parallel tool calls) work natively without compatibility shims.

**Built-in retry logic**: The client automatically retries on rate limits (429) and transient errors (500, 502, 503, 504) with exponential backoff.

---

## Configuration

Configure an OpenAI model server (responses-native) in your `gym_config.yaml`:

```yaml
policy_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      openai_base_url: ${policy_base_url}
      openai_api_key: ${policy_api_key}
      openai_model: ${policy_model_name}
```

Set credentials in `env.yaml` (in your project root):

```yaml
policy_base_url: https://api.openai.com/v1
policy_api_key: sk-your-api-key
policy_model_name: gpt-4o-mini
```

Then start the server:

```bash
ng_run "+config_paths=[gym_config.yaml]"
```

### Configuration Options

| Parameter         | Type  | Required | Description                                                   |
| ----------------- | ----- | -------- | ------------------------------------------------------------- |
| `openai_base_url` | `str` | Yes      | OpenAI API endpoint (`https://api.openai.com/v1`)             |
| `openai_api_key`  | `str` | Yes      | OpenAI API key                                                |
| `openai_model`    | `str` | Yes      | Model name (e.g., `gpt-4o-mini`, `gpt-4o`, `gpt-4.1-2025-04-14`) |

---

## When to Choose Responses-Native vs Conversion

| Consideration            | Responses-Native (OpenAI)      | Conversion (vLLM)                    |
| ------------------------ | ------------------------------ | ------------------------------------ |
| **Latency**              | Lower (no conversion)          | Higher (conversion overhead)         |
| **Cost**                 | Pay-per-token API pricing      | Self-hosted (infrastructure cost)    |
| **Data privacy**         | Data sent to OpenAI            | Data stays on-premises               |
| **Model selection**      | GPT models                     | Any Hugging Face model               |
| **Training integration** | No token IDs (evaluation)      | Full token ID support for training   |
| **Rate limits**          | Subject to OpenAI limits       | Self-managed                         |

**Choose responses-native when**:

- Prototyping or evaluating agents
- Using GPT models for production inference
- Latency and simplicity matter more than cost

**Choose vLLM conversion when**:

- Training models with GRPO/DPO (requires token IDs)
- Running custom/fine-tuned models
- Data must stay on-premises

---

## Implementation Details

The OpenAI model server (`responses_api_models/openai_model/app.py`) extends `SimpleResponsesAPIModel`:

```python
async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
    body_dict = body.model_dump(exclude_unset=True)
    body_dict["model"] = self.config.openai_model
    openai_response_dict = await self._client.create_response(**body_dict)
    return NeMoGymResponse.model_validate(openai_response_dict)
```

The method passes the request directly to `create_response()`, which calls OpenAI's `/v1/responses` endpoint. No conversion logic exists.

Compare this to the vLLM server, which uses `VLLMConverter.responses_to_chat_completion_create_params()` to translate the request format before calling `/v1/chat/completions`.

### Error Handling

The `NeMoGymAsyncOpenAI` client (`nemo_gym/openai_utils.py`) handles transient errors automatically:

- **Retry codes**: 429 (rate limit), 500, 502, 503, 504, 520
- **Backoff**: 0.5s delay between retries
- **Max retries**: Configurable (default: 3, extends for rate limits)

This applies to both responses-native (OpenAI) and conversion (vLLM/Azure) backends.

---

## Related Topics

- {doc}`openai` — Full OpenAI model server documentation
- {doc}`vllm` — vLLM model server with conversion layer
- {doc}`/about/architecture` — NeMo Gym system architecture
- [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) — Official API reference
