(integrate-expose-openai-endpoint)=

# Expose an OpenAI-Compatible Endpoint

Configure your generation backend to serve an HTTP endpoint that Gym can connect to for rollout collection.

::::{grid} 2
:gutter: 3

:::{grid-item-card} {octicon}`clock;1em;` **Time**
20 minutes
:::

:::{grid-item-card} {octicon}`bookmark;1em;` **Prerequisites**

- vLLM or SGLang installed and working for inference
- Understanding of your model's chat template
- Python 3.10+

:::

::::

:::{tip}
**Using NeMo RL?** This is handled automatically. Skip to {doc}`/tutorials/integrate-training-frameworks/train-with-nemo-rl`.
:::

---

## Goal

Gym communicates with your training policy through an OpenAI-compatible HTTP server. By the end of this guide, you'll have an endpoint that:

- Accepts `/v1/chat/completions` requests
- Returns responses with token IDs and log probabilities
- Handles tool calling for multi-step rollouts

---

## Understand the Requirement

Gym requires an OpenAI-compatible HTTP server similar to:

- [OpenAI API](https://platform.openai.com/docs/api-reference/chat/create)
- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [SGLang OpenAI-compatible APIs](https://docs.sglang.ai/backend/openai_api_vision.html)

**Why HTTP?** During training, Gym runs as a separate process (often on different nodes) from your generation backend. HTTP provides a clean interface boundary that works across process and node boundaries.

---

## Enable the HTTP Server (vLLM)

If your training framework uses vLLM, enable the HTTP server in your generation config:

```yaml
# your_training_config.yaml
policy:
  generation:
    vllm_cfg:
      async_engine: true           # Required for HTTP server
      expose_http_server: true     # Enables the endpoint
      
      # Tool parsing for your model (example: Qwen3)
      http_server_serving_chat_kwargs:
        enable_auto_tool_choice: true
        tool_call_parser: hermes
```

**Key settings**:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Setting
  - Purpose
* - `async_engine: true`
  - Required for HTTP server mode
* - `expose_http_server: true`
  - Exposes `/v1/chat/completions` and `/tokenize` endpoints
* - `http_server_serving_chat_kwargs`
  - Tool parsing configuration (model-specific)
```

---

## Configure Tool Parsing

Different models require different tool call parsers. Set the correct parser for your model:

::::{tab-set}

:::{tab-item} Qwen 3

```yaml
http_server_serving_chat_kwargs:
  enable_auto_tool_choice: true
  tool_call_parser: hermes
```

:::

:::{tab-item} Llama 3.1+

```yaml
http_server_serving_chat_kwargs:
  enable_auto_tool_choice: true
  tool_call_parser: llama3_json
```

:::

:::{tab-item} Mistral

```yaml
http_server_serving_chat_kwargs:
  enable_auto_tool_choice: true
  tool_call_parser: mistral
```

:::

::::

:::{note}
Refer to [vLLM tool calling documentation](https://docs.vllm.ai/en/latest/features/tool_calling.html) for the full list of supported parsers.
:::

---

## Apply Configuration in Code

In your training script, ensure these settings are applied before starting generation:

```python
def setup_generation_config(config):
    """Configure generation for Gym compatibility."""
    generation_config = config["policy"]["generation"]
    
    # Enable HTTP server (required for Gym)
    generation_config["vllm_cfg"]["async_engine"] = True
    generation_config["vllm_cfg"]["expose_http_server"] = True
    
    # Gym handles stopping, not the generation backend
    generation_config["stop_strings"] = None
    generation_config["stop_token_ids"] = None
    
    return generation_config
```

:::{important}
**Stop tokens**: Set `stop_strings` and `stop_token_ids` to `None`. Gym manages conversation flow and will handle stopping conditions through the agent orchestration layer.
:::

---

## Verify the Endpoint

After your training script initializes the generation backend, verify the endpoint is accessible:

```bash
# Replace with your actual host and port
curl http://localhost:8000/v1/models
```

**✅ Success**: You should see a JSON response listing available models:

```json
{
  "object": "list",
  "data": [
    {
      "id": "your-model-name",
      "object": "model",
      ...
    }
  ]
}
```

---

## Test Chat Completions

Send a test request to verify the endpoint handles chat completions:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [
      {"role": "user", "content": "Say hello"}
    ],
    "max_tokens": 50
  }'
```

**✅ Success**: You should receive a response with:

- `choices[0].message.content` containing generated text
- `usage` object with token counts

---

## Test Tool Calling (Optional)

If your tasks use tools, verify tool calling works:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [
      {"role": "user", "content": "What is the weather in San Francisco?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get weather for a city",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"}
            },
            "required": ["city"]
          }
        }
      }
    ],
    "max_tokens": 200
  }'
```

**✅ Success**: The response should include a `tool_calls` array if the model decides to use the tool.

---

## Troubleshooting

### Endpoint not accessible

**Symptom**: Connection refused or timeout

**Fix**: Ensure your generation backend has fully initialized before testing. Check firewall rules if testing across nodes.

### Tool calls not parsing

**Symptom**: Model generates tool call text but `tool_calls` array is empty

**Fix**: Verify you're using the correct `tool_call_parser` for your model. Different model families use different tool calling formats.

### Missing log probabilities

**Symptom**: Responses don't include `logprobs`

**Fix**: Ensure your request includes `logprobs: true` and your vLLM build supports log probability output.

---

## Next Step

Your endpoint is ready. Next, connect Gym to your training loop.

:::{button-ref} connect-gym-to-training
:color: primary
:outline:

Next: Connect Gym to Your Training Loop →
:::

---

## Reference

- [vLLM serving documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- {doc}`/about/concepts/training-integration-architecture` — Architecture deep-dive

