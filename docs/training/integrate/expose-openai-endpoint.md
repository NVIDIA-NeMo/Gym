---
description: "Configure your generation backend to expose an OpenAI-compatible HTTP endpoint for Gym rollout collection"
categories: ["how-to-guides"]
tags: ["vllm", "sglang", "openai-api", "http-server", "tool-calling", "integration"]
personas: ["mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
---

(integrate-expose-openai-endpoint)=

# Expose an OpenAI-Compatible Endpoint

Configure your generation backend to serve an HTTP endpoint that Gym can connect to for rollout collection.

## How It Works

Gym communicates with your training policy through an OpenAI-compatible HTTP server. During training, Gym runs as a separate process (often on different nodes) from your generation backend. HTTP provides a clean interface boundary that works across process and node boundaries.

The endpoint must support:

- `/v1/chat/completions` for generating responses
- Token IDs and log probabilities in responses
- Tool calling for multi-step rollouts (optional)

## Before You Start

**Prerequisites**:

- vLLM or SGLang installed and working for inference
- Understanding of your model's chat template
- Python 3.10+

:::{dropdown} Supported Server Implementations
:icon: info

Gym supports any OpenAI-compatible HTTP server, including:

- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [SGLang OpenAI-compatible APIs](https://docs.sglang.ai/backend/openai_api_vision.html)
- [OpenAI API](https://platform.openai.com/docs/api-reference/chat/create) (reference implementation)

:::

---

## Usage

Enable the HTTP server in your training configuration:

::::{tab-set}

:::{tab-item} vLLM

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

:::

:::{tab-item} vLLM (Python)

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

:::

::::

:::{important}
Set `stop_strings` and `stop_token_ids` to `None`. Gym manages conversation flow and handles stopping conditions through the agent orchestration layer.
:::

---

## Configuration

```{list-table} Key Configuration Parameters
:header-rows: 1
:widths: 25 15 60

* - Parameter
  - Default
  - Description
* - `async_engine`
  - `false`
  - Required for HTTP server mode
* - `expose_http_server`
  - `false`
  - Exposes `/v1/chat/completions` and `/tokenize` endpoints
* - `http_server_serving_chat_kwargs`
  - `{}`
  - Tool parsing configuration (model-specific)
```

### Tool Call Parsers

Different models require different tool call parsers:

```{list-table}
:header-rows: 1
:widths: 30 30 40

* - Model Family
  - Parser
  - Configuration
* - Qwen 3
  - `hermes`
  - `tool_call_parser: hermes`
* - Llama 3.1+
  - `llama3_json`
  - `tool_call_parser: llama3_json`
* - Mistral
  - `mistral`
  - `tool_call_parser: mistral`
```

For the full list of supported parsers, refer to [vLLM tool calling documentation](https://docs.vllm.ai/en/latest/features/tool_calling.html).

---

## Verify the Endpoint

After your training script initializes the generation backend, verify the endpoint is accessible:

```bash
# Check available models
curl http://localhost:8000/v1/models
```

**Expected response**:

```json
{
  "object": "list",
  "data": [{"id": "your-model-name", "object": "model"}]
}
```

:::{dropdown} Test Chat Completions
:icon: check-circle

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 50
  }'
```

**Success criteria**: Response includes `choices[0].message.content` with generated text.

:::

:::{dropdown} Test Tool Calling
:icon: tools

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }],
    "max_tokens": 200
  }'
```

**Success criteria**: Response includes a `tool_calls` array when the model decides to use the tool.

:::

---

## Troubleshooting

:::{dropdown} Endpoint Not Accessible
:icon: alert

**Symptom**: Connection refused or timeout

**Solutions**:
- Ensure your generation backend has fully initialized before testing
- Check firewall rules if testing across nodes
- Verify the port is not already in use

:::

:::{dropdown} Tool Calls Not Parsing
:icon: alert

**Symptom**: Model generates tool call text but `tool_calls` array is empty

**Solutions**:
- Verify you are using the correct `tool_call_parser` for your model family
- Check that `enable_auto_tool_choice: true` is set
- Different model families use different tool calling formats

:::

:::{dropdown} Missing Log Probabilities
:icon: alert

**Symptom**: Responses do not include `logprobs`

**Solutions**:
- Ensure your request includes `logprobs: true`
- Verify your vLLM build supports log probability output

:::

---

## Next Step

Your endpoint is ready. Next, connect Gym to your training loop.

:::{button-ref} connect-gym-to-training
:color: primary
:outline:

Connect Gym to Your Training Loop →
:::

## Resources

- [vLLM serving documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- {doc}`/about/concepts/training-integration-architecture` — Architecture deep-dive
