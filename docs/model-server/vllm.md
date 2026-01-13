(model-server-vllm)=
# vLLM Model Server

```{warning}
This article has not been reviewed by a developer SME. Content may change.
```

[vLLM](https://docs.vllm.ai/) provides high-throughput, low-latency LLM inference. The NeMo Gym vLLM model server (`responses_api_models/vllm_model/`) wraps vLLM endpoints to provide Responses API compatibility, enabling self-hosted models to work seamlessly with NeMo Gym's agentic workflows.

::::{grid} 2
:gutter: 3

:::{grid-item-card} {octicon}`clock;1em;` **Setup Time**
15-30 minutes
:::

:::{grid-item-card} {octicon}`bookmark;1em;` **Prerequisites**

- CUDA-capable GPU(s)
- vLLM 0.8.0+ installed
- Model weights downloaded

:::

::::

---

## When to Use vLLM

Use vLLM when you need:

- **Self-hosted inference** with your own models or fine-tunes
- **Maximum throughput** for rollout collection at scale
- **Token ID information** for NeMo RL training (via `return_token_id_information`)
- **Data privacy** with on-premise deployment
- **Custom models** not available via cloud APIs

## Starting the vLLM Server

Before configuring NeMo Gym, you need a running vLLM server. Here are common startup patterns:

:::::{tab-set}

::::{tab-item} Basic Startup

```bash
pip install -U "vllm>=0.8.0"

vllm serve <model-name> \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key token-abc123
```

::::

::::{tab-item} With Tool/Function Calling

For agentic workflows with tool calling, specify the appropriate tool parser for your model:

```bash
# Example: Qwen3 model with Hermes tool parser
vllm serve Qwen/Qwen3-30B-A3B \
    --dtype auto \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --host 0.0.0.0 \
    --port 8000
```

::::

::::{tab-item} Multi-GPU (Tensor Parallelism)

For large models that don't fit on a single GPU:

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --host 0.0.0.0 \
    --port 8000
```

::::

::::{tab-item} Reasoning Models

:::{important}
Do **NOT** use vLLM's `--reasoning-parser` flag. NeMo Gym handles reasoning via `<think>` tags internally through the `uses_reasoning_parser` config option. Using both will cause conflicts.
:::

For Nemotron, QwQ, or DeepSeek-R1:

```bash
# Correct: Let NeMo Gym handle reasoning parsing
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --tool-call-parser hermes \
    --host 0.0.0.0 \
    --port 8000
```

Then enable reasoning parsing in NeMo Gym config:

```yaml
uses_reasoning_parser: true
```

::::

:::::

## NeMo Gym Configuration

Configure the vLLM model server in your config YAML:

```yaml
policy_model:
  responses_api_models:
    vllm_model:
      entrypoint: app.py
      base_url: http://localhost:8000/v1
      api_key: token-abc123
      model: Qwen/Qwen3-30B-A3B
      return_token_id_information: false
      uses_reasoning_parser: false
```

And set credentials in `env.yaml`:

```yaml
policy_base_url: http://localhost:8000/v1
policy_api_key: token-abc123
policy_model_name: Qwen/Qwen3-30B-A3B
```

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | `str` or `list[str]` | Required | vLLM server endpoint(s). Supports multiple for load balancing. |
| `api_key` | `str` | Required | API key matching vLLM's `--api-key` flag |
| `model` | `str` | Required | Model name as registered in vLLM |
| `return_token_id_information` | `bool` | Required | Enable token IDs and log probs for training |
| `uses_reasoning_parser` | `bool` | Required | Parse `<think>` tags for reasoning models |
| `replace_developer_role_with_system` | `bool` | `false` | Convert developer role to system for compatibility |
| `chat_template_kwargs` | `dict` | `null` | Override chat template parameters |
| `extra_body` | `dict` | `null` | Pass additional vLLM-specific parameters |

### Advanced: `chat_template_kwargs`

Override chat template behavior for specific models:

```yaml
chat_template_kwargs:
  add_generation_prompt: true
  enable_thinking: false  # Model-specific
```

### Advanced: `extra_body`

Pass vLLM-specific parameters not in the standard OpenAI API:

```yaml
extra_body:
  guided_json: '{"type": "object", "properties": {...}}'
  min_tokens: 10
  repetition_penalty: 1.1
```

## Load Balancing

The vLLM model server supports multiple endpoints for horizontal scaling. Requests are distributed across endpoints using session-based affinity (same session stays on same endpoint):

```yaml
base_url:
  - http://gpu-node-1:8000/v1
  - http://gpu-node-2:8000/v1
  - http://gpu-node-3:8000/v1
```

## Function/Tool Calling

NeMo Gym's vLLM integration fully supports function calling. The model server:

1. Converts Responses API tool definitions to Chat Completions format
2. Sends requests to vLLM with tool definitions
3. Parses tool calls from vLLM responses back to Responses API format

### Supported Tool Parsers

Different models require different vLLM tool parsers:

| Model Family | Tool Parser | vLLM Flag |
|--------------|-------------|-----------|
| Qwen3 | Hermes | `--tool-call-parser hermes` |
| Llama 3.1+ | Llama | `--tool-call-parser llama3_json` |
| Mistral | Mistral | `--tool-call-parser mistral` |

Refer to [vLLM's tool calling documentation](https://docs.vllm.ai/en/latest/features/tool_calling.html) for the full list.

## Reasoning Model Support

For models that output reasoning in `<think>` tags (QwQ, DeepSeek-R1, Nemotron reasoning models):

```yaml
uses_reasoning_parser: true
```

This extracts reasoning content from `<think>...</think>` tags and converts it to Responses API reasoning items, enabling proper handling in NeMo Gym workflows.

## Training Integration

For NeMo RL training workflows, use the training-optimized config which enables token ID tracking:

```yaml
# Use vllm_model_for_training.yaml
return_token_id_information: true
```

This enables:
- `prompt_token_ids`: Token IDs for the input prompt
- `generation_token_ids`: Token IDs for generated text
- `generation_log_probs`: Log probabilities for each generated token

These are required for policy gradient methods like GRPO.

### Switching to vLLM for Training

To use vLLM instead of OpenAI for training:

```bash
# Change from openai_model to vllm_model
config_paths="resources_servers/your_server/configs/your_server.yaml,\
responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"

ng_run "+config_paths=[${config_paths}]"
```

## API Endpoints

The vLLM model server exposes two endpoints:

| Endpoint | Description |
|----------|-------------|
| `/v1/responses` | Responses API (preferred for NeMo Gym) |
| `/v1/chat/completions` | OpenAI Chat Completions API |

Both endpoints route to the same underlying vLLM server, with format conversion handled automatically.

## Troubleshooting

::::{dropdown} Context Length Errors
:icon: alert

**Symptom**: Empty responses or `400 Bad Request` errors

vLLM returns errors when input exceeds the model's context length. NeMo Gym handles this gracefully by returning an empty response rather than crashing.

**Solutions**:
- Increase `--max-model-len` when starting vLLM (if GPU memory allows)
- Reduce input length in your prompts
- Use a model with longer context support

::::

::::{dropdown} Connection Errors
:icon: alert

**Symptom**: `Connection refused` or timeout errors

```bash
# Verify vLLM is running
curl http://localhost:8000/v1/models

# Check vLLM logs for errors
# Common issues: OOM, model loading failures
```

::::

::::{dropdown} Tool Calling Not Working
:icon: alert

**Symptom**: Model outputs text instead of tool calls

1. Ensure `--enable-auto-tool-choice` is set when starting vLLM
2. Verify the correct `--tool-call-parser` for your model
3. Check that the model supports function calling

::::

::::{dropdown} Chat Template Issues
:icon: alert

**Symptom**: Malformed responses or parsing errors

Some models require specific chat template settings:

```yaml
chat_template_kwargs:
  add_generation_prompt: true
```

Or use `replace_developer_role_with_system: true` if the model doesn't support the developer role.

::::

## See Also

- {doc}`/reference/faq` — vLLM usage FAQ
- [vLLM Documentation](https://docs.vllm.ai/) 🔗
- [vLLM Tool Calling Guide](https://docs.vllm.ai/en/latest/features/tool_calling.html) 🔗
