(model-server-vllm)=
# VLLMModel Server

[vLLM](https://docs.vllm.ai/) is a popular LLM inference engine. The NeMo Gym VLLMmodel server wraps vLLM's Chat Completions endpoint and converts requests/responses to NeMo Gym's native format which is OpenAI's [Responses API](https://platform.openai.com/docs/api-reference/responses) schema.

## How To: Use NeMo Gym with a non-Responses compatible API endpoint like vLLM
Most models use Chat Completions format rather than the OpenAI Responses API schema that NeMo Gym uses natively. To bridge this gap, NeMo Gym provides a conversion layer.

As a result, we provide a Responses API to Chat Completions mapping middleware layer in the form of `responses_api_models/vllm_model`. VLLMModel assumes that you are pointing to a vLLM instance (since it relies on vLLM-specific endpoints like `/tokenize` and vLLM-specific arguments like `return_tokens_as_token_ids`).

**To use VLLMModel, just change the `responses_api_models/openai_model/configs/openai_model.yaml` in your config paths to `responses_api_models/vllm_model/configs/vllm_model.yaml`!**
```bash
config_paths="resources_servers/example_multi_step/configs/example_multi_step.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"
ng_run "+config_paths=[$config_paths]"
```

Here is an e2e example of how to spin up a NeMo Gym compatible vLLM Chat Completions OpenAI server.
- If you want to use tools, find the appropriate vLLM arguments regarding the tool call parser to use. In this example, we use Qwen3-30B-A3B, which is suggested to use the `hermes` tool call parser.
- If you are using a reasoning model, find the appropriate vLLM arguments regarding reasoning parser to use. In this example, we use Qwen3-30B-A3B, which is suggested to use the `qwen3` reasoning parser.

```bash
uv venv --python 3.12 --seed 
source .venv/bin/activate
# hf_transfer for faster model download. datasets for downloading data from HF
uv pip install hf_transfer datasets vllm --torch-backend=auto

# Qwen/Qwen3-30B-A3B, usable in Nemo RL!
HF_HOME=.cache/ \
HF_HUB_ENABLE_HF_TRANSFER=1 \
    hf download Qwen/Qwen3-30B-A3B

HF_HOME=.cache/ \
HOME=. \
vllm serve \
    Qwen/Qwen3-30B-A3B \
    --dtype auto \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --host 0.0.0.0 \
    --port 10240
```


## OpenAI Responses vs Chat Completions API
Agents and verifiers work with responses in a standardized format based on the OpenAI Responses API schema. The verifier receives an object where the `output` field conforms to the Response object output [documented here](https://platform.openai.com/docs/api-reference/responses/object#responses/object-output).

The `output` list can contain multiple item types, such as:
- `ResponseOutputMessage` - The main user-facing message content returned by the model.
- `ResponseOutputItemReasoning` - Internal reasoning or "thinking" traces that explain the model’s thought process.
- `ResponseFunctionToolCall` - A request from the model to invoke an external function or tool.

**Example**
If a chat completion contains both thinking traces and user-facing text:
```python
ChatCompletion(
    Choices=[
        Choice(
            message=ChatCompletionMessage(
                content="<think>I'm thinking</think>Hi there!",
                tool_calls=[{...}, {...}],
                ...
            )
        )
    ],
    ...
)
```
In the Responses schema, this would be represented as:
```python
Response(
    output=[
        ResponseOutputItemReasoning(
            type="reasoning",
            summary=[
                Summary(
                    type="summary_text",
                    text="I'm thinking",
                )
            ]
        ),
        ResponseOutputMessage(
            role="assistant",
            type="message",
            content=[
                ResponseOutputText(
                    type="output_text",
                    text="Hi there!",
                )
            ]
        ),
        ResponseFunctionToolCall(
            type="function_call",
            ...

        ),
        ResponseFunctionToolCall(
            type="function_call",
            ...

        ),
        ...
    ]
)
```

Reasoning traces (`Reasoning` items) are parsed before the verifier processes the output. The parsing is **model-specific**, and the verifier does not need to worry about the extracting or interpreting reasoning traces. The verifier receives these items already separated and clearly typed.




## TODO: local vllm usage

# Below is the stub


## VLLMModel configuration

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
| `base_url` | `str` or `list[str]` | — | **Required.** vLLM server endpoint(s). Supports list for load balancing. |
| `api_key` | `str` | — | **Required.** API key matching vLLM's `--api-key` flag. |
| `model` | `str` | — | **Required.** Model name as registered in vLLM. |
| `return_token_id_information` | `bool` | — | **Required.** Set `true` for training (token IDs + log probs), `false` for inference only. |
| `uses_reasoning_parser` | `bool` | — | **Required.** Set `true` for reasoning models (extracts `<think>` tags), `false` otherwise. |
| `replace_developer_role_with_system` | `bool` | `false` | Convert "developer" role to "system" for models that don't support developer role. |
| `chat_template_kwargs` | `dict` | `null` | Override chat template parameters (e.g., `add_generation_prompt`). |
| `extra_body` | `dict` | `null` | Pass additional vLLM-specific parameters (e.g., `guided_json`). |

:::{note}
The five "Required" parameters (`base_url`, `api_key`, `model`, `return_token_id_information`, `uses_reasoning_parser`) must be explicitly set in your config. There are no implicit defaults — the shipped configs (`vllm_model.yaml` and `vllm_model_for_training.yaml`) provide working examples.
:::

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

The vLLM model server supports multiple endpoints for horizontal scaling:

```yaml
base_url:
  - http://gpu-node-1:8000/v1
  - http://gpu-node-2:8000/v1
  - http://gpu-node-3:8000/v1
```

**How it works**:
1. **Initial assignment**: New sessions are assigned to endpoints using round-robin (session 1 → endpoint 1, session 2 → endpoint 2, etc.)
2. **Session affinity**: Once assigned, a session always uses the same endpoint (tracked via HTTP session cookies)
3. **Why affinity?** Multi-turn conversations and agentic workflows benefit from request locality

:::{warning}
If a load-balanced endpoint goes down, sessions assigned to it will fail. There is currently no automatic failover — restart the session or remove the failed endpoint from the config.
:::


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
