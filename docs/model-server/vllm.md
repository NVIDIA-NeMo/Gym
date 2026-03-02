(model-server-vllm)=
# vLLM Model Server

[vLLM](https://docs.vllm.ai/) is a popular LLM inference engine. The NeMo Gym VLLMModel server wraps vLLM's Chat Completions endpoint and converts requests and responses to NeMo Gym's native format, the OpenAI [Responses API](https://platform.openai.com/docs/api-reference/responses) schema.

Most open-source models use Chat Completions format, while NeMo Gym uses the Responses API natively. VLLMModel bridges this gap by converting between the two formats automatically. For background on why NeMo Gym chose the Responses API and how the two schemas differ, see {ref}`responses-api-evolution`.

## Use VLLMModel

VLLMModel provides a Responses API to Chat Completions mapping middleware layer via `responses_api_models/vllm_model`. It assumes you are pointing to a vLLM instance since it relies on vLLM-specific endpoints like `/tokenize` and vLLM-specific arguments like `return_tokens_as_token_ids`.

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


## VLLMModel configuration reference

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


### Advanced: `chat_template_kwargs`

Override chat template behavior for specific models:

```yaml
chat_template_kwargs:
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

## Use VLLMModel with multiple replicas of a model endpoint

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
3. **Why affinity?** Multi-turn conversations and agentic workflows that call the model multiple times in one trajectory need to hit the same model endpoint in order to hit the prefix cache, which significantly speeds up the prefill phase of model inference.


## Training vs Offline Inference
By default, VLLMModel will not track any token IDs explicitly. However, token IDs are necessary when using NeMo Gym in conjunction with a training framework in order to train a model. For NeMo RL training workflows, use the training-dedicated config which enables token ID tracking:

```yaml
# Use vllm_model_for_training.yaml
return_token_id_information: true
```

This enables:
- `prompt_token_ids`: Token IDs for the input prompt
- `generation_token_ids`: Token IDs for generated text
- `generation_log_probs`: Log probabilities for each generated token
