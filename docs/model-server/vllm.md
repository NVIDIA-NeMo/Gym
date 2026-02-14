(model-server-vllm)=
# VLLMModel Server

[vLLM](https://docs.vllm.ai/) is a popular LLM inference engine. The NeMo Gym VLLMmodel server wraps vLLM's Chat Completions endpoint and converts requests/responses to NeMo Gym's native format which is OpenAI's [Responses API](https://platform.openai.com/docs/api-reference/responses) schema.

This document will:
1. Describe the historical context behind why we need such a conversion to/from Chat Completions and Responses.
2. Detail the exact differences between Chat Completions and Responses schema.
3. Provide an example of spinning up vLLM server, along with usage of VLLMModel.
4. Detail the available configuration options for VLLMModel.
5. Explain how to use the VLLMModel with multiple replicas of a model.
6. Explain the differences between offline inference through VLLMModel and training.


## How did OpenAI Responses API evolve and why is it necessary?
Fundamentally, LLMs accept a sequence of tokens on input and produce a sequence of tokens on output. Critically, even if the direct information provided to the LLM is very simple, the outputs can be de-tokenized into a string which can be parsed in a variety of different manners.

In late 2022, OpenAI released text-davinci-003 which used the [Completions API](https://developers.openai.com/api/docs/guides/completions/), accepting a prompt string on input and returning a text string as response.

In March 2023, OpenAI released [GPT-3.5 Turbo](https://developers.openai.com/api/docs/models/gpt-3.5-turbo) along with the Chat Completions API. This API accepted not just a plain prompt string, but rather a sequence of objects representing the conversation input to the model. This API also returned not just a plain text string, but an object representing the model response that contained more directly useful information parsed from the original plain text string. 

In other words, Chat Completions upgraded from Completions to provide a richer user experience in leveraging the response. For example, the Chat Completions response returned a list of "function calls" that were directly usable to select a particular function and call that function with model-provided arguments. This enabled the model to interact not just with the user but with its environment as well.

In March 2025, OpenAI released the [Responses API](https://openai.com/index/new-tools-for-building-agents/) in order to better facilitate building agentic systems. Specifically, the Responses API returned not only a single model response like Chat Completions, but rather a sequence of possibly interleaved reasoning, function calls, function call execution results, and chat responses. So previously, while a single Chat Completion was limited to just a single model generation, the Responses API could generate some model response including a function call, execute that function call on the OpenAI server side, and return both results as part of a single Response to the user.

Responses schema is also a superset of Chat Completions.

Currently, the community has still yet to shift from Chat Completions schema to Responses schema. Part of this issue is that the majority of open-source models are still being trained using Chat Completions format, rather than in Responses format.

Moving forward, Chat Completions will eventually be deprecated, but it will take time for the community to adopt the Responses API. OpenAI has tried to accelerate the effort, for example releasing additional guidance and acceptance criteria for how to implement an [open-source version of Responses API](https://www.openresponses.org/).


## Chat Completions vs Responses API schema.
The primary difference between Chat Completions and Responses API is that the Responses API Response object consists of a sequence of output items, while the Chat Completion only consists of a single model response.

The `output` list for a Response can contain multiple item types, such as:
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


## Use NeMo Gym with vLLM and a Chat Completions model
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


## Training vs Offline inference
By default, VLLMModel will not track any token IDs explicitly. However, token IDs are necessary when using Gym in conjunction with a training framework in order to train a model. For NeMo RL training workflows, use the training-dedicated config which enables token ID tracking:

```yaml
# Use vllm_model_for_training.yaml
return_token_id_information: true
```

This enables:
- `prompt_token_ids`: Token IDs for the input prompt
- `generation_token_ids`: Token IDs for generated text
- `generation_log_probs`: Log probabilities for each generated token
