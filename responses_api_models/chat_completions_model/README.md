# Chat Completions Model

A model server for any inference provider that exposes an OpenAI-compatible `/v1/chat/completions` endpoint.

## Supported Providers

| Provider | Config |
|----------|--------|
| Fireworks | `configs/fireworks.yaml` |
| Together.ai | `configs/together.yaml` |
| OpenRouter | `configs/openrouter.yaml` |
| DeepInfra | `configs/deepinfra.yaml` |
| Nebius | `configs/nebius.yaml` |
| Friendli | `configs/friendli.yaml` |
| Baseten | `configs/baseten.yaml` |
| HF Inference | `configs/hf_inference.yaml` |
| Gemini | `configs/gemini.yaml` |
| Any OpenAI-compatible | `configs/chat_completions_model.yaml` |

## Usage

Set your credentials in `env.yaml`:

```yaml
policy_base_url: https://api.together.xyz/v1
policy_api_key: your-api-key
policy_model_name: meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
```

Then reference the provider config:

```bash
ng_run "+config_paths=[resources_servers/my_benchmark/configs/my_benchmark.yaml,responses_api_models/chat_completions_model/configs/together.yaml]"
```

Or use the generic config and set the base URL in `env.yaml`:

```bash
ng_run "+config_paths=[resources_servers/my_benchmark/configs/my_benchmark.yaml,responses_api_models/chat_completions_model/configs/chat_completions_model.yaml]"
```

## Configuration

| Field | Description | Default |
|-------|-------------|---------|
| `base_url` | Provider's OpenAI-compatible API base URL | Required |
| `api_key` | Provider API key | Required |
| `model` | Model identifier (provider-specific format) | Required |
| `uses_reasoning_parser` | Parse `<think>` tags and `reasoning_content` fields | `false` |
| `num_concurrent_requests` | Max concurrent requests to provider | `1000` |
| `extra_body` | Additional parameters merged into every request body | `{}` |

## When to Use This vs Other Model Servers

- **`chat_completions_model`** — Any hosted inference provider (eval workloads)
- **`openai_model`** — Direct OpenAI with native Responses API support
- **`vllm_model`** — Self-hosted vLLM (training + eval, supports token IDs)
