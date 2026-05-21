(model-server-adapters)=
# Adapter Middleware

Adapter middleware adds an interceptor chain to a Model server's request/response path. Each interceptor can observe or mutate the payload without the host server changing. Use it to inject system prompts, drop unsupported params, cache responses, count turns, normalize reasoning fields, log tokens, and so on.

The middleware is opt-in per server. Omitting the `adapters` block leaves behavior identical to the base server.

## Quickstart

Add an `adapters` list to a Model server config that supports middleware (currently `local_vllm_model_proxy`):

```yaml
policy_model_proxy:
  responses_api_models:
    local_vllm_model_proxy:
      entrypoint: app.py
      return_token_id_information: false
      uses_reasoning_parser: true

      model_server:
        type: responses_api_models
        name: ???

      adapters:
        - name: logging
          config: {}
        - name: log_tokens
          config: {}
        - name: reasoning
          config: {}
```

Run with the example config provided alongside the server:

```bash
config_paths="responses_api_models/local_vllm_model_proxy/configs/local_vllm_model_proxy_with_adapters.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Built-in Interceptors

| Name | Stage | Purpose |
|------|-------|---------|
| `logging` | request + response | Log request body keys and response status/latency. |
| `drop_params` | request | Remove named parameters from the outbound body. |
| `payload_modifier` | request | Add, remove, and rename body fields. |
| `system_message` | request | Inject a system message (prepend / append / replace). |
| `consolidate_system` | request | Merge displaced system messages into one at position 0. |
| `modify_tools` | request | Strip or add properties on `tools[].function.parameters`. |
| `turn_counter` | request | Per-session turn budget; raises `GracefulError` on exhaustion. |
| `caching` | request → response | Disk-backed cache keyed by request body (+ optional session prefix). |
| `endpoint` | request → response | Drive the upstream HTTP call directly. **Only enable in standalone mode** — duplicates the host server's forwarding otherwise. |
| `raise_client_errors` | response | Convert non-retriable 4xx responses into `RuntimeError`. |
| `log_tokens` | response | Log `usage` token counts and latency. |
| `response_stats` | response | Accumulate request count / total tokens / total latency. |
| `reasoning` | response | Normalize `<think>...</think>` content or a `reasoning` field into `reasoning_content`. |
| `progress_tracking` | response | Optional webhook ping every N completed responses. |

Stage order is validated at startup: `REQUEST → REQUEST_TO_RESPONSE → RESPONSE`. Request interceptors run in declared order; response interceptors run in reverse.

## Interceptor Reference

### `logging`

```yaml
- name: logging
  config: {}
```

### `drop_params`

```yaml
- name: drop_params
  config:
    params: ["top_k", "frequency_penalty"]
```

### `payload_modifier`

```yaml
- name: payload_modifier
  config:
    params_to_remove: ["secret_field"]
    params_to_add: { "max_completion_tokens": 4096 }
    params_to_rename: { "old_name": "new_name" }
```

### `system_message`

```yaml
- name: system_message
  config:
    system_message: "You are a careful assistant."
    strategy: prepend   # one of: prepend | append | replace
```

### `consolidate_system`

```yaml
- name: consolidate_system
  config:
    separator: "\n\n"
```

### `modify_tools`

```yaml
- name: modify_tools
  config:
    strip_properties: ["internal_flag"]
    add_properties:
      reasoning:
        type: string
        description: "model's chain-of-thought"
```

### `turn_counter`

```yaml
- name: turn_counter
  config:
    every: 1            # log on every Nth turn
    max_turns: 20       # null disables the budget
```

The session key is taken from `ctx.extra["session_id"]` (set by the middleware when the path matches `/s/<hex>/...`); otherwise a body-hash fallback is used.

### `caching`

```yaml
- name: caching
  config:
    cache_dir: /var/cache/gym-adapter
    bypass: false
```

Cache keys include the session prefix when present, so the same body in different sessions does not collide.

### `endpoint`

```yaml
- name: endpoint
  config:
    upstream_url: http://vllm-host:8000/v1
    api_key: ""
    request_timeout: 120
    max_retries: 2
    retry_on_status: [429, 502, 503, 504]
```

:::{warning}
Do not enable `endpoint` when the middleware is hosted inside a Model server (`local_vllm_model_proxy`, etc.) — the host already forwards the request to the model. Use `endpoint` only when driving the pipeline standalone.
:::

### `raise_client_errors`

```yaml
- name: raise_client_errors
  config: {}
```

429 and 408 are treated as retriable and pass through.

### `log_tokens`

```yaml
- name: log_tokens
  config: {}
```

### `response_stats`

```yaml
- name: response_stats
  config:
    every: 100
```

### `reasoning`

```yaml
- name: reasoning
  config: {}
```

Handles three shapes:
- response already has `reasoning_content` → no-op
- response message has a `reasoning` field → renamed to `reasoning_content`
- response `content` starts with `<think>...</think>` → extracted into `reasoning_content`

### `progress_tracking`

```yaml
- name: progress_tracking
  config:
    webhook_url: http://progress-endpoint/ping
    every: 10
```

Webhook errors are logged and swallowed (`best_effort = True`).

## Path-Based Session Scoping

Posts to `/s/<hex-id>/<path>` have the prefix stripped before forwarding, and `<hex-id>` is recorded as `ctx.extra["session_id"]`. Interceptors that key per-session (`turn_counter`, `caching`) use this id.

## Custom Interceptors

Register a class at runtime via `InterceptorRegistry.register`:

```python
from nemo_gym.adapters import InterceptorRegistry

InterceptorRegistry.register("my_interceptor", "myproject.adapters.my_interceptor")
```

The target module must expose a class named `Interceptor` that subclasses one of `RequestInterceptor`, `RequestToResponseInterceptor`, or `ResponseInterceptor` from `nemo_gym.adapters.types`. The class is instantiated with the YAML `config` dict as kwargs.

## Configuration Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `adapters` | `list[dict] \| null` | `null` | Ordered interceptor specs. Each entry is `{name: <str>, config: <dict>}`. `null` or `[]` disables the middleware. |

Per-interceptor `config` keys are described above.
