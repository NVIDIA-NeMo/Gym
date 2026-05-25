(model-server-adapters)=
# Adapter Middleware

Adapter middleware adds an interceptor chain to a Model server's request/response path. Each interceptor can observe or mutate the payload without the host server changing. Use it to inject system prompts, drop unsupported params, cache responses, count turns, normalize reasoning fields, log tokens, and so on.

The middleware is opt-in. `adapters` is declared on `BaseResponsesAPIModelConfig`, so every Model server that inherits from `SimpleResponsesAPIModel` (i.e. all in-tree servers) accepts an `adapters` block. Omitting the block leaves behavior identical to the base server.

## Quickstart

Add an `adapters` list to any Model server config:

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
| `endpoint` | request → response | Drive the upstream HTTP call directly. **Only used by `start_adapter_proxy`** (standalone host mode) — forbidden inside `install_middleware`. |
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

## Proxy Mode for External-Inference Agents

The middleware modes above only intercept traffic at a Gym FastAPI boundary. When an Agent Server brings its own inference (e.g. `claude_code_agent` with `anthropic_base_url` pointing directly at Anthropic's API), the per-model-call traffic exits Gym at the SDK call site and the Model Server chain doesn't see it.

For that case, run an **adapter proxy** alongside the agent: a localhost uvicorn that hosts the same pipeline, with the agent's SDK `*_BASE_URL` pointing at the proxy. Configure via the `adapter_proxy` field on any `BaseResponsesAPIAgentConfig` subclass:

```yaml
responses_api_agents:
  claude_code_agent:
    anthropic_api_key: ${env:ANTHROPIC_API_KEY}
    # No model_server set → external inference mode
    adapter_proxy:
      upstream_url: https://api.anthropic.com
      adapters:
        - {name: logging, config: {}}
        - {name: log_tokens, config: {}}
        - {name: caching, config: {cache_dir: /tmp/adapter_cache}}
```

The base class starts the proxy in its `setup_webserver` and exposes the handle on `self._proxy_handle`. Subclasses point their SDK at `self._proxy_handle.url` instead of the original `*_base_url`.

**Routing semantics**: POSTs to `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages`, `/v1/embeddings` run through the pipeline. Everything else (GET `/v1/models`, batch endpoints, health checks) passes through to the upstream verbatim so SDK pre-flight works.

**Programmatic API**:

```python
from nemo_gym.adapters import start_adapter_proxy

with start_adapter_proxy(
    upstream_url="https://api.anthropic.com",
    adapters=[{"name": "log_tokens", "config": {}}],
) as proxy:
    # configure your SDK client to point at proxy.url
    ...
```

## Caveats

:::{note}
**Dual session-id systems.** Gym already attaches a session id via Starlette's `SessionMiddleware` (cookie-based). The adapter middleware introduces a second session id via the `/s/<hex>/...` URL prefix and stores it on `ctx.extra["session_id"]`. Interceptors keyed per session (`turn_counter`, `caching`) use the URL prefix only — the cookie session is invisible to them. Pick one convention per deployment and document it for callers.
:::

:::{note}
**`progress_tracking` webhook is awaited inline.** When the webhook fires it `await`s the HTTP call before returning the response to the client. A slow or stuck webhook backs up the response path. Use `every: 10` or larger to amortize the cost, and point at a fast, local sink (or a fire-and-forget queue) rather than a remote service.
:::

:::{note}
**`content-encoding` is stripped from the re-emitted response.** aiohttp auto-decompresses upstream bodies before the pipeline sees them, so re-emitting the original `Content-Encoding: gzip` would mislead the client into trying to gunzip plain bytes. Today the in-tree model servers don't gzip; if a gzipping reverse proxy is added in front of vLLM, this stripping is what prevents a "bad gzip" error on the client.
:::

:::{note}
**Hydra `_inherit_from` does not splice into the `adapters` list.** Composing a YAML on top of another with `_inherit_from` replaces the `adapters:` list wholesale rather than merging element-by-element. To extend a parent's chain, copy the parent's entries into the child config explicitly.
:::

:::{warning}
**Proxy mode is localhost-only by default.** `start_adapter_proxy` refuses any `host` other than `127.0.0.1`/`localhost` unless you pass `unsafe_allow_remote=True`. The proxy forwards the client's `Authorization` header verbatim to the upstream, so binding to `0.0.0.0` would leak the upstream API key to any caller on the network.
:::

:::{warning}
**Anthropic-via-proxy and the `ANTHROPIC_AUTH_TOKEN = "local"` fallback.** Agents like `claude_code_agent` set `ANTHROPIC_AUTH_TOKEN = api_key or "local"` when a non-default base URL is used — that branch is for local/vLLM-style upstreams that use Bearer auth. Proxy mode is **Anthropic-upstream from the SDK's POV**, so it skips that branch: `ANTHROPIC_API_KEY` flows through normally as `x-api-key`. Setting `ANTHROPIC_AUTH_TOKEN` to your real key (or to `"local"`) when proxying to api.anthropic.com would 401.
:::

:::{note}
**Per-replica proxy means per-replica disk cache.** Each agent replica owns its own proxy thread on its own kernel-assigned port. If you point multiple replicas at the same `cache_dir` for the `caching` interceptor, sqlite write contention is real — configure a per-replica path (e.g. `cache_dir: /tmp/adapter_cache_${name}_${pod_id}`) or accept the race.
:::
