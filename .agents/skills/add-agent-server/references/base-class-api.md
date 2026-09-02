# Base class API reference for agent servers

Everything here is verified against the code. `BaseServer` and `SimpleServer` live in
`nemo_gym/server_utils.py` (there is no `base_server.py` / `simple_server.py`). The agent-specific
classes live in `nemo_gym/base_responses_api_agent.py`.

## Class hierarchy

```
BaseServer (server_utils.py:567)              config: BaseRunServerInstanceConfig
└── SimpleServer (server_utils.py:705)        + server_client: ServerClient
    │                                         abstract: setup_webserver() -> FastAPI
    └── SimpleResponsesAPIAgent (base_responses_api_agent.py:67)
                                              abstract: responses(), run()
```

`SimpleResponsesAPIAgent` inherits from `BaseResponsesAPIAgent`, `AggregateMetricsMixin`, and
`SimpleServer`.

### What you must implement

| Method | Signature | Notes |
|---|---|---|
| `responses` | `(body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse` | base_responses_api_agent.py:188-190. Handles one `/v1/responses` call. You may add `request: Request` and `response: Response` parameters — FastAPI injects them, and you need both (see cookies below). |
| `run` | `(body: BaseRunRequest = Body()) -> BaseVerifyResponse` | base_responses_api_agent.py:192-194. The rollout driver. Most agents subclass `SimpleAgent` and inherit its `run()` rather than writing one. |

Optional overrides: `aggregate_metrics` (:196), and `compute_metrics` / `get_key_metrics` from
`AggregateMetricsMixin` (`nemo_gym/reward_profile.py:803-832`).

### Endpoints the base class registers for you

`SimpleResponsesAPIAgent.setup_webserver()` (base_responses_api_agent.py:70-102) registers:

- `POST /v1/responses`
- `POST /ng-rollout/{rollout_id}/v1/responses`
- `POST /ng-rollout/{rollout_id}/training-token-capture/v1/responses`
- `POST /run` (wrapped in `rollout_context`)
- `POST /aggregate_metrics`

All three `/v1/responses` variants dispatch to the **same** `self.responses` method. `GET /` (liveness)
is added by `run_webserver()` via `setup_liveness`. There is no `/health` endpoint on agent servers.

That three-way routing is the single most important thing to understand: **your one `responses()`
implementation is reached by three different URLs, and it must forward whichever one it was reached by.**
See `references/correctness-checklist.md`, pitfall 1.

## Rollout path helpers

Path segments come from `nemo_gym/config_types.py:903-904`:
`ROLLOUT_PATH_PREFIX = "ng-rollout"`, `TOKEN_CAPTURE_PATH_SEGMENT = "training-token-capture"`.

### `url_path_for_request(url_path, request)` — use inside `responses()`

base_responses_api_agent.py:167-178:

```python
def url_path_for_request(self, url_path: str, request: Optional[Request]) -> str:
    path_params = getattr(request, "path_params", None)
    rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
    request_path = getattr(getattr(request, "url", None), "path", "")
    token_capture = f"/{TOKEN_CAPTURE_PATH_SEGMENT}/" in request_path
    return f"{rollout_path_prefix(rollout_id, token_capture=token_capture)}{url_path}"
```

Reads the rollout id from the **inbound request's path params** and detects capture mode by
string-matching the **inbound path**. This is what preserves the capture mode a caller asked for.

### `url_path_for_run(url_path, body)` — use inside `run()`

base_responses_api_agent.py:144-154:

```python
def url_path_for_run(self, url_path: str, body: Any) -> str:
    return (
        f"{rollout_path_prefix(self.rollout_id_from_run(body), token_capture=self._token_id_capture_enabled())}"
        f"{url_path}"
    )
```

Derives the id from the **run body** and the capture mode from **config**, not from an inbound path —
because in `run()` there is no upstream path to inherit; this agent is the one originating the call.

Companion for agents that need a base URL rather than a path (external CLIs, LiteLLM-style clients):
`base_url_for_run(base_url, body)` (:156-165), and `resolve_model_base_url(model_server_name,
rollout_id=None)` (:180-184).

### `rollout_path_prefix(rollout_id, *, token_capture=False)` — do not call directly

Free function in `server_utils.py:1069-1074` (imported at base_responses_api_agent.py:46):

```python
def rollout_path_prefix(rollout_id: Optional[str], *, token_capture: bool = False) -> str:
    """Return the leading model-server path prefix for a rollout, if available."""
    if not rollout_id:
        return ""
    capture_segment = f"/{TOKEN_CAPTURE_PATH_SEGMENT}" if token_capture else ""
    return f"/{ROLLOUT_PATH_PREFIX}/{rollout_id}{capture_segment}"
```

`token_capture` **defaults to `False`**. Calling this directly with just a rollout id silently produces
the eval-style `/ng-rollout/<id>` prefix and drops training-token capture. Always go through
`url_path_for_request` / `url_path_for_run` / `base_url_for_run` instead. (`apply_rollout_prefix` at
server_utils.py:1077-1081 has the same trap.)

### `rollout_id_from_run(body)`

base_responses_api_agent.py:134-142:

```python
def rollout_id_from_run(self, body: Any) -> Optional[str]:
    if not self._capture_correlation_enabled():
        return None
    return maybe_rollout_id_from_run_body(body)
```

`maybe_rollout_id_from_run_body` (`nemo_gym/rollout_correlation.py:41-77`): an explicit `_ng_rollout_id`
wins (validated against `ROLLOUT_ID_PATTERN`; raises `ValueError` if malformed), otherwise
`f"{task_index}-{rollout_index}"`, with `-a{n}` appended when `attempt_index > 0`. Returns `None` if the
indices are missing.

## Capture-mode predicates

```python
def _model_call_capture_enabled(self) -> bool:
    """Whether evaluation model-call observability is enabled."""
    global_config = getattr(self.server_client, "global_config_dict", None)
    if not isinstance(global_config, Mapping):
        return False
    return bool(global_config.get(OBSERVABILITY_ENABLED_KEY_NAME, False))
```

base_responses_api_agent.py:115-120. `OBSERVABILITY_ENABLED_KEY_NAME = "observability_enabled"`
(`nemo_gym/global_config.py:106`).

Siblings: `_token_id_capture_enabled()` (:122-132) requires `token_id_capture.enabled` **plus** either
`all_agents` or this agent's `config.token_id_capture`; `_capture_correlation_enabled()` (:104-113) is
the OR of both.

## Config

`BaseResponsesAPIAgentConfig` (base_responses_api_agent.py:52-60) adds:

| Field | Default |
|---|---|
| `skip_verification: bool` | `False` |
| `skip_verification_reward: float` | `0.0` |
| `token_id_capture: bool` | `False` |

Inherited through `BaseRunServerInstanceConfig` → `BaseRunServerConfig` → `BaseServerConfig`
(`config_types.py:589-600`): `name: str` (required; unique at runtime, injected from the config path),
`entrypoint: str` (required), `host: str` and `port: int` (both required), `num_workers: Optional[int] =
None`, `domain: Optional[Domain] = None` (resources servers only).

`skip_verification` / `skip_verification_reward` are also injectable run-wide
(`global_config.py:112-113, 567-573`).

`SimpleAgentConfig` (`responses_api_agents/simple_agent/app.py:59-62`) adds `resources_server`,
`model_server`, and `max_steps`.

## Cookies and session middleware

`get_session_middleware_key()` (server_utils.py:750-753):

```python
return f"{self.__class__.__name__}___{self.config.name}"
```

This value is used as **both** the `SessionMiddleware` `secret_key` and the `session_cookie` **name** —
so every server in the run has a distinct cookie name. A cookie set by the resources server's
`/seed_session` is not the same cookie your agent server sets, which is why forwarding has to be
explicit and why cookies must be mirrored back onto the outgoing response.

`setup_session_middleware(app)` (:755-775) is idempotent (guarded by
`app.state.nemo_gym_session_middleware_installed`) and registers an `add_session_id` middleware that
assigns `request.session["session_id"] = ... or str(uuid4())` unconditionally, so Starlette always
re-emits `Set-Cookie`. `SimpleResponsesAPIAgent.setup_webserver` calls it at :73.

**The load-bearing detail:** Gym's shared aiohttp client is constructed with
`cookie_jar=DummyCookieJar()` (server_utils.py:172) — it stores nothing and resends nothing. There is
no ambient cookie handling anywhere in the stack. Every cookie that reaches a downstream server got
there because agent code passed it explicitly, and every cookie that reaches the caller got there
because agent code called `response.set_cookie(...)`.
