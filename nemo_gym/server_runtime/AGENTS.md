# Server runtime and client: contract note

This note covers `nemo_gym/server_runtime` and `nemo_gym/client`. It is for anyone who changes either package or
reviews a change to it. The root [`AGENTS.md`](../../AGENTS.md) holds the repo-wide rules; this note holds what these two
packages promise and rely on. `nemo_gym/client/AGENTS.md` is a link to this file.

## Assumptions

- Every Gym server is its own process. `SimpleServer.run_webserver` reads the server's config from the global config,
  builds one `ServerClient`, builds the app, and hands it to uvicorn.
- With `num_workers > 1`, uvicorn imports the entrypoint module in each worker and serves its module-level `app`. The
  main process never serves requests. Anything set up only in the main process does not exist in the workers.
- Servers reach each other over HTTP, and only through `nemo_gym.client.http_client.request`. It is the one place trace
  context is injected and the connection pool is shared, which is why the repo bans `httpx.AsyncClient`
  (see [aiohttp vs httpx](../../fern/versions/latest/pages/infrastructure/engineering-notes/aiohttp-vs-httpx.mdx)).
- Gym servers trust their network. The session cookie is signed with a key derived from the class and server name
  (`get_session_middleware_key`), not a secret, and uvicorn proxy headers stay off unless an allowlist is configured.
- Package layering is enforced by import-linter (`[tool.importlinter]` in `pyproject.toml`): `server_runtime` sits
  below `models` and `agents` and above `client`, so it must not import either of them. The one allowed upward import
  is `client.server_client` reading `server_runtime._checkpoint.settings`.

## State and ownership

| State | Owner and lifetime | How ownership moves |
| --- | --- | --- |
| Shared aiohttp client and its control pool | One per process, bound to the event loop that created it. Connection limits are a per-server budget divided across workers. | A later `asyncio.run` in the same process gets a new client; the old one is closed. `set_global_aiohttp_client` refuses to replace a live client. |
| `ServerClient` | One per server instance, built in `run_webserver`. Caches each server's base URL by name. | Outside a Gym-launched process, `ServerClient.load_from_global_config` fetches the config from the head server. |
| Current rollout | A `ContextVar` in `nemo_gym.core.episode_context`, set per request from the `/ng-rollout/<id>` path prefix. | `ServerClient.request` puts the prefix back on calls to resources and model servers when observability or checkpointing is on. The value does not cross into a thread pool or `run_in_executor` (see `with_rollout_prefix`). |
| Session cookie | Starlette `SessionMiddleware`, keyed `session_id`. A new ID is dropped again if the request fails. | With several workers, the session router forwards each request to the worker that owns the session. The rules are in the module docstring of [`request_handling/session_routing.py`](../server_runtime/request_handling/session_routing.py). |
| Resources session state | The worker process that seeded it. `SimpleResourcesServer` owns locks, close fences and verification receipts; the server owns its state through three hooks. | Seed creates it, `verify` or close releases it, and close fences the ID for `session_close_retry_window_seconds`. A partial-rollout checkpoint can retire or restore it. |
| Checkpoint participation | Every server, through `setup_checkpoint_participant`. Resources servers and agents default to `checkpoint_mode = "restart_only"`. | The checkpoint coordinator calls participants over the control pool (`_control=True`). |

The hook contract for resources session state is in the docstrings of `SimpleResourcesServer`,
`initialize_session_state`, `release_session_state` and `close_resources_session` in
[`base_resources_server.py`](../server_runtime/base_resources_server.py). The user-facing guide is
[Stateful environments](../../fern/versions/latest/pages/environment-tutorials/stateful-environment.mdx).

## Public surface and outside callers

- `_PUBLIC_API` in each package's `__init__.py` is what code outside `nemo_gym` may import. Code inside `nemo_gym`
  imports submodules directly. `tests/unit_tests/test_public_api.py` and `test_public_api_facades.py` pin the surface.
- Servers in `resources_servers/`, `responses_api_agents/`, `responses_api_models/` and `environment_servers/` subclass
  the bases and call `ServerClient`, `request`, `raise_for_status` and `get_response_json`.
- NeMo RL (upstream `main`) imports these packages through the old paths that
  [`_moved_paths.py`](../_moved_paths.py) keeps working until their removal release:
  - `nemo_gym.server_utils`: `ServerClient`, `BaseServerConfig`, `HEAD_SERVER_KEY_NAME`;
  - `nemo_gym._checkpoint.coordination`: `discover`, `prepare`, `renew`, `retire`, `forget`, `commit`, `restore`,
    `resume`, `Participants`, `PrepareResult`, `CoordinationError`;
  - `nemo_gym._checkpoint.resources.ResourcesParticipant` and `nemo_gym._checkpoint.agent.RestartOnlyAgentParticipant`;
  - `nemo_gym.openai_utils.NeMoGymAsyncOpenAI`, in its tests.

  `_checkpoint` is private by name but is a cross-repo contract. Change a signature there together with NeMo RL.

## Pitfalls

- `ServerClient.request` serializes a Pydantic body with `model_dump(exclude_unset=True)`. A field left at its
  default is not sent, so the receiver must apply the same default.
- Calls through `ServerClient` are internal: on errors other than connection errors they retry without the
  `MAX_NUM_TRIES` limit, unless the caller passes `_max_num_tries` or `_max_connection_retries`. An HTTP error
  status is returned, not retried; call `raise_for_status` on it.
- Raise upstream HTTP errors with `nemo_gym.client.http_client.raise_for_status`, which attaches the body as
  `response_content`. The exception middleware asserts that attribute, and model servers forward the body and status
  from it.
- With checkpointing on, `/run` is never re-sent after a dropped connection, since a second send would run the
  episode attempt twice.
- A resources session abandoned before `verify` or close has no automatic expiry in the base class. A server whose
  state holds outside resources must bound it, for example with a provider-side time limit.
- Middleware order matters and is set in `run_webserver`: session routing sits outside the session and exception
  middleware, and request-context isolation is outermost. Starlette refuses `add_middleware` once the app has started.
- A server that keeps per-session state in memory and cannot use session routing must return a reason from
  `multiple_workers_unsupported_reason`, so `num_workers > 1` fails at startup instead of splitting sessions.
- A server that uses Ray declares `ray_enabled = True` on the class its entrypoint runs.

## Tests

```bash
pytest tests/unit_tests/server_runtime tests/unit_tests/client -x -n 16
pytest tests/unit_tests/test_public_api.py tests/unit_tests/test_public_api_facades.py tests/unit_tests/test_moved_paths.py tests/unit_tests/test_rollout_url_prefix.py tests/unit_tests/test_session_contract_repo.py -x
pre-commit run import-linter --all-files
```

A resources server that keeps session state checks its close and seed rules with `check_resources_session_contract`
in its own tests.

## Dependency sources

Check behavior in the source of the version `uv.lock` pins, not the latest release:

```bash
grep -A1 -E '^name = "(fastapi|starlette|uvicorn|aiohttp)"$' uv.lock
```

- FastAPI: route registration, exception handlers, `RequestValidationError`.
- Starlette: `SessionMiddleware` (when it sends `Set-Cookie`), middleware ordering, `StreamingResponse`.
- uvicorn: multi-worker startup, `HttpToolsProtocol`, graceful shutdown, `timeout_keep_alive`.
- aiohttp: `ClientSession` loop binding, `TCPConnector` limits, the exceptions `request` retries on.
