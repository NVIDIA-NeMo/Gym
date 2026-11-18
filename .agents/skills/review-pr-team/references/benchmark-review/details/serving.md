# Serving: status codes, retries, masking, async (items 7-11)

Line numbers are at main `eeccee711d`.

- `R` = `nemo_gym/server_runtime/base_resources_server.py`
- `BS` = `nemo_gym/server_runtime/base_server.py`
- `BE` = `nemo_gym/server_runtime/base_environment_server.py`
- `T` = `environment_servers/single_agent_turn/app.py`
- `H` = `environment_servers/harbor/app.py`
- `HC` = `nemo_gym/client/http_client.py`

## Status codes from a Resources Server (item 7)

| Reply | When | What the Environment Server does |
|---|---|---|
| 422 | The request body fails validation (FastAPI, `BS:592`); `ResourcesSeedRejectedError` for `task_data` or a task no retry can fix (R:370-379, R:1146-1151); a network policy the provider cannot enforce (`nemo_gym/server_runtime/sandbox_lending.py:185-196`). | Terminal. The episode fails once. |
| 409 | `ResourcesSessionConflictError`: an ID bound to another episode or task, a seed after verify or after close, a cookie/typed mix-up, a close from another episode (R:382-391, R:494, R:1105, R:1139-1144, R:1183-1186). Cookie and body naming different sessions (`request_handling/session_routing.py:44`). | Terminal. |
| 404 | `ResourcesSessionLostError` on a tool call for a session the server no longer holds (R:394-406). | The harness decides; the verify masks the episode. |
| 500 | Any other exception (`BS:371-404`), including a pydantic `ValidationError` raised inside a handler, a downstream `ClientResponseError` (a dependency's 4xx becomes a 500 at this hop, `BS:372-390`), and a `CancelledError`. | Retryable: the whole episode is dispatched again on a new sandbox. |
| 5xx you raise | A failure a retry may fix, such as an unavailable sandbox provider (R:1091-1092). | Retryable. |

**The retry rule** (BE:185-210): 408, 425, 429 and every 5xx, plus a refused or dropped connection, a payload error or a timeout, are retryable. Every other 4xx is terminal. `single_agent_turn` applies it to the Resources seed (T:448, T:477), the agent seed (T:567), the agent call (T:604), the agent close (T:312) and the verify (T:688). At seed, `harbor` marks any `HTTPException` below 500 terminal, including 408, 425 and 429, and other exceptions retryable (H:887-905).

**What to flag:**

- A deterministic failure (bad row, unsupported task, unusable image, a task the server cannot grade) that raises a plain exception: it becomes a 500, and every attempt reruns the episode on a new sandbox. Ask for `ResourcesSeedRejectedError` (seed) or a 422 `HTTPException`.
- A `ValidationError` raised inside a handler (for example `Model.model_validate(body.task_data)` in `verify`): it is a 500, not a 422.
- A transient failure answered with a 4xx other than 408/425/429: the episode is never retried.
- `UnsupportedContentError` (`nemo_gym/core/media.py:51`) is a plain `ValueError`. It becomes a 422 only during body parsing; raised in a handler it is a 500. Only `pi_agent` and `opencode_agent` map it to 422 (`responses_api_agents/pi_agent/app.py:942-948`, `opencode_agent/app.py:1463-1469`). A Resources Server that decodes media maps it itself.

**Sandbox errors pass through, never become timeouts:**

- `exec()` reports a timeout as a result with `error_type="timeout"`, not an exception (`nemo_gym/sandbox/api.py:279-281`; for example `nemo_gym/sandbox/providers/docker/provider.py:738`). Check `return_code` and `error_type` on every result (repo `AGENTS.md`, Sandbox).
- `SandboxUnavailableError` (the sandbox did not answer): let it propagate so the episode is retried (`nemo_gym/sandbox/transfer.py:41-48, 136-141`).
- `SandboxTransferError` is the deterministic case a caller may grade (`transfer.py:37-38`).
- Providers raise `NgFailureEvent` with `boundary=sandbox` (`nemo_gym/sandbox/failures.py:10-40`).
- Flag code that catches these and returns a timeout, an empty result, or a reward.

## Retries: who retries what (item 8)

| Layer | What it retries | How often |
|---|---|---|
| `ServerClient` / `nemo_gym.client.http_client.request` | Exceptions only. **An HTTP error status is never retried**; the response goes back to the caller (HC:735-748). | A refused or dropped connection (`ServerDisconnectedError`, `ClientOSError`) is retried with no count limit unless the caller passes `_max_num_tries`, `_max_connection_retries` or a retry window (HC:749-799). Other exceptions, such as timeouts, stop after `MAX_NUM_TRIES = 3` on external calls only; internal Gym-to-Gym calls skip that cap (HC:373, HC:800-834). Backoff is a fixed 0.5 s, not exponential (HC:842-844). |
| Environment Server | Never reruns the episode in process. It returns an `EpisodeFailure` with a `terminal` flag (BE:240, BE:347, BE:391, BE:515). | `harbor` retries a failed image pull in process (`image_pull_retries`, default 2: H:382-385); `single_agent_turn` keeps one bounded retry of the Resources close (T:404-405). |
| Rollout collection | A delivered, non-terminal failure is dispatched again on the next pass over the same output (resume, chain resubmit) via `store.pending` (`nemo_gym/rollouts/rollout_records.py:848-881, 971-980`; `rollout_collection.py:3828, 4140`). | `NEMO_GYM_MAX_ROLLOUT_ATTEMPTS`, default 3, at least 1 (`nemo_gym/rollouts/rollout_recovery.py:684-703`). Only non-terminal failures that are not interruptions count (`rollout_records.py:661-668`). |
| Masked rows | A masked row is retried within the same cap, spending one attempt (`rollout_records.py:685-691`), when its `failure_kind` is registered and either evaluation-side or carries `failure_outcome_action: retry` (`rollout_recovery.py:422-462, 467-481`). | Never for an unregistered or namespaced `<server>:<kind>`; `judge_failed`, `judge_unparseable`, `agent_route_failed` or a `transport_*` kind (`rollout_recovery.py:420, 461`); `failure_outcome_action: fail`; a row with `_ng_failure_class`; or a row that keeps a saved answer. Those stay masked, or are re-verified from the saved answer (`nemo_gym/orchestration/recovery.py:6128-6176`). |
| Verifier (per server) | `inconclusive_verification_retries`: a fresh sandbox for an inconclusive verification. | Per server; see below. |

**`inconclusive_verification_retries` is not in core.** Each server copies its own loop and default:

| Server | Default | Inconclusive means | After the last retry |
|---|---|---|---|
| `swebench_pro` | 2 (`app.py:236`), plus `verification_total_timeout=2700` (`app.py:249`) | not completed, no parsed tests, or DNS failure without tests (`verification.py:282-301`) | masked, `verifier_error` (`app.py:809-817`) |
| `deepswe` | 2 (`app.py:129`) | an exception other than a command timeout, or no `reward.json` (`app.py:252-265`); retried only if `choose_stage_outcome` says retry (`app.py:910-926`) | masked, with `failure_outcome_action` (`app.py:1008-1009`) |
| `scale_swe` | 1 (`app.py:113`) | not completed | masked on the typed Resources-session path only (`app.py:844-845`); the cookie path (`app.py:695`) is a scored 0 |
| `vision2code` | 1 (`app.py:67`) | render infrastructure failure (`app.py:270-276`) | masked, `provider_unavailable` (`app.py:281-283`) |
| `swe_rebench`, `r2e_gym`, `swe_gym`, `swemer_*`, `swe_internal_v1`, `denovo_swe`, `swe_next` | 1 | not completed | **reward 0, `evaluation_completed: false`, not masked** (`swe_rebench/app.py:376-384`, `r2e_gym/app.py:402-416`) |

**Review rules:**

- A new sandbox-graded benchmark retries inconclusive verification on a fresh sandbox a bounded, configurable number of times (repo `AGENTS.md`, Sandbox), and masks what is still inconclusive (this guide, item 10; `AGENTS.md:173-175` asks only for `evaluation_completed=False`, which is a scored 0). Accept a default of 1 or 2; ask for the knob to be in the config, not hard-coded.
- Ask which layer retries each failure the PR adds, and for a test. A server-side retry loop on top of the collector's attempts multiplies cost: name both counts.
- Retrying a deterministic failure (a 4xx, a bad row) is a finding at every layer.

## Mask infrastructure failures; never score them as 0 (item 9)

**What removes a row from the score is `mask_sample: true`, and nothing else.** `_partition_on_mask` splits on `mask_sample` alone (`nemo_gym/metrics/reward_profile.py:892-902`).

**`evaluation_completed` is not a base field** and the score ignores it. Each server declares its own. Recovery reads it: `True` is a recorded grade (`rollout_recovery.py:130-136`); `False` is treated like a mask for evidence and counted as ungraded (`nemo_gym/orchestration/recovery.py:1050, 6155`; `nemo_gym/eval/rollout_reverification.py:1640-1649`). So **`evaluation_completed: false` without `mask_sample: true` is a scored 0.** Ten SWE servers end that way today (table above).

**Rule:** an infrastructure failure sets `mask_sample: true` with a `failure_kind`. Setting `evaluation_completed: false` as well is fine. Setting it alone is a `RISK` finding, not a `BLOCKER` (decided 2026-11-11): `AGENTS.md:173-175` tells authors to do exactly that, and ten servers on main do. Ask for `mask_sample: true` beside it, and cite this section.

An infrastructure failure scored as a plain 0, with neither field set, is still a `BLOCKER` (owner lens A1).

**Never the model's 0:**

- a sandbox that would not start, exec or answer (`SandboxUnavailableError`, `error_type` set);
- a lost or expired session: `lost_session_verification()` (R:409-432);
- a judge or model-server outage (judges mask automatically: `nemo_gym/verification/judge.py:328-365`);
- a verifier that could not run (a package download, an apt lock, a 504);
- a transcript the server could not read (Harbor's ATIF export, `harbor-format.md`).

**Still a 0:** a real task failure, such as tests that ran and failed, an empty or invalid answer, a missing `reward.txt` the agent was supposed to write. Ask which side of the line each new ending is on.

**Typed failures (the #5675/#5679 pattern, on main):**

- Raise or record an `NgFailureEvent` (`nemo_gym/core/typed_exceptions.py:54-135`; it cannot be subclassed). Boundaries: `sandbox`, `model`, `harness`, `agent`, `resources_server`, `environment`, `collection` (`nemo_gym/core/failure_records.py:39-58`). A Resources Server reports as `resources_server`.
- `failure_kind` comes from `FAILURE_KINDS` (`nemo_gym/core/failure_kinds.py:218-263`) or is namespaced `<server>:<kind>`. `BaseVerifyResponse.failure_kind` only warns on an unregistered kind (R:239-248). A namespaced kind is never retried by recovery: use a registered kind when the row should be retried.
- `observe_failure(err)` adds it to the request's failure log (`nemo_gym/core/failure_transport.py:255-259`).
- The outcome policy picks `verify`, `retry`, `fail` or `count_as_zero` (`nemo_gym/core/outcome_policy.py:31-41`), by status, then kind, then class, then default (`outcome_policy.py:105-119`). At the verification stage the default is `retry` (masked); `cancelled`, `shutdown`, `kill_shaped` and `provider_credentials_rejected` fail (`outcome_policy.py:154-177`). `count_as_zero` is honoured only with reward 0 and no mask (`outcome_policy.py:43-55`).
- The row records `failure_outcome_action` and `failure_outcome_attribution` (R:195-207).
- **Worked examples:** `resources_servers/terminal_bench_2_1/app.py:690-767` for the typed failure and policy (it has no inconclusive-retry loop); `resources_servers/deepswe/app.py:870-1012` for both the loop and the policy.

## Async, concurrency and HTTP (item 10)

- **Async endpoints.** Every route is `async`. `initialize_session_state` and `release_session_state` must be async (enforced, R:619-632); the rest is convention.
- **Bound external work** with `asyncio.Semaphore` (subprocesses, sandbox calls, judge calls). Never `ray.get()` in async code; `await` the Ray future.
- **Subprocess output** decoded with `errors="replace"`.
- **HTTP through aiohttp:** `nemo_gym.client.http_client.request()` or `ServerClient`. Not `httpx.AsyncClient`: its pooling hangs at high concurrency. Wrap a library that uses httpx with an aiohttp adapter, as `resources_servers/tavily_search/app.py` (`TavilySearchAIOHTTPClient`) does. **Nothing enforces this** (no ruff `TID251` rule in `pyproject.toml:832-840`, no test), and at least seven runtime files on main construct an `httpx.AsyncClient`: three Resources Servers (`resources_servers/math_formal_lean/sandbox_client.py:57`, `resources_servers/gdpval/tavily_search.py:48`, `resources_servers/competitive_coding_challenges/ccc_eval.py:108`) and four agent files (`harbor_agent`, `terminus_2_agent`, `stirrup_agent`). Flag a new one; don't cite the old ones as precedent.
- **Cookies.** On the session path the Environment Server keeps the Resources cookie from the seed reply and sends it to verify and in the `direct_http` grant (T:436, T:532, T:668). A server that calls another stateful server forwards its own session cookies on every downstream call (`cookies=request.cookies`).
- **Optional nested fields:** `(body.field or {}).get("key", default)`.
- **Startup.** `model_post_init` and import-time work run once per server build, up to twice per uvicorn worker. Installs (`ensure_<tool>()`), downloads and file writes are idempotent and safe to run concurrently.

## Sandboxes the server starts itself (item 11)

For a sandbox the Resources Server uses for grading (not one it lends):

- **Resolve the provider from config:** `resolve_provider_config(sandbox_provider, global_config_dict)` and merge `resolve_provider_metadata(...)` into `SandboxSpec.metadata`; labels truncated to 63 characters (repo `AGENTS.md`, Sandbox).
- **Deliver files** through `SandboxSpec.files` or `start_with_setup()`, which stop the sandbox if setup fails (`nemo_gym/sandbox/api.py:612-646`). Text goes through `upload`, never interpolated into a shell command; every value in a command is `shlex.quote`d.
- **Stop it on every path:** `try/finally` around use, and a bounded `stop()` (a timeout, or a cancellation shield as `nemo_gym/server_runtime/sandbox_lending.py:210-217` does). Per-session sandboxes keyed by session ID, popped only after `stop()` succeeds, and stopped in the lifespan shutdown. Worked example: `resources_servers/terminal_bench_2_1/app.py` (stop under `asyncio.timeout`, pop after success: `:339-345`; `finally` paths: `:599-654`; shutdown: `:266-284`). `AsyncSandbox.stop` itself has no shield (`nemo_gym/sandbox/api.py:852-880`), so a stop in a request that is being cancelled may never run.
- **`ttl_s`** on every sandbox config, above the episode deadline plus grading. Default is `None` (`nemo_gym/sandbox/config.py:226`) and nothing enforces it.
