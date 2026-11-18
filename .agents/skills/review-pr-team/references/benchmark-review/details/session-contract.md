# The Resources Server session contract (items 1-4)

Line numbers are at main `eeccee711d`.

- `R` = `nemo_gym/server_runtime/base_resources_server.py`
- `T` = `environment_servers/single_agent_turn/app.py`
- `L` = `nemo_gym/server_runtime/sandbox_lending.py`
- `GC` = `nemo_gym/config/global_config.py`
- `C` = `nemo_gym/server_runtime/testing/session_conformance.py`

## Which path new work uses (item 1)

**Environment Servers on main:**

| Server | What it is | New benchmark work |
|---|---|---|
| `single_agent_turn` | Seeds the Resources and agent sessions, grants tools and the lent sandbox, calls `/v1/responses` once, verifies, closes every session (`environment_servers/single_agent_turn/README.md:3-9`). | Yes. The default for one agent turn verified by Resources. |
| `harbor` | Runs Harbor task directories (see `harbor-format.md`). | Yes, for Harbor tasks. |
| `nemo_user_sim`, `swe_together` | Their own episode and task formats. | Only for those interaction patterns. |
| `single_agent_turn_legacy` | The same session protocol behind a flat `/run`-shaped row, for collection from flat rows (`environment_servers/single_agent_turn_legacy/README.md:3-6`). The agent and the Resources Server must implement the session contracts (README:6). | Yes, for flat-row datasets (decided 2026-11-11). A typed taskset on `single_agent_turn` is preferred; suggest it, don't block on it. |
| `legacy_agent` | Relays `/run` to an unmigrated agent without reading either side's contract (`environment_servers/legacy_agent/README.md:3-6`). | Compatibility. A finding in new work, unless the benchmark needs only one harness (decided 2026-11-13; see `review.md` item 1). |

Shipped configs on main: about 350 name `single_agent_turn_legacy`, about 200 name `legacy_agent`, 7 name `single_agent_turn`. The `legacy_agent` count is not a precedent: those configs predate the session contract.

**Deprecated agents.** Do not wire a new benchmark to them:

- `opencode_sandboxed_agent`, `pi_sandboxed_agent`, `hermes_sandboxed_agent` (startup warnings: `responses_api_agents/opencode_sandboxed_agent/app.py:125-134`, `pi_sandboxed_agent/app.py:89-94`, `hermes_sandboxed_agent/app.py:37-41`; README banners at the top of each README).
- `miniswe_sandboxed_agent`, on both paths: it warns at every startup in favour of `mini_swe_agent` sessions (`responses_api_agents/miniswe_sandboxed_agent/app.py:71-79, 244-246`; `README.md:3-9`). It stays only for Harbor tasks with `[agent].user` and existing `/run` presets.
- The `/run` path of `terminus_2_sandboxed_agent` (`app.py:87-96`, warned only when `resources_server` is set: `app.py:649-652`). Its session path is not deprecated.
- `harbor_agent` and `harbor_agent_general` (#4826; see `harbor-format.md`).
- The generated `legacy_agent` relays log "DEPRECATED" (GC:2189-2213). Turning that into an error is opt-in (GC:1275-1283).

**What is not enforced.** A config that names `legacy_agent` explicitly gets no warning. Only the docs steer away from it (`fern/versions/latest/pages/agent-interactions/index.mdx:69-73`). Nothing checks that an agent bound to `single_agent_turn` or `single_agent_turn_legacy` implements sessions; inside `nemo_gym/`, `AGENT_SESSION_AGENT_TYPES` (GC:224-248) is read only when an agent is swapped (GC:993). The reviewer is the check.

**The scaffold still generates `legacy_agent`** (`nemo_gym/environment/scaffold.py:350-360, 594-601`). A PR built from `gym env init` output carries it unless the author moved it. See `known-conflicts.md`.

## Seed (item 2)

- **Typed envelope.** The Environment Server sends `ResourcesSeedSessionRequest{resources_session_id, episode_id, task_id, task_data}` with `extra="forbid"` (R:284-300; T:428-433). A `seed_session` override that accepts only the older cookie body fails every seed. The migration script reports such an override as a blocker (`scripts/add_legacy_agent_environment_servers.py:455-486`).
- **Hooks, all or none.** A stateful server implements `has_session_state` (sync), `initialize_session_state` and `release_session_state` (both async). A class that implements only some fails at class creation (R:619-632).
- **Validate `task_data` with `seed_request_model`** (R:587). The base validates it at seed and turns a `ValidationError` into `ResourcesSeedRejectedError` (R:1147-1151). No server on main sets it yet; ask for it on a new stateful server.
- **Refusals.**
  - Input no retry can fix: `ResourcesSeedRejectedError`, HTTP 422, terminal (R:370-379).
  - An ID bound to another episode or task, a seed after verify, or a seed after close inside the fence: `ResourcesSessionConflictError`, HTTP 409, terminal (R:382-391, R:1137-1142).
  - Never a plain exception: it becomes a 500 and the whole episode is dispatched again (R:1088-1092).
  - A failure a retry may fix, such as an unavailable sandbox provider, stays a 5xx (R:1091-1092).
- **Idempotent.** Repeating the same identifier and episode returns the existing session without initializing again (R:285-288, R:1145).
- **Reply.** It echoes `resources_session_id` and sets the session cookie (R:1161; T:440-444).
- **What the seed sees.** `task_data` only. `_ng_*` keys, `agent_ref` and `task_source` are stripped (`nemo_gym/data/task_materialization.py:44-47`); the server must not read rollout coordinates. On main the seed does not see `responses_create_params`; only `/verify` does (T:696-702). #5987 (open) adds it to the seed (see below).

## Close (item 2)

- Releases by `resources_session_id` even with no cookie, because a lost seed reply leaves none (R:1171-1179; checked at C:45-47, C:135-149).
- Succeeds for an unknown or already-closed ID. Refuses another episode with 409 (R:1178-1190).
- Raises when release fails, so the caller can retry, and keeps ownership until it succeeds: release runs before the session is forgotten and fenced (R:1167-1168, R:1187-1189).
- Fences the ID for `session_close_retry_window_seconds`, so a racing seed cannot recreate it (`ResourcesSessionFences`, R:449-511).
- `single_agent_turn` verifies first and closes in cleanup (T:663-670, T:405).
- Do not add a `/close_session` route of your own; override the hooks.

## Verify (item 2)

- **Body.** `task_data` fields, `responses_create_params` and the agent's `response`, plus `failure_report`; the cookie identifies the episode (T:667, T:696-702). The verify model must not forbid extra fields.
- **Reward.** A float. Infrastructure failures set `mask_sample=True` with a `failure_kind` from the shared vocabulary (R:194-237). See `serving.md` for masking.
- **Lost session.** A server whose score depends on per-session state answers a verify for a session it no longer holds (never seeded, closed, lost) with `lost_session_verification(...)` merged over the body: reward 0.0, `mask_sample: true`, `failure_kind: session_lost` (R:409-432, R:1192-1199). Never a bare 0, and never an exception that reruns the episode.
- **Tool call on a lost session.** Raise `ResourcesSessionLostError`, HTTP 404 (R:394-406).
- Hook servers get the lost-session answer from `verifying()`, where a closed session reports `is_lost` (R:747-819).

## State and workers (item 2)

- Key state by `request.session[SESSION_ID_KEY]`, which equals `resources_session_id` (R:435-443). State keyed by another body field is not routed to the owning worker (`fern/versions/latest/pages/environment-tutorials/resources-server-apis.mdx:436-440`).
- Hook servers route sessions to their owning worker by default, so they need no `num_workers: 1` (R:580-583, R:617). A server that cannot run with several workers overrides `multiple_workers_unsupported_reason` (`nemo_gym/server_runtime/base_server.py:252-258`), as `resources_servers/visual_coding_opencode/app.py:209-210` does.

## Sandbox lending (item 3)

- **Shape.** `SandboxAccess{connection: DirectSandboxConnection{kind: "direct", provider_config_ref, descriptor}, workdir, user?, env?}` (`nemo_gym/sandbox/access.py:11-36`).
- **Use `LentSandboxes`** instead of hand-writing start and stop (L:37-92):
  - `lend()` at seed, under the session lock; idempotent.
  - `get()` to grade in the sandbox before close (L:94-96).
  - `release()` at close; a failed stop raises and keeps the sandbox so the close can retry (L:151-161).
  - `release_all()` from the server's lifespan shutdown (L:163-175).
- **The harness cannot destroy it** on providers that enforce lease scopes: the descriptor has `scope="operate"`, and providers without leases ignore the scope (L:45-46, L:198-207). Either way, only the Resources Server stops a lent sandbox.
- **Stops survive cancellation.** Shielded, bounded by `STOP_TIMEOUT_S` = 120 s (L:27-28, L:210-217).
- **`ttl_s`** on `sandbox_config` above the episode deadline plus grading and cleanup (L:47-49). It defaults to `None` (`nemo_gym/sandbox/config.py:226`), and nothing enforces it.
- **Network.** `network_access` is `inherit`, `model_only` or `model_and_tools`. A policy the provider cannot enforce is a 422 at seed (L:34, L:185-196).
- **Worked example.** `resources_servers/visual_coding_opencode/app.py:233-298` (lifespan, `release_all`, its own lock and conflict logic) with `ttl_s: 7200` in its config.
- **Gap on main.** The base hook seed returns only `resources_session_id` (R:1161), so a lending server cannot return `sandbox_access` through the hooks and hand-writes the lock, fence and conflict logic. Review that logic against `check_resources_session_contract`.

`single_agent_turn` forwards only `sandbox_access` (and the MCP grant) from the Resources seed to the agent seed (T:535-553). The seed reply is validated as `ResourcesSeedSessionResponse`, which forbids extra fields (R:317-326; T:442), so a server that returns a `sandbox_handle` or `sandbox_descriptor` (for example `resources_servers/denovo_swe/app.py:114-115, 279`) fails every episode at seed, terminally.

## Pairing and tool grants (item 4)

**`allowed_agents`:**

- Declared on the Resources Server block as agent types; absent means any harness (GC:264-265, GC:2100-2124). It can also be set per dataset (`nemo_gym/config/config_types.py:581, 680`).
- Enforced when an agent is swapped at config composition (GC:1124-1165) and by rollout collection before dispatch (`nemo_gym/rollouts/rollout_collection.py:5229-5323`). The Environment Server does not check it at runtime.
- Waived by `allow_unsupported_pairing`, `--allow-unsupported-pairing` or `NEMO_GYM_ALLOW_UNSUPPORTED_PAIRING=1` (GC:160-161, GC:2216-2220). The env var waives every pairing guard, including `allowed_model_types`.
- An addition needs a real pairing run and the benchmark owner's review (`.agents/skills/add-harness/SKILL.md:29, 166`).

**Tool grants (`single_agent_turn`):**

- Opt-in on the Environment Server: `resources_tool_transports: [direct_http, mcp]` (T:127; README:22-32). Empty means no Resources tools.
- `direct_http` grants `DirectHTTPToolAccess(name="<server>.direct_http", required=True, base_url, cookies=<resources cookies>)` (T:518-534).
- `mcp` needs `resources_tools` in the seed reply, which `expose_tools_over_mcp: true` produces (R:150-151; `nemo_gym/server_runtime/mcp_auto_exposure.py:631-645`). Missing, every episode fails as terminal at seed (T:487-492). Unlike `tool_servers`, this is not checked at startup.
- `tool_servers` add MCP grants from other Resources Servers, checked at startup (T:153-169, T:453-483).
- Every grant is `required=True`. What required and optional mean: `nemo_gym/server_runtime/tool_access.py:34-41`.
- `simple_agent` refuses a required MCP grant, more than one `direct_http` grant, and any `sandbox_access` (`responses_api_agents/simple_agent/app.py:365-377`). Check the pairing, not only the server.

## #5987: the prompt on the typed seed (open)

- Adds `responses_create_params: dict[str, JsonValue] | None` to `ResourcesSeedSessionRequest`, left off the wire when unset (#5987 R:301-314).
- `single_agent_turn` (and so `single_agent_turn_legacy`) sends it on the Resources seed only; tool-server seeds leave it out (#5987 T:433-437).
- On the hook path the server receives it only if `seed_request_model` declares a `responses_create_params` field (#5987 R:1083-1091). A server that overrides `seed_session` reads `body.responses_create_params`.
- Version skew: a Resources Server venv built from main before #5987 fails every #5987 seed, terminally. A hook server answers 422 (`extra="forbid"`, R:295, R:303-315); a stateless server reads it as a cookie seed and answers `{}` (R:1094-1097), which `single_agent_turn` refuses (T:440-444).
- Review rule until it merges (decided 2026-11-11): only a server that reads the prompt at seed declares the field, and its PR says it depends on #5987. Other servers leave it undeclared, and the hook path drops it for them (#5987 R:1083-1091). Don't ask them to declare it.

## Test with the checker

`check_resources_session_contract(app, seed, keeps_state=..., holds_state=..., verify_body=...)` (C:26-52) drives the real routes: seed, repeated seed, close, repeated close, close of a never-seeded ID and a cookieless close. With `keeps_state=True` it also checks the 409s (conflicting seed, close from another episode, seed after close), and with `verify_body` a verify after close that must be masked as `session_lost` or `session_expired` with reward 0 (C:55-56, C:71, C:100, C:157). A stateful server's tests call it with `keeps_state=True` (`.agents/skills/add-benchmark/SKILL.md:258`).

## The in-repo example breaks the rules

`resources_servers/example_session_state_mgmt/` is the session example `single_agent_turn` runs, but:

- bad `task_data` raises a plain validation error, a 500 (`nooa_app.py:47`);
- a tool call on a closed session gets 409, not 404 (`nooa_app.py:75-79`);
- verify on a lost session returns reward 0 (`app.py:109-117`).

Do not accept "the example does it" as a reason. See `known-conflicts.md`.
