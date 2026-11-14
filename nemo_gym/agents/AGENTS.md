# Agents: contract note

This note covers `nemo_gym/agents`. It is for anyone who changes the package, writes an agent server in
`responses_api_agents/`, or reviews either. The server runtime contract it builds on is in
[`../server_runtime/AGENTS.md`](../server_runtime/AGENTS.md). The full harness guide, with the session endpoints and
their rules, is [Add or update a harness](../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx);
this note does not repeat it.

## Assumptions

- An agent server implements `responses()` and `run()`. `SimpleResponsesAPIAgent.setup_webserver` registers every
  route: `/v1/responses` and its rollout-prefixed twins, `/run`, `/aggregate_metrics`, the agent session routes and
  `/rollouts/{rollout_id}/stop`.
- Two callers drive an agent. Rollout collection calls `/run`. An environment server, such as `single_agent_turn`,
  seeds an agent session, activates it through `/v1/responses`, and closes it.
- The sandbox lives in `nemo_gym.sandbox`, below this package. `nemo_gym.agents.utils.sandbox_session` runs a harness
  inside a sandbox that the environment server or the agent owns.
- `agents` is the highest server package in the import-linter layers; `server_runtime`, `models` and `sandbox` must
  not import it.

## State and ownership

| State | Owner and lifetime | How ownership moves |
| --- | --- | --- |
| Agent session record | `SimpleResponsesAPIAgent`, in the worker process that seeded it: lock, episode ID, activation ledger, close reply. | Seed creates it; close fences it and keeps the close reply for `session_close_retry_window_seconds`. A seed whose setup fails is kept for the caller's close for the same window. |
| Harness state (`AgentSessionState`) | The agent, from `_seed_agent_session_state` until `_close_agent_session_state` returns. | Raising `AgentSessionSetupError` hands partial state to the base for close; it is never activated. A failed close keeps the state so close can retry. |
| Rollout key `/rollouts/<rollout id>` | The worker running the session, from the start of seed until close. | A stop request reaches that worker without a cookie. |
| Agent session cookie | Signed session cookie, key `agent_session_id`. | Stays set after close, so a stale activation cannot fall back to the sessionless path. |
| Sessions open at shutdown | The app's lifespan drains them within `session_close_timeout_seconds` and logs any cleanup it could not confirm. | No timer closes an active session; the environment server and the sandbox provider own normal and crash cleanup. |
| Checkpoint participation | Default `checkpoint_mode = "restart_only"`: the episode starts over after a crash. | `"exported"` agents implement `export_agent_sessions`, `restore_agent_sessions` and `retire_agent_session`. |

## Public surface and outside callers

- `_PUBLIC_API` in [`__init__.py`](__init__.py) is the surface for agent servers: the base class, session request and
  response types, outcomes, sandbox session helpers, skills and the testing helpers in `nemo_gym.agents.testing`.
- Every agent in `responses_api_agents/` imports this package; many also import `nemo_gym.sandbox`.
- Inside Gym, `environment_servers/single_agent_turn`, `swe_together` and `nemo_user_sim` call the session routes, and
  `nemo_gym.rollouts.rollout_collection` calls `/run`.
- NeMo RL (upstream `main`) does not import `nemo_gym.agents`. It reaches agents over HTTP through rollout collection,
  so the `/run` request and response shapes are its contract.

## Pitfalls

- Seed and close bookkeeping belongs to the base class. An agent implements only the hooks
  (`_seed_agent_session_state`, `_close_agent_session_state`, `_rollout_sandbox_session`);
  `tests/unit_tests/test_session_contract_repo.py` fails an agent that overrides `seed_agent_session` or
  `close_agent_session`, or seeds without a close hook.
- A seed the agent can never serve raises `AgentSeedRejectedError` (422). Any other unexpected exception answers 500,
  which the environment server retries with a new sandbox each time.
- Build model URLs with `base_url_for_run`, `resolve_model_base_url` or `model_endpoint`, so calls carry the rollout
  prefix. A call without it is not captured, and when training capture cannot prove a call's chain, the rollout is
  masked.
- `calls_model_from_sandbox` is read from the entrypoint's source, not through inheritance: each class whose
  entrypoint runs a harness that calls the model from inside the sandbox declares it.
  `assert_model_url_reachable_from_sandbox` refuses a loopback model URL for a remote sandbox.
- `report_outcome` attaches the failure cause only when the agent calls it on a failed response; overriding
  `classify_outcome` alone attaches nothing.
- With `skip_verification`, `aggregate_metrics` warns and returns empty metrics.

## Tests

```bash
pytest tests/unit_tests/agents tests/unit_tests/test_session_contract_repo.py -x -n 16
pytest responses_api_agents/<agent>/tests -x
```

An agent with sessions tests each seed refusal with `check_agent_seed_rejected` and failure paths with
`raises_agent_failure`.

## Dependency sources

Agents depend on the server runtime's FastAPI, Starlette, uvicorn and aiohttp; see the server runtime note. A harness
adapter also reads its harness's source at the version the agent pins, as the harness guide describes.
