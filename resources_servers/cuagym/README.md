# CUA-Gym-Hub resources server

Multimodal browser environment over [CUA-Gym-Hub](https://github.com/xlang-ai/CUA-Gym-Hub)
mock web applications, speaking the same `/seed_session`, `/step`, `/verify`,
`/close` protocol as `resources_servers/webarena` — so
`responses_api_agents/webarena_agent` drives it unchanged.

## Why this exists next to webarena

WebArena is one shared mutable deployment: concurrent tasks (and repeats of
the same task) contaminate each other, which is why the webarena server
carries collision planning and deployments need periodic resets. CUA-Gym-Hub
namespaces all state by a per-episode session id (`?sid=`), so:

- every rollout gets a private copy of the app world (state injection);
- `num_repeats > 1` yields i.i.d. samples from identical initial states;
- parallelism is bounded by browser workers and hub throughput, not by
  task-interference semantics;
- there is no collision machinery and no cross-episode reset in this server.

## Self-contained rows, no files

Task rows inline everything the episode needs as code strings (produced by
the dataset converter in the CUA-Gym repo, `scripts/convert_to_nemo_gym.py`):

```json
"cuagym": {
  "bundle_id": "…",                  // provenance only
  "app_dir": "slack_mock",
  "initial_setup": "…python…",       // injects the initial app state
  "eval_reward_code": "…python…"     // scores the episode, prints REWARD: <float>
}
```

The programs contain two kinds of placeholders, both filled by the server at
episode time with plain `str.replace`:

- `__CUA_GYM_SID__` — the per-episode session id (the server mints a fresh
  uuid4 per episode; there is no sid handoff file and nothing under `/tmp`);
- `__CUA_GYM_<APP>_URL__` — hub endpoints, resolved from `hub_base_url` +
  the app's port (position in the hub's sorted `websites/*_mock` list,
  mirroring `deploy-all.sh`).

The converter also strips the bundles' trailing GUI browser launch
(`launch_gui`/google-chrome) — the server's pooled Playwright browser does
the real navigation, and the launch would leave dangling chrome processes.

## Episode lifecycle

1. **seed** — mint sid, fill placeholders into `initial_setup`, run it in a
   subprocess (source over stdin; it POSTs the initial state to
   `/post?sid=`), then navigate the pooled Playwright browser to
   `<app>/?sid=<sid>`.
2. **step** — identical structured actions to webarena
   (`navigate`/`computer`/`tabs_*`/`terminate` from
   `resources_servers.webarena.actions`).
3. **verify** — fill placeholders into `eval_reward_code`, run it (it reads
   `/go?sid=` from the hub), parse the printed `REWARD: <float>` (rubric
   scores may be fractional; values are clamped to [0, 1]). The agent's
   terminate answer/status are exposed as
   `CUA_GYM_AGENT_ANSWER`/`CUA_GYM_AGENT_STATUS` env vars.
4. **close** — release the browser context.

The dataset guarantees `reward(initial) = 0.0` and `reward(golden) = 1.0`;
the resource-only smoke asserts the former without a model.

## Configuration

```yaml
hub_base_url: http://<hub-ip>     # substituted at episode time; rotating an
hub_base_port: 8000               # ephemeral hub IP is a one-line change here
setup_timeout_seconds: 180
reward_timeout_seconds: 180
```

## Data

`data/example.jsonl` holds five tasks across distinct mocks with verified-
rendering builds. Task rows validate as `WebArenaTaskRow` (webarena_agent's row
schema): the `eval` block is an inert stub and everything CUA-Gym-specific
rides in the extra `cuagym` field.

## Smoke

See `examples/web_agent_cuagym_smoke/` in NeMo-RL (mirrors
`web_agent_vllm_smoke`): `00_run_gym_servers.sh`, `01_smoke_resource_only.py`
(no model; asserts state injection + `reward(initial)=0.0`),
`02_collect_and_summarize.sh` (real rollouts).

## Known limits

- Hub-side `.mock-states/<sid>.json` files accumulate (one per episode); the
  hub has no delete endpoint. Prune server-side if disk becomes a concern.
- Only `platform=web` bundles convert; desktop bundles need a VM executor,
  and web bundles with auxiliary asset files are skipped by the converter.
- Broken hub builds are quarantined at conversion time via `--exclude-apps`
  (at hub commit b8207bf: `monday_mock`, `outlook_web_mock`,
  `uber_eats_mock` fail to render; monday additionally violates the
  `reward(initial)=0` guarantee).
- `requirements.txt` pins `playwright==1.62.0` to match the webarena server
  venv so both resolve the same cached Chromium revision.
