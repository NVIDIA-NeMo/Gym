# NeMo UserSim Resources Server

This environment initializes one deterministic NeMo UserSim scenario at the
beginning of each `UserSimEnvironmentServer` episode. The Resources Server
loads a persona panel previously created by `usersim panel` during
`gym eval prepare`. It does not construct or sample the panel at runtime.

## Environment initialization

The benchmark configuration pins the persona dataset version to `0.0.2` and
uses:

```text
benchmarks/usersim/data/personas/
└── 0.0.2/
    └── panels/
        ├── en_US.parquet
        └── en_US.manifest.json
```

For every configured locale, server startup:

1. Requires the panel and manifest prepared by the benchmark recipe.
2. Validates the panel's version, size, row count, and SHA-256.
3. Loads the panel into memory for episode sampling.

Run `gym eval prepare --benchmark usersim` before starting the Resources
Server. Preparation delegates population sampling to NeMo UserSim and treats
the resulting panel as the immutable artifact. Startup fails with that
instruction when the panel is absent or does not match its manifest.

## Episode data contracts

The episode uses separate contracts for each lifecycle. Static protocol and
population settings live in YAML. A benchmark dataset row contains only task
selectors and optional per-role Agent request parameters:

```json
{
  "sampling": {
    "locale": "en_US",
    "seed": 1042,
    "probe_type": "general_open_ended"
  },
  "agent_responses_create_params": {}
}
```

`probe_type` is optional. When omitted, the resources server selects it from
the configured `probe_mix`. The same locale and seed always resolve to the same
persona, probe, and theme for an unchanged persona dataset and server config.

The Environment Server sends only `sampling` to `/seed_session`. The server
returns both the executable scenario and its immutable selection provenance:

```json
{
  "scenario": {
    "persona": {"first_name": "Morgan"},
    "probe_type": "general_open_ended",
    "theme": {"type": "local food", "description": "Seek a practical recommendation."},
    "goal": "Seek a practical recommendation.",
    "locale": "en_US"
  },
  "usersim_context": {
    "locale": "en_US",
    "seed": 1042,
    "personas_dataset_version": "0.0.2",
    "personas_panel_sha256": "sha256-without-prefix"
  }
}
```

Scenario content and selection provenance are intentionally separate. The
context does not duplicate the selected persona, probe, theme, or goal.

At `/seed_session`, the server:

1. Selects one persona from the prepared panel, plus one probe and theme,
   deterministically.
2. Stores the resolved context in task-scoped session state.
3. Records the persona dataset version and panel SHA-256 for replay.
4. Returns a `UserSimScenario` to the Environment Server before its first participant
   invocation.

The Environment Server gives the scenario to NeMo UserSim's conversation
generator, routes its participant and support-model calls through Gym, and submits the completed
episode to `/verify`. It then closes the Resources session on every outcome.

`UserSimEnvironmentServer` directly owns this protocol; there is no generic
multi-agent engine. Its native `UserSimEpisodeResponse` contains exactly one
of `result` or `failure`. A successful result retains the verifier output,
native UserSim result, and one ordered `UserSimInvocation` list for User,
Assistant, judge, summary, and API-response calls. Each invocation contains
its model alias, executor, exact Responses API request and response, optional
`AgentObservationBundle`, the environment `state_after` that activation, and
an optional final `termination_reason`. Function calls and
their model-visible results remain ordered inside `response.output`.

## Participant tools and shared state

Probe tools are resolved by the Resources Server and exposed only to the
Assistant Agent. Each Agent executes its own tool loop against this Resources
Server while the Environment Server forwards one shared Resources session
cookie.

The agents also have independent model-server references:
`user_policy_model` and `assistant_policy_model`. Judge, summary, and
API-response calls use a third `simulation_support_model` reference. All three
inherit the standard `policy_*` settings by default, while their role-specific
`user_policy_*`, `assistant_policy_*`, and `simulation_support_*` settings can
select different endpoints or models.

The runnable example demonstrates three idempotent endpoints:

- `record_user_context`: the User stores context without repeating it in a
  message.
- `read_user_context`: the Assistant reads that task-scoped context on a later
  activation.
- `finish_episode`: the User records the environment termination reason.

`/episode_status` lets the Environment Server snapshot state after every participant
activation. `/close_session` removes both the resolved scenario and mutable
state. Different Resources session cookies never share state.

## Static and dynamic configuration

The YAML config owns static population and probe policy:

- `personas_cache_dir`
- `personas_dataset_version`
- `personas_locales`
- `probe_mix`
- `probe_themes`
- agent, model, and resources-server references
- turn limits
- typed `protocol_config` simulation behavior

Each dataset row owns dynamic task identity:

- `sampling.locale`
- `sampling.seed`
- optional `sampling.probe_type`
- optional per-role `agent_responses_create_params`

Changing the dataset version selects a different prepared-panel cache path.

## Supported probes

This first implementation supports:

- `general_open_ended`
- `general_educational`

These probes resolve a theme and user goal. The included general-open-ended
example adds Gym Agent-owned tools around that conversation. NeMo UserSim's own
tool-calling, safety, sovereign-AI, finance, health, and trajectory-evaluator
probe semantics still require probe-specific Resources Server adapters.

The current reward is an integration signal: `1.0` when both assistant and
simulated-user trajectories contain at least one turn, otherwise `0.0`. It is
not an assistant-quality benchmark score.

## Run

Configure `policy_base_url`, `policy_api_key`, and `policy_model_name`, then
prepare the UserSim panel before collecting rollouts:

```bash
gym eval prepare --benchmark usersim

.venv/bin/gym eval run \
  --benchmark usersim \
  --agent usersim_assistant \
  --output results/usersim.jsonl \
  ++observability_enabled=true \
  ++model_call_capture_dir=/absolute/path/to/model-calls
```

## Evaluation and training attribution

The native episode result preserves all calls under `invocations`; select
either participant policy without relabeling the other participant's outputs:

```python
selected = [
    {"responses_create_params": call["request"], "response": call["response"]}
    for call in rollout["result"]["invocations"]
    if call["role"] in requested_roles
]
```

Use `{"assistant"}`, `{"user"}`, or both for `requested_roles`. Judge and
Summary Agent calls use distinct roles, while API-response synthesis remains
Resources-owned. Participant filtering provides the explicit per-invocation
contract for downstream SFT, RL projection, or custom collation.
