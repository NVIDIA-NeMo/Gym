# NeMo UserSim Resources Server

This environment initializes one deterministic NeMo UserSim scenario at the
beginning of each `UserSimEnvironmentServer` episode. The Resources Server
prepares a deterministic panel from a previously downloaded, versioned
Nemotron Personas source. It does not access NGC, generate personas with an
LLM, or run a complete Data Designer pipeline.

## Environment initialization

The benchmark configuration pins the NGC resource version to `0.0.2` and uses:

```text
benchmarks/usersim/data/personas/
└── 0.0.2/
    ├── source/
    │   ├── en_US.parquet
    │   └── en_US.manifest.json
    ├── panels/
    │   ├── en_US-n1000-seed42.parquet
    │   └── en_US-n1000-seed42.manifest.json
    └── locks/
```

For every configured locale, server startup:

1. Requires the pinned source Parquet prepared by the benchmark recipe.
2. Validates the Parquet and records its row count, size, and SHA-256.
3. Reuses a matching deterministic panel when present; otherwise streams the
   source dataset once and materializes a bounded panel.
4. Loads only the panel into memory for episode sampling.

File locks and atomic replacement prevent concurrent server processes sharing
the prepared assets from publishing partial panels. A matching source manifest
avoids hashing or scanning the full source again.

NGC and its credentials are preparation-time concerns. Run
`gym eval prepare --benchmark usersim` before starting the Resources Server.
Startup fails with that instruction when the pinned source is absent. See
[`benchmarks/usersim`](../../benchmarks/usersim/) for credential and artifact
details.

## Episode data contracts

The episode uses separate contracts for each lifecycle. Static protocol and
population settings live in YAML. A benchmark dataset row contains only task
selectors and optional per-alias Responses API overrides:

```json
{
  "responses_create_params": {"input": []},
  "usersim_sampling": {
    "locale": "en_US",
    "seed": 1042,
    "probe_type": "general_open_ended"
  },
  "model_responses_create_params": {}
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
    "personas_source_sha256": "sha256-without-prefix",
    "personas_panel_seed": 42
  }
}
```

Scenario content and selection provenance are intentionally separate. The
context does not duplicate the selected persona, probe, theme, or goal.

At `/seed_session`, the server:

1. Selects one persona from the prepared panel, plus one probe and theme,
   deterministically.
2. Stores the resolved context in task-scoped session state.
3. Records the source version, SHA-256, and panel seed for replay.
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

User and Assistant tools are configured independently in
`model_responses_create_params.user_model.tools` and
`model_responses_create_params.assistant_model.tools`. Both SimpleAgent
instances execute their own tool loops against this Resources Server while the
Environment Server forwards one shared Resources session cookie.

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
- `personas_panel_size` and `personas_panel_seed`
- `probe_mix`
- `probe_themes`
- agent, model, and resources-server references
- turn limits
- typed `protocol_config` simulation behavior

Each dataset row owns dynamic task identity:

- `usersim_sampling.locale`
- `usersim_sampling.seed`
- optional `usersim_sampling.probe_type`
- focal and optional per-alias Responses API parameters

Changing the dataset version or panel configuration creates a different cache
path rather than silently overwriting an existing panel.

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
prepare the pinned persona source before collecting rollouts:

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
    if call["alias"] in requested_aliases
]
```

Use `{"assistant_model"}`, `{"user_model"}`, or both for
`requested_aliases`. Support-model calls use distinct aliases, so judge,
summary, and API-response outputs cannot be mistaken for participant training
tokens. Participant filtering provides the explicit per-invocation contract
for downstream SFT, RL projection, or custom collation.
