# NeMo UserSim Resources Server

This environment initializes one deterministic NeMo UserSim scenario at the
beginning of each `UserSimEnvironmentServer` episode. The Resources Server
loads a persona panel previously created by `usersim panel` during
`gym eval prepare`, hosts episode-scoped simulated tools for agentic probes,
and retains native verification evidence. It does not construct or sample the
panel at runtime.

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
  "responses_create_params": {}
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
generator, routes its User, Assistant, Judge, and Summary calls through Agent Servers, and submits the completed
episode to `/verify`. It then closes the Resources session on every outcome.

`UserSimEnvironmentServer` directly owns this protocol; there is no generic
multi-agent engine. Its native `UserSimEpisodeResponse` contains exactly one
of `result` or `failure`. A successful result retains the verifier output,
native UserSim result, and one ordered `UserSimInvocation` list for User,
Assistant, judge, and summary calls owned by the Environment Server. Each
invocation contains its semantic role, exact Responses API request and response, optional
`AgentObservationBundle`, the environment `state_after` that activation, and
an optional final `termination_reason`. Function calls and
their model-visible results remain ordered inside `response.output`.
Probe API-response calls are Resources Server implementation details and are
retained with probe runtime evidence rather than added to this invocation list.

## Probe tools and episode state

NeMo UserSim selects any Assistant tool schemas required by the resolved probe.
The Environment Server passes those schemas only to the Assistant Agent, which
owns the model/tool iteration. Each selected tool is exposed through the
standard Resources Server `POST /{tool_name}` route. The shared Resources
session cookie selects that episode's allowlist, simulated state, and verifier
evidence, so another episode cannot call or mutate those tools.

The four Agents have independent model-server references:
`user_policy_model`, `assistant_policy_model`, `judge_policy_model`, and
`summary_policy_model`. Resources-owned tool-result synthesis uses
`tool_simulation_model`, while native probe scoring uses
`probe_scorer_model`, which references `judge_policy_model` by default.
These references can share a Model Server when they use the same provider and
model. `/close_session` removes the resolved scenario and mutable runtime state.

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
- optional per-role `responses_create_params`

Changing the dataset version selects a different prepared-panel cache path.

## Supported probes

`data/example.jsonl` contains one runnable row for every first-party NeMo
UserSim probe:

- General: `general_open_ended`, `general_educational`, and `tool_calling`
- Sovereign AI: `sov_ai_facts`, `sov_ai_dynamic`, and
  `sov_ai_multilingual_parity`
- Safety: `safety_chat_pressure` and `safety_agentic`
- Financial services: `financial_services`
- Health disclosure: `health_general_disclosure`,
  `health_therapy_disclosure`, `health_triage_disclosure`, and
  `health_decision_support_disclosure`

The three probes that expose Assistant tools—`tool_calling`,
`safety_agentic`, and `financial_services`—use the episode-scoped external
runtime in the Resources Server. The remaining probes execute their native
UserSim conversation shape through the Environment Server. Asset-backed probes
derive their task from the selected persona and pinned UserSim assets; the
dataset row only needs a stable locale, seed, and probe name. The
`tool_calling` row additionally supplies its candidate tool schema.

During `/verify`, the Resources Server invokes UserSim's registered scorer for
tool use, sovereign-AI, safety, financial-services, and guarded
health-disclosure trajectories. The four health labels share
`health_disclosure_concealment`; the default health variant has no concealment
ground truth, so that scorer is intentionally not applied. The two general
probes have no registered dedicated scorer in UserSim and retain native
conversation-completion verification. A scorer rejection, inconclusive status,
structured error, or raised exception fails verification. The scorer name and
complete result envelope are retained in `verifier_data`.

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

The native episode result preserves every Environment-owned call under
`invocations`; select either participant policy by semantic role:

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
