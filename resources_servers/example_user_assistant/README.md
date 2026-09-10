# NeMo-Sim User–Assistant Simulation

This example runs a multi-turn conversation between two independently
configured Responses API agents:

- `example_assistant` is the policy being evaluated.
- `example_user` is an isolated NeMo-Sim-backed agent that derives the
  customer's behavioral profile, disclosure style, interaction style, and
  prompt.
- `example_user_assistant_processor` is dependency-free and owns turn
  ordering, termination, and trajectory attribution.
- `example_user_assistant` owns task-scoped shared state and verification.

The example persona is Morgan Lee, a 42-year-old building inspector whose
OCEAN traits and demographics are JSON-encoded in
`user_responses_create_params.metadata.nemo_sim`.
NeMo-Sim converts those attributes into the simulated user's model-visible
instructions. Morgan wants a vegetarian meal costing at most $20, persists
that requirement through `save_preference`, and later accepts a matching
recommendation through `accept_recommendation`. The verifier rewards only an
accepted recommendation that matches the persisted diet and budget.

## Integration boundary

NeMo-Sim and NeMo Gym both provide conversation orchestration. This example
does not nest NeMo-Sim's complete `ConversationLoop` inside Gym. Instead:

- The `nemo_sim_user` agent owns population-grounded persona formatting,
  behavioral derivation, and the user model/tool loop.
- The generic `UserAssistantProcessor` owns server routing, the episode loop,
  trajectories, and verification.
- The resources server owns shared environment state and task reward.

The user agent imports `conversation_plugin.core.persona` and
`conversation_plugin.core.behavioral` from a commit-pinned
[NeMo-Sim](https://gitlab-master.nvidia.com/sdg-research/sdg-nemo-sim)
dependency. It does not invoke Data Designer or NeMo-Sim's own model clients.
The user-agent environment overrides Data Designer's older PyArrow constraint
because NeMo Gym requires a newer PyArrow; the imported persona and behavioral
modules do not use PyArrow.

Access to NVIDIA's internal GitLab repository over SSH is required when the
user-agent environment is first created.

## Why orchestration is a processor

Neither participant owns `/run`. Each agent only handles `/v1/responses` and
may use its own instructions, tools, model, and tool-call loop. The
`UserAssistantProcessor` owns the complete episode:

```text
rollout collector
  → UserAssistantProcessor /run
      → resources server /seed_session
      → assistant agent /v1/responses
      → resources server /episode_status
      → NeMo-Sim user agent /v1/responses
          → persona + behavior prompt derivation
          → model server /v1/responses
      → resources server /episode_status
      → ... until environment termination or max_turns
      → resources server /verify
```

Cookies are forwarded through every call, so tools used by either participant
operate on the same task-scoped resources-server state.

## Run the example

Configure `policy_base_url`, `policy_api_key`, and `policy_model_name` in
`env.yaml`, then run:

```bash
.venv/bin/gym eval run \
  --config resources_servers/example_user_assistant/configs/example_user_assistant.yaml \
  --agent example_user_assistant_processor \
  --split validation \
  --output results/example_user_assistant.jsonl
```

The rollout row contains:

- `response`: assistant-only Responses API output for evaluation and training.
- `user_responses_create_params.metadata.nemo_sim`: JSON-encoded source
  persona, locale, and private user goal consumed by the user agent.
- `assistant_trajectory`: exact request/response pairs for assistant turns.
- `user_trajectory`: exact request/response pairs for simulated-user turns.
- `episode_trajectory`: ordered participant outputs, state snapshots, and the
  explicit termination event.
- `termination_reason` and `turns_completed`.

## Customize the pattern

1. Put a JSON-encoded NeMo-Sim persona, locale, and user goal under
   `user_responses_create_params.metadata.nemo_sim` in each dataset row.
2. Point `assistant_agent` and `user_agent` at any independently hosted
   Responses API agents.
3. Put assistant tools in `responses_create_params.tools` and user tools in
   `user_responses_create_params.tools`.
4. Implement `/episode_status` on the resources server. Return
   `{"terminated": bool, "reason": str | null, "state": {...}}`.
5. Keep shared mutable state on the resources server and access it through
   propagated session cookies.
6. Implement `/verify` against the final state and attributed trajectories.

Environment termination is authoritative. `max_turns` remains a bounded
fallback when the environment does not terminate naturally.
