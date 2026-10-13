# Single-agent multi-turn user simulation

`single_agent_multi_turn_user_sim` runs successive activations of one native agent conversation. Resources prepares the task and manages the user simulator, follow-up messages, stopping policy, and verification.

Compose a Resources Server with a continuation-capable Agent Server:

```yaml
interactive:
  environment_servers:
    single_agent_multi_turn_user_sim:
      entrypoint: app.py
      resources_server: {type: resources_servers, name: tasks}
      agent_server: {type: responses_api_agents, name: candidate}
      setup_timeout_seconds: 600
      interaction_timeout_seconds: 4800
      activation_timeout_seconds: 1800
      grading_timeout_seconds: 1800
      cleanup_timeout_seconds: 180
      default_episode_timeout_seconds: null
```

Materialized task rows contain `task_input.task_data`; Resources turns that data into the initial Responses API input. Submit episodes through `/run`, as with `single_agent_turn`.

The sequence is task seed, agent seed, ordered activation and `/next-user-turn` calls, confirmed agent close, verification, and Resources close. Activation IDs start at zero and increase by one. Later activations carry only the new user input; the adapter retains the native conversation. Identical retries join or replay the original operation; changed inputs and out-of-order IDs are rejected. This state is process-local and requires one server worker.

Resources returns `kind: continue` with the next `responses_create_params`, or `kind: stop` with a reason. Native agent output, events, usage, and timing pass through the environment without benchmark interpretation. A task runtime policy passes unchanged to the adapter, which validates its format and settings before setup.

The environment owns separate setup, interaction, turn, grading, and cleanup limits. Setup completes before the interaction clock starts. Candidate work and simulator calls share the immutable interaction window; each turn deadline is the earlier of the interaction deadline and the per-turn limit. Deadlines use UTC timestamps, so participating hosts require synchronized clocks. Owners stop their work before returning interrupted checkpoints. The environment also bounds its wait and closes an unresponsive participant before grading. Grading uses a separate deadline, and final teardown uses the existing cleanup context.

Agent adapters opt in with `_agent_continuation_capabilities()` and implement `_activate_agent_session_state()`. Their seed hook prepares the runtime once in the borrowed sandbox. Each activation must clamp execution to `activation_deadline_unix_seconds`, including after input staging, and preserve the native conversation identity. Close must confirm execution stopped and return cumulative activations and available trajectory evidence. The existing `/v1/responses` path retains its single-turn behavior.

The shared contracts belong in Gym because the environment, Resources, and independently selected adapters all exchange them. The base agent's operation ledger joins retries and fences close at the existing session boundary. `SandboxSession.begin_activation()` reuses a borrowed sandbox only after its preceding execution was reaped; its timeout resolver prevents staging from extending a deadline. Collector acceptance of producer `ng_trajectory` uses its existing identity validation and capture projection. These interfaces require no benchmark imports.
