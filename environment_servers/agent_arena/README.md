# AgentArena Environment Server

Runs each row of the [`agentic_turn_simulation`](../../resources_servers/agentic_turn_simulation/README.md)
environment (AgentArena and its training rows) as one episode, with no Resources Server in the episode. The policy
plays one whole agent turn through
[`agentic_turn_simulation_agent`](../../responses_api_agents/agentic_turn_simulation_agent/README.md), against tools
this server simulates, and this server scores the turn. It is a `BaseEnvironmentServerWithTools`: the row's tools are
the episode's tools.

## One episode

1. **Set up** the turn from the whole row: pick its harness, replay its history into the model of the agent's files,
   prepare the tool simulator's prompt, arm the row's obstacle.
2. **Serve** the row's tools for the episode (`serve_tools`). Each is `POST /<tool name>` on this server, reached with
   the episode token of the agent's grant. A call is answered as the resources server's `/simulate_tool_call` answers
   it: in code from the model of the files, else by the tool simulator LLM, or with the row's obstacle. The tools are
   the ones the scoring accepts (the row's, or Cline's for a Cline row); the agent answers a call to another tool
   itself, and scoring gates it as `nonexistent_tool`. A row tool that no episode can serve (a name a grant cannot
   carry, parameters that are not an object schema, or the name of one of this server's routes) fails the episode
   here, before the seed: terminal, with stage `seed` and the cause in the failure reason.
   [`validate_rows.py`](../../resources_servers/agentic_turn_simulation/scripts/validate_rows.py) checks rows for this
   before a run (with `--mcp` when `tool_transports` include `mcp`, which mounts the `/mcp` route).
3. **Seed** an agent session with that grant and the row's harness in its `agent_context` (`TurnHarnessContext`: the
   stop tools, the tool-call protocol and the tool names), and **invoke** the agent with the row's
   `responses_create_params` as they are. A task row's own `agent_context` goes with it, and the agent validates it:
   `agentic_turn_simulation_agent` accepts only the harness keys, so it refuses (422) a row that carries any context.
   A row context that sets a harness key fails the episode before the seed.
4. **Close** the agent session. `agentic_turn_simulation_agent` raises on a model error, so its reply is an HTTP 500
   and the episode fails as a retryable agent failure, as on the relay. Its only failed reply with an `AgentOutcome` is
   a refused grant (`environment_protocol_violation`), which `agent_outcome_policy` fails terminally: every attempt
   would meet the same refusal. An agent that classifies more of its failures gets the policy's retry, fail or score.
5. **Score** the turn in process: masking, code gates, judges. The result carries the same fields as the resources
   server's `/verify` response, and `/aggregate_metrics` computes the same metrics (plus Gym's performance metrics).
   A judge that gives no verdict (unreachable, or refusing a conversation too long for it) masks the turn with
   `failure_kind: judge_failed`, as that `/verify` does, so `gym eval reverify --judge-failed-only` re-judges it from
   the saved answer.
   With `skip_verification`, the turn gets `skip_verification_reward` instead, unscored and marked
   `verification_skipped: true`, and `/aggregate_metrics` reports the performance metrics only.
6. **Stop** serving the tools in the episode's cleanup.

### What the simulator sees

A tool call reaches the server as its tool's name and arguments. The simulator's view of the turn so far is rebuilt
from the episode's earlier calls and their results (`TurnToolset.turn_so_far`), after the row's history. On the relay,
the agent sent the turn itself, so the simulator also saw what the policy wrote between its calls; here it does not.

A benchmark's flat row gets a flat result, as `single_agent_turn_legacy` answers it; a typed episode request gets the
episode reply.

## Configuration

```yaml
agentic_turn_simulation_environment_server:
  environment_servers:
    agent_arena:
      entrypoint: app.py
      agent_server:
        type: responses_api_agents
        name: agentic_turn_simulation_agent
      resources_server:
        type: resources_servers
        name: agentic_turn_simulation_resources_server
      sandbox_provider:
        local: {}
```

| Setting | Default | What it does |
| --- | --- | --- |
| `agent_server` | required | The agent that plays the turn: `agentic_turn_simulation_agent`, or one that takes the row's harness from its seed's `agent_context` (`TurnHarnessContext`) and calls the tools through its direct HTTP grant the same way. |
| `resources_server` | required | The `agentic_turn_simulation` instance whose tool simulator, judge and scoring settings the episode uses. No call goes to it during an episode; `gym eval reverify` re-scores saved rows through it, with the same settings. |
| `agent_outcome_policy` | Gym's default | What a failed agent reply with an `AgentOutcome` does: retry the episode, fail it, or score the turn. `agentic_turn_simulation_agent` sends one only for a refused grant (`environment_protocol_violation`: fail). |
| `default_episode_timeout_seconds` | `21600` | The episode's wall-clock bound. The agent's own `max_time_seconds` usually ends the turn first. |
| `cleanup_timeout_seconds` | `180` | The bound of each cleanup step (agent close, the end of the tools). |
| `skip_verification`, `skip_verification_reward` | unset | Return the fixed reward (default `0.0`) without scoring the turn. Unset, each is the bound agent's, which Gym fills from the global `skip_verification` and `skip_verification_reward` flags, so a run moved off the relay keeps skipping as the agent's `/run` did. |
| `sandbox_provider` | required | `{local: {}}`: no episode starts a sandbox. |
| `tool_transports`, `tool_call_timeout_seconds` | `[direct_http]`, none | As `BaseEnvironmentServerWithTools` defines them. `agentic_turn_simulation_agent` takes a direct HTTP grant. |

`benchmarks/agent_arena/config.yaml` and the environment's own
`resources_servers/agentic_turn_simulation/configs/agentic_turn_simulation.yaml` use this server. Their
`*_legacy_agent.yaml` twins keep the deprecated `legacy_agent` relay.

## Migrating from the legacy_agent relay

The relay configs are deprecated and will be removed after the Environment Server migration. The agent logs a
deprecation warning at its first turn on the relay. Replace each legacy config with its twin:

| Replace (deprecated relay) | With (this server) |
| --- | --- |
| `benchmarks/agent_arena/config_legacy_agent.yaml` | `benchmarks/agent_arena/config.yaml` |
| `resources_servers/agentic_turn_simulation/configs/agentic_turn_simulation_legacy_agent.yaml` | `resources_servers/agentic_turn_simulation/configs/agentic_turn_simulation.yaml` |
| `--benchmark agent_arena/config_legacy_agent` | `--benchmark agent_arena` |

For example, replace

```bash
gym env start \
  --config benchmarks/agent_arena/config_legacy_agent.yaml \
  --model-type vllm_model \
  --model-url http://<policy-host>:8000/v1 --model <policy-model-name> --model-api-key EMPTY \
  ++judge_base_url=http://<judge-host>:8000/v1 ++judge_api_key=EMPTY ++judge_model_name=<judge-model-name>
```

with

```bash
gym env start \
  --config benchmarks/agent_arena/config.yaml \
  --model-type vllm_model \
  --model-url http://<policy-host>:8000/v1 --model <policy-model-name> --model-api-key EMPTY \
  ++judge_base_url=http://<judge-host>:8000/v1 ++judge_api_key=EMPTY ++judge_model_name=<judge-model-name>
```

Everything else stays: the rows, the agent instance and its settings (the same
`++agentic_turn_simulation_agent.responses_api_agents.agentic_turn_simulation_agent.max_time_seconds=1200` override,
for example), the resources server's settings and their overrides, `skip_verification`, the `gym eval run` command,
the result fields and metrics, and `gym eval reverify`. What changes:

- The simulator no longer sees what the policy wrote between its tool calls (see "What the simulator sees").
- `obstacle_injection` is refused at startup; keep the relay config to run obstacle injection.
- This server may run several workers; it routes each tool call to the worker that runs its episode. Each worker
  keeps its own `tool_simulator_max_concurrency` and `judge_max_concurrency` limits.
- The turn's state lives in its episode and goes with it: the resources server's `session_ttl_seconds` no longer
  applies.
- A config that inherits `agentic_turn_simulation_environment_server` (`_inherit_from`) and overrides
  `environment_servers.legacy_agent.*` no longer resolves: the instance is `environment_servers.agent_arena` now, and
  the override adds a second server type to it. Move the overrides under `environment_servers.agent_arena`
  (`agent_server` and `resources_server` keep their names; see [Configuration](#configuration)), or inherit from the
  `_legacy_agent` twin to stay on the relay.

On all 433 AgentArena validation rows, two rollouts each, with one checkpoint as policy, simulator and judge, the mean
measured reward was 0.952 here and 0.949 on the relay: a paired difference by task of +0.0013 (standard error 0.0109).

## Limits

- **Restart-only checkpointing.** With rollout checkpointing on, a crash restarts the episode from its input; a
  half-played turn is never continued.
- **No sandbox.** A call is answered from the model of the agent's files or by a model call, so no episode starts a
  sandbox and `sandbox_provider` is `local`. The server logs the base class's warning about `local` at startup.
- **Typed tasksets keep masked turns.** A masked time-cap or simulator-failed turn saves `_ng_failure_stage`, which
  rollout collection refuses in a typed episode's result, so the typed result leaves it out (the flat result keeps it,
  as the relay's row does). Recovery then keeps such a row of a typed taskset as a final masked result, where a flat
  row is retried. Admitting `_ng_failure_stage` in an episode result, in Gym's core, would close this for every
  resources server behind `single_agent_turn` as well.
- **No verifier invocation record.** The judges' calls run in a verifier invocation context, so with observability on
  the rollout's trajectory lists the verification as a verifier invocation built from the captured judge calls
  (`ng_trajectory.verifier_invocations`, status `unknown`). The result carries no `ng_verifier_invocations` record
  with its outcome, which the resources server's `/verify` adds (`observe_verifier`). `observe_verifier` turns any scoring error into a 500 reply, which
  would hide a real scoring failure from this server's own failure handling, so it is not reused as it is.
- **No obstacle injection.** An obstacle's decision needs the reply a call belongs to (a stop tool beside it, sibling
  calls of the same reply), and a call reaches this server alone. The server refuses `obstacle_injection.enabled` at
  startup; run it on the relay config until the call carries its reply.
