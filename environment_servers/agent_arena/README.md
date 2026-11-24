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
   the ones the scoring accepts (the row's, or Cline's for a Cline row), so a call to another tool gets 404 here and
   `nonexistent_tool` at scoring.
3. **Seed** an agent session with that grant, and **invoke** the agent with the row's `responses_create_params`. Their
   metadata name the row's stop tools and tool-call protocol, which the agent takes out before it calls the model.
4. **Close** the agent session. A failed agent reply goes through `agent_outcome_policy` (retry, fail or score).
5. **Score** the turn in process: masking, code gates, judges. The result carries the same fields as the resources
   server's `/verify` response, and `/aggregate_metrics` computes the same metrics (plus Gym's performance metrics).
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
| `agent_server` | required | The agent that plays the turn: `agentic_turn_simulation_agent`, or one that reads the row's harness from the request's metadata and calls the tools through its direct HTTP grant the same way. |
| `resources_server` | required | The `agentic_turn_simulation` instance whose tool simulator, judge and scoring settings the episode uses. No call goes to it during an episode; `gym eval reverify` re-scores saved rows through it, with the same settings. |
| `agent_outcome_policy` | Gym's default | What a failed agent reply does: retry the episode, fail it, or score the turn. |
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
- The resources server may run several workers; this server routes each tool call to the worker that runs its episode.

On all 433 AgentArena validation rows, two rollouts each, with one checkpoint as policy, simulator and judge, the mean
measured reward was 0.952 here and 0.949 on the relay: a paired difference by task of +0.0013 (standard error 0.0109).

## Limits

- **Restart-only checkpointing.** With rollout checkpointing on, a crash restarts the episode from its input; a
  half-played turn is never continued.
- **No sandbox.** A call is answered from the model of the agent's files or by a model call, so no episode starts a
  sandbox and `sandbox_provider` is `local`. The server logs the base class's warning about `local` at startup.
- **No obstacle injection.** An obstacle's decision needs the reply a call belongs to (a stop tool beside it, sibling
  calls of the same reply), and a call reaches this server alone. The server refuses `obstacle_injection.enabled` at
  startup; run it on the relay config until the call carries its reply.
