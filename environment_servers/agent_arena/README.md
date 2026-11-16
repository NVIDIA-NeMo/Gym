# AgentArena Environment Server

Runs each row of the [`agentic_turn_simulation`](../../resources_servers/agentic_turn_simulation/README.md)
environment (AgentArena and its training rows) as one episode, with no Resources Server in the episode. The policy
plays one whole agent turn through
[`agentic_turn_simulation_agent`](../../responses_api_agents/agentic_turn_simulation_agent/README.md), against tools
this server simulates, and this server scores the turn.

## One episode

1. **Set up** the turn from the whole row: pick its harness, replay its history into the model of the agent's files,
   prepare the tool simulator's prompt, arm the row's obstacle. The turn's session stays in this server's memory under
   a secret drawn for the episode.
2. **Seed** an agent session whose direct HTTP grant points at this server, with the secret in its headers.
3. **Invoke** the agent with the row's `responses_create_params`. The agent asks `/harness` for the row's stop tools
   and tool-call protocol, and sends each tool call to `/simulate_tool_call`, both through the grant. A call without a
   live episode's secret gets 403.
4. **Close** the agent session. A failed agent reply goes through `agent_outcome_policy` (retry, fail or score).
5. **Score** the turn in process: masking, code gates, judges. The result carries the same fields as the resources
   server's `/verify` response, and `/aggregate_metrics` computes the same metrics (plus Gym's performance metrics).
6. **Release** the turn's session in the episode's cleanup.

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
```

| Setting | Default | What it does |
| --- | --- | --- |
| `agent_server` | required | The agent that plays the turn: `agentic_turn_simulation_agent`, or one that drives `/harness` and `/simulate_tool_call` the same way. |
| `resources_server` | required | The `agentic_turn_simulation` instance whose tool simulator, judge and scoring settings the episode uses. No call goes to it during an episode; `gym eval reverify` re-scores saved rows through it, with the same settings. |
| `agent_outcome_policy` | Gym's default | What a failed agent reply does: retry the episode, fail it, or score the turn. |
| `default_episode_timeout_seconds` | `21600` | The episode's wall-clock bound. The agent's own `max_time_seconds` usually ends the turn first. |
| `cleanup_timeout_seconds` | `180` | The bound of each cleanup step (agent close, session release). |

`benchmarks/agent_arena/config.yaml` and the environment's own
`resources_servers/agentic_turn_simulation/configs/agentic_turn_simulation.yaml` use this server. Their
`*_legacy_agent.yaml` twins keep the deprecated `legacy_agent` relay.

## Limits

- **One worker.** A turn's session lives in the worker that runs its episode, and the agent's tool calls must reach
  that worker, so the server refuses `num_workers` > 1.
- **Restart-only checkpointing.** With rollout checkpointing on, a crash restarts the episode from its input; a
  half-played turn is never continued.
