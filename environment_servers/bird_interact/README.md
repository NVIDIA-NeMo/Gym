# BIRD-Interact Environment Server

Runs [BIRD-Interact](https://bird-interact.github.io/) episodes on `BaseUserSimulationEnvironmentServer`
(`nemo_gym/base_user_simulation_environment_server.py`). The `bird_interact` Resources Server holds each episode's
database sandbox, user simulator and grader; see `resources_servers/bird_interact/README.md`.

| `mode` | Turns | Agent access | `next_user_turn` |
|---|---|---|---|
| `c_interact` | Up to `max_turns`. The agent must resume its session across turns. | None | The reply is the text of the turn's last message, so the messages of a harness's earlier steps don't count. A reply with a ```` ```sql ```` or ```` ```sqlite ```` block is a submission (`/c_interact/submit`); any other reply is a question (`/c_interact/ask`). The next user message is the simulator's answer, the debug prompt or the follow-up question. |
| `a_interact` | One (`max_turns: 1`). | The Resources Server's tools, over each transport in `resources_tool_transports`: `direct_http` (the default) or `mcp` | `Stop("agent_finished")` |

The first turn's `responses_create_params` has the instructions and the user's ambiguous request. In c-Interact it
also has the schema, the column meanings and the visible knowledge base. In a-Interact over `direct_http` it has the
nine function tools instead. Over `mcp` alone it declares none: the agent lists the tools from the Resources Server,
and Codex refuses a request that declares tools. It is built from the context in the Resources Server's seed reply.

`mcp` needs `expose_tools_over_mcp` on the Resources Server. Without it, every episode fails at seed.

`finish_episode` calls the Resources Server's `/verify`. The result has the base fields (`reward`, `turns`,
`stop_reason`, agent observations) and BIRD's (`phases`, `questions_asked`, `coins_spent`, `finished`).

Stop reasons: `solved`, `failed` (the second attempt failed), `no_sql_submitted` (a question after the question
budget, after a submission, or in phase 2), `agent_finished` (a-Interact), and the base class's `max_turns` and
`interaction_timeout`.

## Harnesses

What each harness's Gym adapter accepts today, read from its session seed checks. `simple_agent`, in both modes, and
Codex, in a-Interact over `mcp`, have run BIRD with a model; the other rows are not results from runs.

| Harness | c-Interact | a-Interact |
|---|---|---|
| `simple_agent` | Yes | Yes, over `direct_http` |
| Codex, Pi, Hermes | No: the adapter has no multi-turn sessions | Yes, over `mcp`, in the harness's own sandbox |
| Stirrup | No: no multi-turn sessions | Yes, over `direct_http`, in its own sandbox. `code_exec: false` leaves out its shell. |
| OpenCode | Yes, in its own sandbox. `opencode_config.tools` turns off its built-in tools. | No: it refuses required tool grants |
| Claude Code, Kilo, OpenClaw | No: no multi-turn sessions | No: they refuse required tool grants, MCP included |

- **c-Interact needs multi-turn sessions.** With `max_turns` above 1, the seed asks the agent to resume its session
  across turns, and an adapter without that support refuses the seed.
- **Neither mode grants a sandbox.** BIRD's sandbox holds the database and the ground truth. A sandboxed harness
  runs in its own sandbox, from the `sandbox_provider` in the agent's config, with its built-in tools such as a
  shell. Those tools cannot reach the database, but BIRD-Interact has no such tools. Turn them off where the adapter
  allows it, so scores stay comparable with the paper's.
- **Over `mcp`, the harness calls the Resources Server from inside its sandbox**, so the Resources Server's address
  must be reachable from there, as the model server's already must be. To run a-Interact on Codex or Pi, point
  `agent_server` at the harness and set `resources_tool_transports: [mcp]`. `benchmarks/gdpval/codex.yaml` and
  `benchmarks/gdpval/pi.yaml` show the agent settings, such as the version and the model's limits.
- **With the Docker provider**, start the servers with `+default_host` set to an address the container can reach,
  such as the `docker0` bridge address. If the host's firewall drops traffic from the bridge to host ports, Codex
  times out on the MCP handshake; give the harness's sandbox provider `network: host`. BIRD's database sandbox needs
  neither, because the Resources Server runs commands in it and it calls nothing.

The gaps are in the adapters, not in BIRD: multi-turn sessions in Codex, Pi, Hermes, Stirrup, Claude Code, Kilo and
OpenClaw for c-Interact, and tool grants in Claude Code, Kilo, OpenClaw and OpenCode for a-Interact.

## Configuration

The configuration is in `resources_servers/bird_interact/configs/bird_interact.yaml`. It defines one Environment
Server for each mode and routes the tasksets `bird_interact:c_interact_example` and
`bird_interact:a_interact_example` to them.
