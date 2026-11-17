# Single-agent-turn environment server

This Environment Server runs one agent turn against one Resources Server.
It seeds the Resources and agent sessions, plus a session on each configured tool server, grants configured task-scoped tool access and any sandbox access returned by Resources to the agent, invokes `/v1/responses` once, verifies the response, and closes every session it seeded.

`resources_server` is required and belongs to the Environment Server. Configuration resolution follows an unambiguous top-level `_inherit_from` rename of that Resources Server. Copies and multiple destinations require an explicit binding in the child Environment Server; the agent's own binding does not choose it. See the [configuration migration guide](../../fern/versions/latest/pages/troubleshooting/configuration.mdx#move-a-simple_agent-off-the-relay).

Use this server when one agent turn produces the response that Resources verifies.
Other interaction patterns should define their own task input, episode result, and Environment Server implementation.

`/verify` receives the Resources Server's own flat verify body, the same one an Agent's `/run` sends: the task's `task_data` fields, `responses_create_params`, and the agent's `response`. The request carries the Resources session cookie from seeding, which identifies the episode, so an existing Resources Server verifies these episodes without changes. The verify response becomes the episode result, with the agent's session observations added as `ng_agent_observations`.

Set `forward_agent_evidence: true` when the Resources verifier needs execution evidence, such as tool-use reward shaping. It defaults to `false`. When enabled, `/verify` also receives the available `ng_trajectory` and `ng_agent_observations`. This setting only controls the verifier request; saved episode results retain that evidence with either value.

```yaml
environment_servers:
  single_agent_turn:
    forward_agent_evidence: true
```

Set `skip_verification: true` to return `skip_verification_reward` (default `0.0`) without calling `/verify`, for example for data generation or infrastructure-only runs. The result keeps the flat shape: the verify body, the fixed `reward`, and `verification_skipped: true`, so anything reading the result can tell it from a measured reward. Every session is still closed, and `/aggregate_metrics` skips the Resources Server and reports performance metrics only: `perf_summary`, per-repeat token statistics, and their across-repeat statistics, with confidence intervals for two or more repeats, computed from each rollout's `ng_perf`. Quality metrics such as `mean/reward` are omitted. When the Environment Server leaves either field unset, it uses the bound agent's value, which Gym fills from the global `skip_verification` and `skip_verification_reward` flags. A pairing moved off the `legacy_agent` relay therefore keeps the setting its agent's `/run` honored. This applies to every bound agent, so the global flag also skips verification for agents whose own `/run` never read it:

```yaml
environment_servers:
  single_agent_turn:
    skip_verification: true
    skip_verification_reward: 1.0
```

Resources tool access is opt-in. Set `resources_tool_transports` on the Environment Server deployment to any combination of `direct_http` and `mcp`. Leave it empty when the agent should receive no Resources tools:

```yaml
environment_servers:
  single_agent_turn:
    resources_tool_transports:
      - direct_http
      - mcp
```

`direct_http` grants the Agent Server scoped HTTP access to the Resources Server. `mcp` requires the Resources seed response to provide HTTP MCP connection metadata. Selecting neither transport grants no tools from the episode's Resources Server; `tool_servers` can still add MCP grants. When neither source adds a grant, the Environment Server sends `tool_accesses=[]` when the agent session is seeded.

To grant tools from other Resources servers as well, such as a search server, list them in `tool_servers`. Each episode seeds its own session on every listed server, with the episode's task data, grants that server's MCP tools as a required `MCPToolAccess`, and closes the session at cleanup. A listed server must be a Resources server that exposes its tools over MCP (`expose_tools_over_mcp: true`), and it cannot be the episode's own Resources server, whose tools `resources_tool_transports: [mcp]` grants. The Environment Server does not start otherwise. Unset, `tool_servers` falls back to the bound agent's own `tool_servers`, so an agent moved off the `legacy_agent` relay keeps the tool servers its `/run` seeded; set it to `[]` to grant none:

```yaml
environment_servers:
  single_agent_turn:
    tool_servers:
      - type: resources_servers
        name: tavily
```

When the tools served to an episode come from several MCP endpoints, each endpoint lists itself under its server's `mcp_server_name`, or else its server's name, and the harness names the endpoint's tools after that name in its own way: Pi and mini-SWE as `mcp__<name>__<tool>`, OpenCode as `<name>_<tool>` (each agent's README says how). The Environment Server does not start when two granted endpoints list themselves under one name, since their tools would collide in every harness. Distinct names can still collide in a harness's own naming, which that harness's agent refuses: the OpenCode agent refuses `company` beside `company_files`, whose tools `files_list` and `list` would both be `company_files_list`.

### Driving the agent one step at a time

By default the agent runs its own loop in the episode's one activation. Set `max_episode_steps` to drive it one step per activation instead (one model call and the tool calls it makes), up to that many steps in the episode, and `end_on_task_state` to end the episode once the Resources Server reports that the task ended. This suits a Resources Server whose task ends through its state, such as a task row that several operations can close, rather than through one tool's call:

```yaml
environment_servers:
  single_agent_turn:
    resources_tool_transports: [direct_http]
    max_episode_steps: 50
    end_on_task_state: true
```

Each activation sends the whole conversation, with every step before it, numbers the step in `x-ng-agent-activation`, and carries the episode's step limit in the `x-ng-agent-episode-step-limit` header. The agent then takes one step, whatever its own step limit, records one invocation for the episode, as its own loop would, and counts a step as incomplete only at the episode's limit. The simple agent also keeps the model server's cookies from step to step, so the episode keeps one model server session. The episode ends after the first step in which the model called no tool or that did not complete (a reply without a status counts as completed), and with `end_on_task_state` after the step that ended the task: between steps, though not after the last, the server reads `GET /task_state` on the episode's Resources session, which answers `ResourcesTaskState` (`{"task_ended": ...}`). A 404 whose body names `failure_kind: session_lost` (`ResourcesSessionLostError`) counts as the task's end: the episode goes on to `/verify`, where the lost session gets a masked sample, as when a tool call finds the session gone. Any other failed read fails the episode. The reply carries every step's output and their summed usage, as one activation's would, and each step's failure report is kept. When every step called a tool and the episode ran out of steps, only the reply's metadata says so (`ng_termination_reason: max_episode_steps`): its status stays the last step's. A failure after a step completed keeps the steps before it as the failure's `partial_response`.

The agent must be of a type that takes one step per such activation, listed in `STEP_DRIVEN_AGENT_TYPES` (`nemo_gym.config`; today the simple agent), since keeping an agent session is not enough, and it must run the tools it calls. The server checks that at startup, since a step taken after the task ended, within one activation, would count. A Resources Server whose pairing needs this agent can set `allowed_agents`, so that `gym env resolve` refuses another `--agent-type` template composed in place of its agent before anything starts. `single_agent_turn_legacy` takes these settings too, for flat rows as well as episode requests. The step loop records no checkpoint boundary between steps, so its episodes take part in checkpoints as restarts.

### Stopping an agent's own loop when its task ends

An agent that runs its own loop, such as a harness in a sandbox, cannot be stopped between steps. With `end_on_task_state` and no `max_episode_steps`, the server watches the episode's Resources session while the agent runs, and stops the run once the task has ended:

```yaml
environment_servers:
  single_agent_turn:
    resources_tool_transports: [mcp]
    end_on_task_state: true
```

Before the agent runs, the server reads `GET /task_state?watch=true`, and from then on the Resources Server holds its answer to the call that ends the task. While the agent runs, the server reads the state again, each read waiting up to 30 s for the end (`wait_s`). Once the task has ended, it stops the run with `POST /rollouts/{rollout_id}/stop` and `{"reason": "task_ended", "completed": true}`, waits for the activation to answer (at most 120 s), and only then releases the held call (`POST /task_state/release`). The harness never reads that call's answer while its run could go on, so no model call follows the call that ended the task. The stopped run answers as completed, and its metadata's `ng_rollout_stop` says `"completed": true`. A run that ends first is not stopped. A lost session (404 naming `failure_kind: session_lost`) counts as the task's end; any other failed read, or a failed stop, fails the episode.

The agent must be of a type whose run `POST /rollouts/{rollout_id}/stop` can end in flight, listed in `ROLLOUT_STOP_AGENT_TYPES` (`nemo_gym.config`); the server checks that at startup. The stop writes a marker the harness's supervisor watches, so the sandbox provider must let an upload run while the harness's command does (`uploads_independent_of_execs`): Docker and the local provider do, and OpenSandbox does with `operations.background_exec: true`, which Gym's OpenSandbox config sets.

With partial-rollout checkpointing on, the agent of an episode that grants `mcp` access or any tool server holds an MCP session token that names the episode's attempt, and the Resources Server refuses that token once a restore replaces the attempt. An episode that grants any tool server restarts from its input after a crash instead of continuing from a checkpoint. One that grants only `mcp` access continues from a checkpoint only if its agent borrows the sandbox the Resources seed lent (`sandbox_access`) and re-binds restored sessions to new tool grants (`rebinds_tool_accesses`, as the OpenCode agent does). Otherwise it restarts from its input.

A restore rebuilds a borrowed sandbox under a new ID. When the replacement attempt continues an episode whose agent had not closed yet, it first seeds the Resources and agent sessions again: the Resources Server returns the restored sandbox, and the agent re-binds its restored session to that sandbox and to the replacement attempt's tool grants, such as a new MCP session token. Seeding again is idempotent, so a checkpoint does not wait for it, and a crash during it seeds again after the next restore. An episode without a borrowed sandbox seeds nothing again.

The server supports `num_workers > 1`. Episode state and cleanup remain local to the worker handling that request. `max_concurrent_episodes` and queue admission are also enforced independently by each worker, so the deployment-wide concurrency limit is approximately `num_workers * max_concurrent_episodes`.
