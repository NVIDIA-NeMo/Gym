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

With partial-rollout checkpointing on, the agent of an episode that grants `mcp` access or any tool server holds an MCP session token that names the episode's attempt, and the Resources Server refuses that token once a restore replaces the attempt. An episode that grants any tool server restarts from its input after a crash instead of continuing from a checkpoint. One that grants only `mcp` access continues from a checkpoint only if its agent borrows the sandbox the Resources seed lent (`sandbox_access`) and re-binds restored sessions to new tool grants (`rebinds_tool_accesses`, as the OpenCode agent does). Otherwise it restarts from its input.

A restore rebuilds a borrowed sandbox under a new ID. When the replacement attempt continues an episode whose agent had not closed yet, it first seeds the Resources and agent sessions again: the Resources Server returns the restored sandbox, and the agent re-binds its restored session to that sandbox and to the replacement attempt's tool grants, such as a new MCP session token. Seeding again is idempotent, so a checkpoint does not wait for it, and a crash during it seeds again after the next restore. An episode without a borrowed sandbox seeds nothing again.

## A sandbox for each attempt

This server can own a sandbox group for each attempt: it starts it, lends it to the agent and to the verifier, and stops it, so a harness that runs its loop in a sandbox, such as Simple Agent, never runs on a host. The Resources Server and the agent only borrow it. The lifecycle is the base `BaseEnvironmentServer`'s, configured through `AttemptSandboxEnvironmentServerConfig`. Set `sandbox_provider` to a named sandbox provider block and `sandbox_config` to the group's settings; set both or neither. Unset, this server starts no sandbox, as before, and fails an attempt whose Resources Server declares a group (see below) for good.

```yaml
environment_servers:
  single_agent_turn:
    sandbox_provider: attempt_sandboxes  # a top-level sandbox provider block
    sandbox_config:
      image: python:3.13-slim
      workdir: /workspace
      ttl_s: 25200  # above the episode timeout plus the cleanup timeout
      network: model_only
      model_server: policy_model  # the agent's model server
```

For OpenSandbox, the example requires packet-level egress enforcement on the deployment.
Set `networking.enforces_network_policy: true` in the OpenSandbox provider block once it enforces that policy.

Each attempt then runs in one sandbox:

| Step | What happens |
|------|--------------|
| Seed | After the Resources and tool server seeds, the server starts the sandbox through its `OwnedSandboxes` with the typed network rule, then seeds the agent with operate-scope `sandbox_access` to it. A Resources seed that lends a sandbox of its own fails the attempt (not retried): one attempt has one sandbox owner. |
| Agent turn | The agent borrows the sandbox and runs its turn in it. `model_only` lets the sandbox reach the model server's host and no other host. |
| Verify | After the agent closes, `/verify` also receives `ng_sandbox_access` (`NG_SANDBOX_ACCESS_KEY`), operate-scope access to the same sandbox. A verifier that grades in it declares the field on its own verify request model, excluded from dumps, and borrows it with `borrow()`. Every other verifier ignores it, and this server drops it from the verify reply, so it never reaches the stored result, even from a verifier that echoes its request. |
| Cleanup | The server stops the sandbox on every path, after the agent session closes. `ttl_s` reclaims one it could not stop. |

`sandbox_config` must name a network rule (`network`) and a lifetime (`ttl_s`), so no attempt starts with the provider's default egress or outlives a crashed server; `ttl_s` must exceed `default_episode_timeout_seconds` plus `cleanup_timeout_seconds`. A rule allows each named server's whole host, so `model_only` also reaches other servers on the model server's node. An episode that grants tools (`resources_tool_transports` or `tool_servers`) needs `network: model_and_tools` with those servers in `sandbox_config.tool_servers`, since a harness in the sandbox calls them from there.

At startup the server builds the provider and refuses one that cannot apply the rule, such as Docker for `model_only`, or that no other process can connect to, such as the `local` provider, which runs commands on this host. There is no host fallback: if the sandbox cannot start, or cannot be lent to the verifier, the attempt fails as a retryable infrastructure failure and records no reward. The agent must run in the sandbox it is lent: an agent that would run elsewhere refuses the seed. Simple Agent runs its loop in any sandbox it is lent. A checkpoint cannot rebuild a sandbox this server started, so with partial-rollout checkpointing on, such an episode starts over from its input after a crash.

Grading runs with the network rule the sandbox started with, and anything the agent left running keeps running until the sandbox stops: providers cannot change a running sandbox's egress yet.

### What the Resources Server declares

At seed, the Resources Server may declare what the group holds, as `sandbox_group` (`SandboxGroupSpec`) on its seed reply, and that its verifier executes nothing (`verifier_executes_nothing`). The group starts from `sandbox_config` (provider, network rule, `ttl_s`, resources) with the declared fields over it. A server that inherits `SimpleResourcesServer.seed_session` declares through class hooks; one that builds its own seed reply passes them on with `ResourcesSeedSessionResponse(..., **self.sandbox_declaration(body))`.

| Hook on the Resources Server | Declares |
|------|------|
| `sandbox_group(body)` | The task's `image`, `workdir`, `env`, text `files` delivered before anyone borrows the group, and `sidecars` that serve its environment tools. `image` and `workdir` fall back to `sandbox_config`. Sidecars need a provider that starts them. |
| `verifier_executes_nothing = True` | The verifier only reads the answer (rules, math-verify, a judge call): it executes nothing and calls no tools, so it may opt out of the sandbox. A verifier that runs model output, or passes it to `eval` or to sympy's `parse_expr`, leaves it unset. |
| `prepare_sandbox(body)` | Serves `/prepare_sandbox`: once the group has started, this server sends the seed's `task_data` and operate-scope access to the group, and the Resources Server prepares the task through `borrow()`, such as starting a sidecar's tool service, before the agent or the verifier borrows it. |

A seed reply lends a sandbox (`sandbox_access`) or declares one (`sandbox_group`), never both. An episode that holds a group this server starts is a restart: no checkpoint exports it, and the declared files and environment variables never reach a checkpoint.

### Opting out

Every task and every verifier runs in the group unless `sandbox_opt_out` says otherwise. Only what executes nothing may opt out, to save sandbox capacity on large QA runs:

```yaml
environment_servers:
  single_agent_turn:
    sandbox_provider: attempt_sandboxes
    sandbox_config: {...}
    sandbox_opt_out:
      task: true      # a pure QA task: the agent is lent no sandbox, so Simple Agent runs its loop in its own process
      verifier: false # the verifier still gets a sandbox: one of its own, started after the agent closes
```

| `task` | `verifier` | What starts |
|------|------|------|
| `false` | `false` | One group at seed: the agent borrows it, then `/verify`. |
| `true` | `false` | No sandbox during the agent's turn. A group for the verifier alone starts after the agent closes and is lent to `/verify`. |
| `false` | `true` | One group at seed for the agent. It stops once the agent closes, and `/verify` runs in the Resources Server. |
| `true` | `true` | Nothing. |

Every group stops once `/verify` replies, and again in the episode's cleanup if that stop failed. The opt-outs are checked at seed, and a refused one fails the attempt for good (a terminal seed failure), since every attempt would fail alike:

- `task` is refused for a task that declares tools (`responses_create_params.tools`), for an episode that grants tools (`resources_tool_transports` or `tool_servers`; also refused at startup), and for a group with sidecars. Pair it with an agent whose loop runs in its own process, such as Simple Agent: an agent that runs a harness in a sandbox of its own is not pure QA.
- `verifier` is refused unless the Resources Server declares `verifier_executes_nothing`, so a verifier that is not declared so, or whose server's seed reply drops the declaration, still gets a sandbox.

A group no attempt can start (no image or working directory, or sidecars on a provider that cannot start them) is also refused at seed, before the agent's turn. A group for the verifier alone that cannot start fails the attempt as a retryable infrastructure failure, with the agent's response kept as evidence, and records no reward.

The server supports `num_workers > 1`. Episode state and cleanup remain local to the worker handling that request. `max_concurrent_episodes` and queue admission are also enforced independently by each worker, so the deployment-wide concurrency limit is approximately `num_workers * max_concurrent_episodes`.
