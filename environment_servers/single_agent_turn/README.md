# Single-agent-turn environment server

This Environment Server runs one agent turn against one Resources Server.
It seeds both participant sessions, grants configured task-scoped tool access and any sandbox access returned by Resources to the agent, invokes `/v1/responses` once, verifies the response, and closes both sessions.

Use this server when one agent turn produces the response that Resources verifies.
Other interaction patterns should define their own task input, episode result, and Environment Server implementation.

`/verify` receives the Resources Server's own flat verify body, the same one an Agent's `/run` sends: the task's `task_data` fields, `responses_create_params`, and the agent's `response`. The request carries the Resources session cookie from seeding, which identifies the episode, so an existing Resources Server verifies these episodes without changes. The verify response becomes the episode result, with the agent's session observations added as `ng_agent_observations`.

Resources tool access is opt-in. Set `resources_tool_transports` on the Environment Server deployment to any combination of `direct_http` and `mcp`. Leave it empty when the agent should receive no Resources tools:

```yaml
environment_servers:
  single_agent_turn:
    resources_tool_transports:
      - direct_http
      - mcp
```

`direct_http` grants the Agent Server scoped HTTP access to the Resources Server. `mcp` requires the Resources seed response to provide HTTP MCP connection metadata. Selecting neither transport sends `tool_accesses=[]` when the agent session is seeded.

The server supports `num_workers > 1`. Episode state and cleanup remain local to the worker handling that request. `max_concurrent_episodes` and queue admission are also enforced independently by each worker, so the deployment-wide concurrency limit is approximately `num_workers * max_concurrent_episodes`.

With `observability_enabled: true`, the Environment Server derives `response.usage` from the shared Model Server capture before verification. This applies to every harness using this lifecycle, including the legacy flat-row adapter. It counts exchanges from Agent session creation through Agent close, including delegated work, compaction and retries that return usage. Resources seeding, earlier calls under the same rollout ID, and later verifier/judge calls are excluded. Native CLI counters and response IDs are not needed.

Configure an absolute `model_call_capture_dir` shared by the Model and Environment Servers. Submitted evaluations already enable capture and derive this directory unless explicitly disabled. Direct local deployments can set:

```yaml
observability_enabled: true
model_call_capture_dir: /absolute/path/to/run/model-calls
```

Input counts include cached input; output counts include reasoning. Anthropic cache-read and cache-write tokens are folded into input counts. Optional details remain `null` when unavailable. Missing counts on any exchange, truncated streams, or unavailable/damaged capture yield `usage: null`, not zero or a partial sum. The shared count covers only traffic routed through Gym Model Servers with the episode's rollout prefix; it is not a provider billing audit. Custom Environment Servers can use `nemo_gym.model_usage.ModelUsageCapture` around their own agent lifecycle.

When observability is disabled, or when calling an Agent Server directly, usage remains whatever that harness reports. Direct calls do not get this Environment Server accounting.
