# partner_if Environment Server

Runs one benchmark whose tasks use different harnesses, such as [partner_if](../../benchmarks/partner_if/README.md),
as one dataset with one score.

- `task_routes` names the Environment Server of each task, by `verifier_metadata.partner_if.task_id`. A task routed
  to this server runs here, through the [single-agent-turn protocol](../single_agent_turn/README.md) with the
  configured `agent_server` and `resources_server`, as in
  [`single_agent_turn_legacy`](../single_agent_turn_legacy/README.md). Any other task's flat row goes to the
  `/run` of the server it names, which owns that episode's sessions, sandboxes and grading.
- It takes flat rows only: a typed episode request, or a task with no route or no task ID, is refused with HTTP 422
  before any episode starts. The row's
  `agent_ref` and `task_source` must name this server's pairing; they are rewritten to the routed server's agent,
  and every other field passes unchanged. A routed server's HTTP failure and its body come back unchanged.
- A route must name a `single_agent_turn_legacy` or `legacy_agent` Environment Server, whose `/run` takes the flat
  row, that binds one agent.
- This server's episode deadline and admission limits (`default_episode_timeout_seconds`, `max_concurrent_episodes`,
  `queue_timeout_seconds`) apply to the tasks it runs. A routed task is bounded by its own server's limits.
- `/aggregate_metrics` behaves as in `single_agent_turn`: `resources_server` aggregates the benchmark metrics of
  every task, wherever it ran, so all tasks share one score.
- For a routed simple_agent episode, the tool executions in the agent's reply are added to its agent observations,
  so rollout collection can build `ng_trajectory.tool_calls`.
- For a route to `legacy_agent` with `gymnasium_agent` over partner_if's recorded-context Resources Server, the
  Resources Server's terminal report is checked against the row (checksum, actor, task data, grade fields) and
  becomes the result row, keeping terminal tool outputs, usage and grades; a report that fails a check is HTTP 502.

`configs/partner_if.yaml` runs partner_if's text and mock-tool examples (`data/example.jsonl`), each in its own
environment:

```bash
gym env start --config environment_servers/partner_if/configs/partner_if.yaml \
    +policy_base_url=... +policy_api_key=... +policy_model_name=... \
    +judge_base_url=... +judge_api_key=... +judge_model_name=...

gym eval run --no-serve --agent partner_if_simple_agent \
    --input environment_servers/partner_if/data/example.jsonl \
    --output partner_if_rollouts.jsonl
```
