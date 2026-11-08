# Single-agent-turn legacy environment server

This Environment Server runs the [single-agent-turn protocol](../single_agent_turn/README.md) behind the flat `/run` contract of an unmigrated agent.
It accepts either a flat legacy row or a `SingleAgentTurnRequest` episode request, runs the episode, and returns the flat verify-response row that rollout collection expects. A handled failure is returned as `_ng_failure_*` fields.
A flat row's `_ng_task_index`, `_ng_rollout_index`, `_ng_group_id` and `_ng_group_attempt` become the episode's `rollout_coordinates`, which reach `/verify` as described in the [single-agent-turn README](../single_agent_turn/README.md).

Use this server for an agent and Resources Server pairing that implements the session contracts but is still collected from flat rows. The row's `task_source`, when present, must name the configured Resources Server, the configured Agent Server, or this Environment Server; collation sets it to whichever instance declares the dataset. The row's `agent_ref`, when present, must match the configured Agent Server. `/aggregate_metrics` is forwarded to the Resources Server.
