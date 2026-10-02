# Single-agent-turn legacy environment server

This Environment Server runs the [single-agent-turn protocol](../single_agent_turn/README.md) behind the flat `/run` contract of an unmigrated agent.
It accepts either a flat legacy row or a native `SingleAgentTurnRequest`, runs the episode, and returns the flat verify-response row that rollout collection expects. A handled failure is returned as `_ng_failure_*` fields.

Use this server for an agent and Resources Server pairing that implements the session contracts but is still collected from flat rows. The row's `task_source` and `agent_ref`, when present, must match the configured Resources and Agent Servers. `/aggregate_metrics` is forwarded to the Resources Server.
