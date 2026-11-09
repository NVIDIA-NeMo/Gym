# ClawEval conversation protocol

This Environment Server coordinates ClawEval Resources and a configurable
Agent Server through typed tasks and agent sessions. It owns user-turn
orchestration, not the assistant's model/tool loop. See the
[Resources guide](../../resources_servers/claweval/README.md) for configuration,
lifecycle, evidence and capability limits.

`tool_transport` selects `mcp` or `direct_http`. `turn_sessions: continue`
requires standard harness continuation and sends indexed user deltas;
`separate` opens a session per user turn and sends complete history.
Neither path calls agent `/run`.
