# RoboDojo Agent

See [benchmark documentation](../../benchmarks/robodojo/README.md) for runtime
inputs, grading semantics, and evaluation commands.

`RoboDojoAgent` (`SimpleResponsesAPIAgent`) owns the Gym agent-session
lifecycle (`_seed_agent_session_state` / `_close_agent_session_state`) for
`RoboDojoEpisodeBinding` while EmberBridge's OpenCode harness executes visual
manipulation turns through the environment server's OpenCode relay.
