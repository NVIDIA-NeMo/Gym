# RoboLab Agent

See [benchmark documentation](../../benchmarks/robolab/README.md) for runtime
inputs, grading semantics, and evaluation commands.

`RoboLabAgent` (`SimpleResponsesAPIAgent`) owns the Gym agent-session lifecycle
(`_seed_agent_session_state` / `_close_agent_session_state`) for
`RoboLabEpisodeBinding` while the Bridge OpenCode harness runs visual
manipulation turns through the environment server.
