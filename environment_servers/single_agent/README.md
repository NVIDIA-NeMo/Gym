# Single-agent environment server

This Environment Server runs one Responses API agent against one Resources Server.
It seeds both participant sessions, grants task-scoped tool and sandbox access to the agent, invokes the agent once, verifies the response, and closes both sessions.

Use this server for tasksets whose input contract is `nemo_gym.single_agent.v1`.
Other interaction patterns should define their own task input, episode result, and Environment Server implementation.
