# Single-agent-turn environment server

This Environment Server runs one agent turn against one Resources Server.
It seeds both participant sessions, grants task-scoped tool and sandbox access to the agent, invokes `/v1/responses` once, verifies the response, and closes both sessions.

Use this server when one agent turn produces the response that Resources verifies.
Other interaction patterns should define their own task input, episode result, and Environment Server implementation.
