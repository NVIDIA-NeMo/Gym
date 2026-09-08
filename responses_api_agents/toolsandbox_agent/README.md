# Description

Agent harness for the [ToolSandbox](https://github.com/apple/ToolSandbox)
multi-turn tool-use benchmark. It is a thin fork of `aviary_agent`: ToolSandbox
is a conversation, not a pure tool loop, so when the policy model replies in
natural language (no tool calls) that reply is forwarded to the **user
simulator** and the episode continues, rather than ending the rollout on a
no-tool-call turn.

The harness seeds a session, drives the policy model turn by turn against
`resources_servers/toolsandbox` (`/step` until `done`), then closes and verifies
the episode. See `resources_servers/toolsandbox/README.md` for architecture,
setup, and run instructions.

# Licensing information
Code: Apache 2.0
Data: CC-BY-NC-4.0 (apple/ToolSandbox scenarios)

Dependencies
- nemo_gym: Apache 2.0
- tenacity: Apache 2.0
