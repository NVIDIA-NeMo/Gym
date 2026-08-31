# Description

`visgym_agent` is the Path B action-transport agent for the VisGym resources
server (see [Doc 2](../../../../../docs/design-docs/doc-2-nemo-gym-game-agent-action-transport.md)).
It runs a multi-turn rollout against a Responses-API model server and the VisGym
`/step` endpoint, extracting the model's action from the **last** `\boxed{...}`
token in the assistant's plain text rather than from a tool call envelope.

The agent is the side-by-side counterpart of `aviary_agent` (Path A). Both
paths share VisGym's unified `/step` schema (`tool_calls` for Path A,
`action_string` for Path B) and the `VisGymEnvStateEasyInputMessage` channel for
env metadata.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
- tenacity: Apache 2.0
