# OfficeEnv environment server

This Environment Server runs an [OfficeEnv](../../resources_servers/officeenv/README.md) episode and ends it when its
task ends. It runs the [single-agent-turn protocol](../single_agent_turn/README.md) behind the flat `/run` contract,
as [`single_agent_turn_legacy`](../single_agent_turn_legacy/README.md) does: it seeds the Resources Server and Agent
Server sessions, activates the agent, closes its session, verifies, and cleans up, including on failure.

With `max_steps` set, it drives the agent one step at a time: each activation is one model call and the tool calls it
makes (the simple agent with `max_steps: 1`), with the conversation so far as its input. Between steps it reads
`GET /task_state` from the OfficeEnv Resources Server, and the episode ends before the next model call once the task
has ended (`complete_task` succeeded), as it does when the model answers without calling a tool, when a step does not
complete, or after `max_steps` steps. The episode's reply carries every step's output and their summed usage, so
verification and the length penalty see what one activation of the agent's own loop would give. The server refuses to
start when the bound agent's own `max_steps` is not 1.

Without `max_steps`, the agent runs its own loop in one activation, as `single_agent_turn_legacy` runs it.

The episode's model calls all go through its rollout's route, so token capture records them as one rollout. A crashed
episode starts over from its input (`checkpoint_mode: restart_only`). `/aggregate_metrics` is forwarded to the
Resources Server.

`resources_servers/officeenv/configs/officeenv.yaml` binds it to the simple agent with `max_steps: 50`.
