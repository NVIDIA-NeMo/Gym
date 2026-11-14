# Agent servers

Each directory here is an agent server built on `nemo_gym.agents`, and often on `nemo_gym.sandbox`. The contract they
implement, and the tests to run, are in [`nemo_gym/agents/AGENTS.md`](../nemo_gym/agents/AGENTS.md). The harness guide,
[Add or update a harness](../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx), covers session
endpoints, sandbox and tool contracts, and output conversion; the `add-harness` skill walks through the steps.

What agent server changes commonly get wrong:

- Sending model calls to a URL without the rollout prefix, including a harness's auxiliary calls such as summaries,
  compaction and subagents. Those calls are not captured.
- Dropping the model server's cookies between calls of one rollout.
- Overriding `seed_agent_session` or `close_agent_session` instead of implementing `_seed_agent_session_state` and
  `_close_agent_session_state`.
- Answering a seed the agent can never serve with a 500 instead of `AgentSeedRejectedError`, so every retry creates
  a new sandbox.
- Returning a failed response without calling `report_outcome`, so the environment server cannot tell why it failed.
- Inheriting `calls_model_from_sandbox` or `ray_enabled` instead of declaring them on the entrypoint's class.
- Using `httpx.AsyncClient` for Gym calls instead of `ServerClient` or `nemo_gym.client.request`.
- Ignoring a request or config field silently instead of applying it, rejecting it, or documenting that it is ignored.
