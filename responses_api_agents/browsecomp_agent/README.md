# Description

The BrowseComp research agent, paired with the [`browsecomp_advanced_harness`](../../resources_servers/browsecomp_advanced_harness) resources server. Its loop calls the harness's `search`, `browse` and `bash_command` tools. It compacts old tool results (`keep_rounds`) and resets the context past a token threshold (`context_reset_tokens`). With `progress: true` it keeps a progress board that survives the resets. With `rollout_timeout_s` set, it stops at a wall-clock limit.

It runs an episode in one of two ways:

- **Through an Environment Server** (`benchmarks/browsecomp/config_single_agent_turn.yaml`). `single_agent_turn_legacy` seeds the harness session and an agent session, calls `/v1/responses` once, closes the agent session, then verifies and closes the harness session.
  - The seed must grant exactly one direct HTTP tool access to the harness (`resources_tool_transports: [direct_http]`), and no sandbox or required MCP access. Any other seed is refused with a final 422.
  - Tool calls go to the grant's URL with its headers and cookies. The close returns the harness cookies, which the Environment Server sends to `/verify`.
  - `max_run_retries` retries an empty final answer (after `<think>` stripping) inside the one activation, keeping the harness session, as `/run` does.
  - A model call that fails with an HTTP error returns a `failed` response that names the status (`model_request_error`). The Environment Server's outcome policy then verifies 400, 413 and 422, records 408, 425, 429 and 5xx as non-terminal failures that a resume retries, and fails the episode on any other status. Other errors, such as a dropped connection, fail the activation with a 500, which is also recorded as a non-terminal failure.
- **Through its own `/run`** (`benchmarks/browsecomp/config.yaml`, behind the `legacy_agent` relay). `/run` seeds the harness, retries an empty answer, and verifies. A rollout stopped at `rollout_timeout_s` scores 0 without a verify. An error scores 0 with `agent_error`, except an infrastructure failure (5xx, a dropped connection, a timeout), which is set aside unscored for resume.

On both paths, a rollout stopped at `rollout_timeout_s` scores 0 without a judge call and reports `timed_out: 1`.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
