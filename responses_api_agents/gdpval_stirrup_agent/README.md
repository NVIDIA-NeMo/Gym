# GDPVal Stirrup agent (training)

RL rollouts for GDPVal with the **Stirrup / GDPVal-AA v2** harness, the same agent
the GDPVal evaluation campaigns run, on the sandbox fleet and judge path of
`resources_servers/gdpval_disaggregated`. It is a sibling of
`gdpval_sandboxed_agent` (OpenHands) and speaks the same `/run` protocol:
seed the resource session, attach to its OpenSandbox, run the agent, `/verify`,
always `/cleanup_session`. Data rows need no change (`agent_ref.name` stays
`gdpval_train` / `gdpval_val`); the training config selects the harness.

What the model sees (parity with `responses_api_agents/stirrup_agent` in
GDPVal mode):

* Stirrup's base system prompt (`{max_turns}` steps, finish-tool contract).
* The AA v2 user prompt (`prompts/gdpval_user_prompt.txt`, upstream text with
  `{working_dir}` instead of the hard-coded `/root`): `code_exec` semantics
  (independent commands, 5-minute cap), the installed stack, reference files
  listed relative to the working directory, `finish` with absolute paths,
  `abandon_task_finish` as last resort, the task wrapped as
  `Sector / Occupation / prompt`.
* Tools: `code_exec`, and with `web_tools_enabled` the eval harness's `web_search` /
  `fetch_web_page` pair (`exa_search.ExaToolProvider`: same names, parameter schemas and
  result XML as the Tavily provider, Exa as the search backend; keys from
  `web_search_api_keys_file` and/or the `EXA_API_KEYS` env var, rotated on auth/quota/5xx,
  never logged), plus the coercing `finish` and `abandon_task_finish` finish tools. Tool results are presented as user turns
  (`NeMoAgent(tool_response_as_user=True)`), serialized back as tool-role
  messages for the model.

What differs from the eval harness, deliberately:

* **Sandbox**: `opensandbox_provider.SandboxCodeExecToolProvider` runs
  `code_exec` in the resources server's OpenSandbox (`/workspace`, references
  under `/workspace/reference_files`) instead of a node-local Apptainer image.
  The image is the same `gdpval:aa-v2` build.
* **No summarization, no turn unwinding** (`context_summarization_cutoff=2.0`,
  `recover_from_context_overflow=False`): RL needs one contiguous conversation.
  Context exhaustion is reported as `agent_context_limit`; exhausting
  `agent_max_turns` as `agent_iteration_limit`. The resources server masks
  whichever reasons its `mask_policy_failures` lists.
* **Policy client** (`policy_client.GDPValPolicyClient`): per-call
  `max_completion_tokens` from the previous reply's `usage` plus the characters
  added since (no tokenizer on the agent server), typed context-limit errors,
  OpenAI-SDK read timeout + stateless retries (`model_read_timeout_s`,
  `model_read_retries`), and a request log used to export the trajectory even
  when the loop ends in an exception.
* **Deliverables**: the files named in `finish` are copied (flattened by
  basename) into `/workspace/deliverables`, which the resources server collects
  and judges. `abandon_task_finish` or a `finish` without files is a finished,
  zero-deliverable rollout (judged as such), not a masked one.
* **Token capture** is external: calls go to the rollout-specific model URL
  and the exported response reuses the served id of the last policy call, so
  the capture ledger attributes the terminal turn without heuristics.

Trace per rollout (collected with the sandbox outputs under `/trace/stirrup/`):
`summary.json` (finish kind, turns, commands, token totals, failure fields),
`finish_params.json`, `run_metadata.json`, `policy_calls.json` (per-call usage,
budget, latency), `messages.json` (the full chat history as sent).

Config keys of note (see `app.py:GDPValStirrupAgentConfig`): `agent_max_turns`
(200 in training, 250 in eval), `agent_timeout`, `model_read_timeout_s`,
`model_read_retries`, `max_context`, `max_output_tokens` (per-call cap),
`shell_timeout`, `working_dir`, `temperature` / `top_p` / `enable_thinking`.

Tests: `tests/test_gdpval_stirrup_agent.py` (units) and
`tests/test_agent_loop_offline.py` (the real Stirrup loop against a scripted
policy and an in-memory sandbox). Run with a venv that has `stirrup` and
`nemo_gym`:

```bash
PYTHONPATH=$PWD python -m pytest responses_api_agents/gdpval_stirrup_agent/tests -q
```
