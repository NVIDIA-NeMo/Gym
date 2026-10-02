# Claude Code Sandboxed Agent

Runs the Claude Code CLI (`claude -p`) inside the task's own sandbox, the way
`opencode_sandboxed_agent` runs OpenCode: the resources server seeds the sandbox, this agent
attaches to it, runs one command under `sandbox_timeout`, reads the transcript the CLI left behind,
calls `/verify` and stops the sandbox. Gym stays on the host; the sandbox needs only the binary.

Claude Code talks to the Gym model server's `/v1/messages` route. Every Gym model server serves
that route, so the agent works with any of them. With `token_id_capture: true` (the default) each
call carries the rollout's training-capture URL prefix, so training gets the exact sampled tokens.

```bash
gym env start \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    --config resources_servers/swe_rebench/configs/swe_rebench_claude_code.yaml
```

Each SWE resources server that has an OpenCode wiring config (`<server>_opencode.yaml`) has a
Claude Code one next to it (`<server>_claude_code.yaml`) with the agent instance
`<server>_claude_code_sandboxed_agent` and the environment server
`<server>_claude_code_environment_server` that fronts it; rows route to it with
`agent_ref.name = <server>_claude_code_sandboxed_agent`. A run may load both wiring configs.

## Installing Claude Code in the sandbox

Stage the release binary where every sandbox can read it, for example on a read-only bucket
mount, and set `remote_claude_code_binary_path` (and `remote_claude_code_musl_binary_path` for
musl-based images). The agent copies it to a writable directory, makes it executable and checks
that it reports `claude_code_version`. Without a staged binary the official installer runs inside
the sandbox, which needs network access.

```bash
VERSION=2.1.287
curl -fsSL -o claude https://downloads.claude.ai/claude-code-releases/$VERSION/linux-x64/claude
curl -fsSL -o manifest.json https://downloads.claude.ai/claude-code-releases/$VERSION/manifest.json
grep -q "$(sha256sum claude | cut -d' ' -f1)" manifest.json && echo MATCH || echo MISMATCH
```

## What the agent sets

- A fresh `CLAUDE_CONFIG_DIR`, binary and transcript directory per rollout under
  `remote_work_dir` (`/tmp`), outside the task repository; stdin closed.
- The full harness: `bare: false`. `--bare` is Claude Code's minimal mode, with a one-line system
  prompt and only the Bash, Read and Edit tools.
- `--setting-sources user`: only the rollout's empty user settings and `--settings` load, so a task
  repository's `.claude/` settings and hooks cannot change the run.
- No auto-memory (`CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`): each rollout starts from an empty config
  directory, and notes written mid-session would change the context of later calls.
- `--dangerously-skip-permissions` with `IS_SANDBOX=1`.
- `--disallowedTools` for what does not fit a one-shot run in a task sandbox: web access and skills
  (as for the OpenCode agent), worktrees (they would move the edits out of the repository the
  verifier diffs), scheduled prompts, wakeups, workflows, agent messaging and the code-review report
  tool. Agent (subagents), Bash, Read, Edit, Write, NotebookEdit and TaskStop stay.
- Every model role (`ANTHROPIC_MODEL`, the opus/sonnet/haiku defaults, subagents) points at `model`.
- No update checks, telemetry, error reports, experimental betas or per-request attribution text.
  The attribution text would change the system prompt on every call and break capture's call chain.
- `CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS=1800000` and `API_TIMEOUT_MS=3600000`. A Gym model server
  sends a streamed reply's headers only after generation finishes, and Claude Code's default wait
  of 3 to 5 minutes for headers would abort and retry long calls.
- `auto_compact: false` sets `DISABLE_COMPACT`, `DISABLE_AUTO_COMPACT` and the `autoCompactEnabled`
  and `precomputeCompactionEnabled` settings to off. Compaction rewrites the conversation, which
  token capture cannot follow; training must turn it off. With it off, Claude Code sends requests
  until the model server rejects one, like OpenCode with compaction off.
- The CLI runs under `timeout` until `harness_timeout_margin_s` before `sandbox_timeout`, so it
  cannot keep editing the repository or calling the model while the verifier runs.

`claude_code_env` and `claude_code_settings` are layered over all of the above.

## Output

The stream-json output is downloaded to `results/<session>/stream.jsonl` (or under
`artifacts_dir`) and converted to Responses items: the main conversation's reasoning, text, tool
calls and tool results. Subagent traffic and Claude Code's own error notices are left out. A
`generation.json` receipt with the response and the execution fields is written before grading.
When observations are collected, the session transcripts are archived and read with
`claude_code_agent.observability`, which also reports any compaction.

The `/run` result carries, besides the verifier's fields:

- `claude_code_finished`, `claude_code_exit_code`, `claude_code_error_type`, `claude_code_failed`
  (exec error, nonzero exit, missing transcript or error result), `claude_code_export_found`,
  `claude_code_results_fpath`, `claude_code_run_stdout`, `claude_code_run_stderr`.
- How the session ended, from its final `result` event: `claude_code_result_success`,
  `claude_code_result_error`, `claude_code_result_missing` (the deadline or a crash cut it off),
  `claude_code_context_overflow` (the last request no longer fit the context), and
  `claude_code_duration_s`.
- What it did, main conversation and subagents together: `claude_code_main_turns`,
  `claude_code_subagent_turns`, `claude_code_subagent_calls`, `claude_code_tool_calls`,
  `claude_code_tool_errors` (calls Claude Code refused) and `claude_code_unknown_tool_calls`
  (refused for naming a tool that does not exist).

With `execution_failure_reward_zero: true` a failed run is graded as an empty response instead of
whatever patch it left behind; the verifier must score an empty response as an unmasked zero.
