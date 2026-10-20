# GDPVal sandboxed agent

The implementation name follows `opencode_sandboxed_agent`. This adapter runs
the pinned OpenHands CodeAct harness inside the sandbox supplied by
`resources_servers/gdpval_disaggregated`.

The `/run` endpoint seeds the resource session, attaches to its sandbox, installs
the staged runtime, uploads policy configuration and the controller entrypoint,
executes OpenHands, exports the trajectory, and calls resource verification.
It forwards resource-session cookies and always attempts session cleanup.

`remote_openhands_install_script_path` and `remote_openhands_archive_path`
parallel SWE's staged installer/binary configuration. The controller uses the
rollout-specific Gym model URL so training token capture remains external.

The instance names `gdpval_train` and `gdpval_val` remain compatible with existing
dataset rows. The ECR task image and S3 runtime paths are unchanged by this split.

## Context compaction (NeMo RL multi-trace training)

Default off. `context_summarization_cutoff: <fraction>` (same semantics as the Stirrup
agent's field: fraction of `max_context`) turns on the GDPVal condenser inside the sandbox
(`compaction_condenser.py`, uploaded with the runner). When the last reply's
`prompt + completion` tokens reach `cutoff * max_context`, OpenHands summarizes everything
after the task prompt in one policy call (the stock `LLMSummarizingCondenser` prompt) and
continues from `system prompt + task + summary`; nothing else is kept, so no stale assistant
tokens are re-sent into the new context. The summary call goes through the same
`DirectModelClient` as agent calls (rollout-prefixed URL, retries, `model-requests.jsonl`,
`/trace/completions` log) and its log entry carries `gdpval_call_kind = "compaction_summary"`.
`/trace/compactions.jsonl` records each compaction (trigger usage, forgotten range, size).

After the rollout, `compaction.py` splits the completion logs into context segments and
`/run` adds `responses`: one Gym response per segment, output items covering only the
messages after the segment's first request, the echoed token fields on every assistant item,
and NeMo RL's compaction-trace metadata (`training_mode=compaction`, `session_id` =
`root_session_id` = the task id, `segment_index`, `segment_boundary_reason` ∈
`"" | "compaction" | "post_compaction"`, `request_ids`, `policy_calls`). If a segment cannot
be reconstructed (assistant turn count ≠ policy calls, partial echo, misaligned logprobs)
the row is masked (`agent_finished=false`, `incomplete_reason=agent_exec_incomplete`, empty
`incomplete` top-level response), never trained on a guess.

Training-config keys for a first smoke (mirror of `tools/gen_stirrup_training_config.py
--compaction-cutoff 0.7 --gym-prefix-supply` in the gdpval_rl_v2 workspace, NeMo RL tree
`nemo_rl_cc`):

```yaml
env:
  nemo_gym:
    training_mode: compaction
    gdpval_train:                      # and gdpval_val
      responses_api_agents:
        gdpval_sandboxed_agent:
          context_summarization_cutoff: 0.7
          context_summarization_max_event_chars: 10000   # optional (default)
          token_id_capture: true       # only with the Gym lineage store below (prefix supply)
    token_id_capture:                  # exact-prefix supply across the summary boundary
      enabled: true
      all_agents: false
      dir: ${env.nemo_gym.nemo_gym_log_dir}/token_capture
      lineage_store: nemo_gym.token_id_capture.lineage:FileLineageStore
      lineage_store_kwargs: {root: ${env.nemo_gym.nemo_gym_log_dir}/token_capture}
      external_staging: false
      rebuild_response: false
      delta_records: true
    policy_model:
      responses_api_models:
        vllm_model:
          supply_prefix_token_ids: true
async_rl:
  multi_trace_enabled: true
token_capture:                         # NeMo RL's capture ledger off: tokens come from the echo
  enabled: false
  top_logprobs: 0                      # 20 (+ vLLM max_logprobs 21, loss_fn.score_centering true) for the SC variant
loss_fn:
  score_centering: false
```

Offline tests: `tests/test_compaction.py` (segmentation, export, contract, condenser wiring
against stubbed OpenHands) and `tests/test_app_compaction.py` (the `/run` body; imports
`app.py`, slow). Run them with the agent venv python, e.g.
`.venv/bin/python -m pytest responses_api_agents/gdpval_sandboxed_agent/tests`.
