# Description

Multi-turn tool-calling loop for the Vals AI finance agent benchmarks. Two server instances
configure it, and they are the only difference between the two harnesses — the loop itself has no
benchmark-specific branches.

| Instance | Config | Resource server |
|----------|--------|-----------------|
| `finance_agent` | `resources_servers/finance_sec_search/configs/finance_sec_search.yaml` | SEC filing search (also the training path) |
| `finance_agent_v2` | `resources_servers/finance_agent_v2/configs/finance_agent_v2.yaml` | Vals finance-agent-v2 tools |

Three fields have no default, so each instance states its own policy:

| Field | `finance_agent` | `finance_agent_v2` |
|-------|-----------------|--------------------|
| `no_tool_call_nudge` | `Continue.` | names `submit_final_result` |
| `max_time_seconds` | `null` (turn-bounded) | `3600` |
| `abort_on_tool_error_types` | `[]` (every tool error is fed back) | `[RetryExhaustedError]` |

`continue_if_not_tool_call` defaults to `true` in the agent (eval keeps nudging).
GRPO sets `false` in the NVFlow overlay so training does not inject a synthetic
user turn. Shipped yaml does not need to set it.

The v2 values mirror `vals-ai/finance-agent-v2` and are checked against the installed upstream
package by `resources_servers/finance_agent_v2/tests/test_upstream_parity.py`.

Beyond that: a prose-only turn is nudged rather than ending the run, tool failures come back to the
model as a tool result so the rollout survives, and every response carries `stop_reason` and `steps`
in its metadata so a truncated trajectory is distinguishable from a submitted answer. Context
overflow optionally drops the oldest exchange and retries (`truncate_on_overflow`), which is for
eval only — during training the full trajectory has to be preserved for reward assignment.

# Running behind an Environment Server

The loop also runs behind the `single_agent_turn_legacy` Environment Server, which seeds the resources session and an
agent session, calls `/v1/responses` once, closes the agent session, verifies, and closes the resources session. A
crash then repeats only the step it interrupted instead of the whole episode. Each relay config has a counterpart that
does this; the relay configs are unchanged:

| Relay config | Environment Server config |
|--------------|---------------------------|
| `resources_servers/finance_sec_search/configs/finance_sec_search.yaml` | `resources_servers/finance_sec_search/configs/finance_sec_search_single_agent_turn.yaml` |
| `resources_servers/finance_agent_v2/configs/finance_agent_v2.yaml` | `resources_servers/finance_agent_v2/configs/finance_agent_v2_single_agent_turn.yaml` |
| `benchmarks/finance_sec_search/config_no_web_search.yaml` | `benchmarks/finance_sec_search/config_no_web_search_single_agent_turn.yaml` |
| `benchmarks/finance_sec_search/config_web_search.yaml` | `benchmarks/finance_sec_search/config_web_search_single_agent_turn.yaml` |
| `benchmarks/finance_agent_v2/config.yaml` | `benchmarks/finance_agent_v2/config_single_agent_turn.yaml` |

An agent session takes its tools from the Environment Server's grant, so the Environment Server must set
`resources_tool_transports: [direct_http]`. The agent refuses, with a final 422, a session with no direct HTTP grant,
more than one, a required MCP grant, or sandbox access. Every tool call goes to the grant's base URL with its headers
and the resources cookie jar, and close returns the updated jar. When model calls are captured
(`observability_enabled`), close also returns the loop's observations: one invocation with its stop reason and model
calls, and a record per tool call.

Outcomes differ from the legacy `/run`, which verifies every loop it finishes and turns any exception into reward 0.
Under the Environment Server, a failed model call marks the response `failed` with its cause, and the Environment
Server's `agent_outcome_policy` decides what happens:

| Event | Legacy `/run` | Environment Server, default policy |
|-------|---------------|------------------------------------|
| Model call fails with HTTP 400, 413 or 422, such as a context overflow the model server propagates | verified, usually 0 | `model_request_error`, verified |
| Model call fails with HTTP 408, 425, 429 or 5xx | verified, usually 0 | `model_request_error`, retryable failure |
| Model call fails with another HTTP 4xx | verified, usually 0 | `model_request_error`, terminal failure |
| Model call times out (`model_call_timeout`) or cannot connect | verified, usually 0 | `model_transport_error`, retryable failure |
| Seed, verify or the agent itself fails | reward 0 | failure classified by the Environment Server |
| Tool error or timeout, or an error type in `abort_on_tool_error_types` | verified | verified, as before |
| Context overflow that `vllm_model` answers itself (its default) | verified | verified, as before |

A failure is not scored: rollout collection writes it to the failures sidecar instead of the rollouts file, and
`--resume` dispatches a retryable one again, up to `NEMO_GYM_MAX_ROLLOUT_ATTEMPTS`. So an outage no longer counts as a
wrong answer, and the score covers fewer rollouts when episodes fail. To count failures as 0, as the legacy run did,
pass `+count_failure_classes_as_zero=[<class>, ...]`.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
