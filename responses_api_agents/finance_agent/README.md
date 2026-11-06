# Description

Multi-turn tool-calling loop shared by finance research benchmarks. Server instances configure it;
the loop itself has no benchmark-specific branches.

| Instance | Config | Resource server |
|----------|--------|-----------------|
| `finance_agent` | `resources_servers/finance_sec_search/configs/finance_sec_search.yaml` | SEC filing search (also the training path) |
| `finance_agent_v2` | `resources_servers/finance_agent_v2/configs/finance_agent_v2.yaml` | Vals finance-agent-v2 tools |
| `big_finance` | `resources_servers/big_finance/configs/big_finance.yaml` | BigFinanceBench example fixture |
| `big_finance_benchmark_agent` | `benchmarks/big_finance/config.yaml` | BigFinanceBench tools and verifier |

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

Four benchmark-neutral loop policies have compatibility-preserving defaults in the base
config:

| Field | Values | Default |
|-------|--------|---------|
| `prose_only_behavior` | `nudge`: inject `no_tool_call_nudge` after prose; `finish`: end any turn without tool calls, including empty or reasoning-only turns | `nudge` |
| `tool_call_execution` | `sequential`; `concurrent` | `sequential` |
| `tool_error_observation` | `json`: preserve the error payload; `error_prefix`: render `[ERROR] <message>` | `json` |
| `done_tool_requires_success` | Require a successful terminal tool before ending either execution mode | `false` |

The shipped Vals v1/v2 profiles therefore retain their existing nudge and sequential execution
behavior without overrides. A profile whose protocol treats prose as the final answer can select
`finish`. A profile whose tools are independent can select `concurrent`; every call in the turn is
then executed, while `function_call_output` items are appended in the model's original call order.
`done_tool_requires_success` controls whether a failed terminal tool can end the loop in
either mode. Concurrent execution always finishes the complete batch before ending.
Custom overlays should use `done_tool_requires_success` in place of the earlier
`sequential_done_tool_requires_success` key.

Tool failures come back to the model as a tool result so the rollout survives.
Resources servers can set `x-nemo-gym-tool-error: true` or `false` on tool responses to
identify failures explicitly. Successful response bodies are preserved even when they
contain an `error` field or begin with `[ERROR]`. Responses without this header retain
the legacy JSON error detection. Explicit per-call statuses are saved as a JSON mapping
in response metadata under `tool_call_error_status` for stateless reverification.
Every response
carries `stop_reason` and `steps` in its metadata so a truncated trajectory is distinguishable from
a submitted answer. Prose completion also records `final_model_turn_start` so an empty
final turn cannot reuse earlier analysis as its answer. Context overflow optionally drops the oldest exchange and retries
(`truncate_on_overflow`), which is for eval only — during training the full trajectory has to be
preserved for reward assignment.

# Running behind an Environment Server

The loop also runs behind the `single_agent_turn_legacy` Environment Server, which seeds the resources session and an
agent session, calls `/v1/responses` once, closes the agent session, verifies, and closes the resources session. The
agent and the finance resources servers export no checkpoint state (`restart_only`), so after a crash the episode
still starts over from its input. What changes is that a failed model call, seed or judge becomes a classified failure
instead of reward 0. Each finance_sec_search and finance_agent_v2 relay config has a counterpart that does this; the
relay configs are unchanged. The big_finance configs stay on the relay until the big_finance resources server serves
typed resources sessions:

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

`finance_agent` is registered in `AGENT_SESSION_AGENT_TYPES`, so composing it onto another environment keeps that
environment's `single_agent_turn_legacy` Environment Server, which must then grant `direct_http` too; otherwise every
seed gets 422.

Outcomes differ from the legacy `/run`, which verifies every loop it finishes and turns any exception into reward 0.
Under the Environment Server, a failed model call other than a context overflow, or a tool abort, marks the response
`failed` with its cause, and the Environment Server's `agent_outcome_policy` decides what happens:

| Event | Legacy `/run` | Environment Server, default policy |
|-------|---------------|------------------------------------|
| Model call fails with a context overflow in its error text, at any HTTP status (`openai_model` answers a provider's 400 with a 500 unless `propagate_upstream_http_status_codes` lists it) | verified, usually 0 | response `incomplete`, no failure, verified |
| Model call fails with another HTTP 400, 413 or 422 | verified, usually 0 | `model_request_error`, verified |
| Model call fails with HTTP 408, 425, 429 or 5xx | verified, usually 0 | `model_request_error`, retryable failure |
| Model call fails with another HTTP 4xx | verified, usually 0 | `model_request_error`, terminal failure |
| Model call times out (`model_call_timeout`), or cannot connect before it does. Without `model_call_timeout`, a refused connection is resent without limit, so the episode runs into the Environment Server's deadline instead | verified, usually 0 | `model_transport_error`, retryable failure |
| Seed, verify or the agent itself fails | reward 0 | failure classified by the Environment Server |
| Tool error or timeout | verified | verified, as before |
| A tool reports an error type in `abort_on_tool_error_types` (v2: `RetryExhaustedError`, the SEC API or sec.gov still answering 429 or 503 after its retries) | verified, usually 0 | `unknown`, retryable failure |
| Context overflow that `vllm_model` answers itself (its default) | verified | verified, as before |

The tool-abort row departs from upstream on purpose: `vals-ai/finance-agent-v2` ends the question with an empty answer,
and the legacy `/run` still grades the rollout like upstream, but on the Environment Server path an SEC outage is
retried instead of scored 0.

A failure is not scored: rollout collection writes it to the failures sidecar instead of the rollouts file, and
`--resume` dispatches a retryable one again, up to `NEMO_GYM_MAX_ROLLOUT_ATTEMPTS`. So an outage no longer counts as a
wrong answer, and the score covers fewer rollouts when episodes fail. To count these failures as 0, pass
`+count_failure_classes_as_zero=[environment_server_failed]`. Like the legacy run, that also zeroes seed, judge and
verify failures, which share the class.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
