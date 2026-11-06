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
agent session, calls `/v1/responses` once, closes the agent session, verifies, and closes the resources session. The
agent and the finance resources servers export no checkpoint state (`restart_only`), so after a crash the episode
still starts over from its input. What changes is that a failed model call, seed or judge becomes a classified failure
instead of reward 0. Each relay config has a counterpart that does this; the relay configs are unchanged:

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

A failed reply from the model server carries that server's failure report, and the agent classifies it as
`simple_agent` does (`policy_call_failure` in `nemo_gym.agents`). The response keeps the report, so the failure row
takes the kind of the cause the report selects. A model call that timed out after upload on every attempt is
`model_policy_response_timeout`, retried by default. A reply whose model request failed after its response headers
is not answered as a session failure at all, since the provider may have generated an answer. A reply whose selected
cause has another HTTP status than the reply fails the episode. `openai_model` answers a provider's HTTP error with a
500 unless `propagate_upstream_http_status_codes` lists its status, so by default a provider's 503 fails the episode
instead of being retried; list the statuses to retry there.

The tool-abort row departs from upstream on purpose: `vals-ai/finance-agent-v2` ends the question with an empty answer,
and the legacy `/run` still grades the rollout like upstream, but on the Environment Server path an SEC outage is
retried instead of scored 0.

A failure is not scored: rollout collection writes it to the failures sidecar instead of the rollouts file, and
`--resume` dispatches a retryable one again, up to `NEMO_GYM_MAX_ROLLOUT_ATTEMPTS`. So an outage no longer counts as a
wrong answer, and the score covers fewer rollouts when episodes fail. A failure row's class is the kind its
report selected, such as `model_request_error` or `model_policy_response_timeout`, and `environment_server_failed`
otherwise. To count these failures as 0, pass
`+count_failure_classes_as_zero=[model_request_error,model_policy_response_timeout,environment_server_failed]`. Like
the legacy run, that also zeroes seed, judge and verify failures, which share the last class.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
