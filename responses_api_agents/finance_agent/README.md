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

The v2 values mirror `vals-ai/finance-agent-v2` and are checked against the installed upstream
package by `resources_servers/finance_agent_v2/tests/test_upstream_parity.py`.

Two additional, benchmark-neutral loop policies have compatibility-preserving defaults in the base
config:

| Field | Values | Default |
|-------|--------|---------|
| `prose_only_behavior` | `nudge`: inject `no_tool_call_nudge`; `finish`: return the assistant text | `nudge` |
| `tool_call_execution` | `sequential`; `concurrent` | `sequential` |

The shipped Vals v1/v2 profiles therefore retain their existing nudge and sequential execution
behavior without overrides. A profile whose protocol treats prose as the final answer can select
`finish`. A profile whose tools are independent can select `concurrent`; every call in the turn is
then executed, while `function_call_output` items are appended in the model's original call order.
A successful terminal tool ends the loop after the complete concurrent batch finishes.
Both shipped BigFinance profiles select `finish`, `concurrent`, and
`done_tools: [final_answer]`; profile tests load those values from the shipped
configs to guard against silent protocol drift.

Tool failures come back to the model as a tool result so the rollout survives, and every response
carries `stop_reason` and `steps` in its metadata so a truncated trajectory is distinguishable from
a submitted answer. Context overflow optionally drops the oldest exchange and retries
(`truncate_on_overflow`), which is for eval only — during training the full trajectory has to be
preserved for reward assignment.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
