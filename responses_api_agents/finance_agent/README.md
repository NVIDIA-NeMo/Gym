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

The v2 values mirror `vals-ai/finance-agent-v2` and are checked against the installed upstream
package by `resources_servers/finance_agent_v2/tests/test_upstream_parity.py`.

Beyond that: a prose-only turn is nudged rather than ending the run, tool failures come back to the
model as a tool result so the rollout survives, and every response carries `stop_reason` and `steps`
in its metadata so a truncated trajectory is distinguishable from a submitted answer. Context
overflow optionally drops the oldest exchange and retries (`truncate_on_overflow`), which is for
eval only — during training the full trajectory has to be preserved for reward assignment.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
