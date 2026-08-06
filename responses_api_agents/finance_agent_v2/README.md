# Description

Agent loop for the Vals Finance Agent Benchmark v2 (FABv2). Drives the
`finance_agent_v2` resource server's tools and mirrors the loop policy of
`vals-ai/finance-agent-v2`, which has diverged from the v1 policy implemented by
`responses_api_agents/finance_agent`.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
- model-library: MIT (vals-ai/model-library@0.1.25, from PyPI)
- finance-agent: MIT (vals-ai/finance-agent-v2)

# Why this is a separate component

Forked from `responses_api_agents/finance_agent/app.py` at commit `7b2e174e`.

Vals maintains v1 and v2 as separate repositories. Both build their loop from
the same engine (`model_library.agent.Agent`) and fork only the ~30 lines of
`AgentHooks` that carry per-benchmark policy. Those hook sets have already
diverged:

| | v1 (`vals-ai/finance-agent`) | v2 (`vals-ai/finance-agent-v2`) |
|---|---|---|
| nudge on a text-only turn | `"Continue."` | names `submit_final_result` |
| run bound | `max_turns=50`, no time limit | no turn cap, `TimeLimit(3600s)` |
| tools | web_search, edgar_search, parse_html_page, retrieve_information | adds calculator, price_history |
| retry exhaustion | — | `on_tool_result` raises, aborting the run |

The v1 loop is therefore not a faithful v2 harness. Running FABv2 on it sends
v1's bare `"Continue."` where upstream now explicitly tells the model to submit,
which plausibly changes submission rate and so the score.

Mirroring Vals exactly would mean extracting Gym's loop into a shared engine and
giving each benchmark its own hooks. That refactor touches the v1 loop, which is
on the training path, and the abort-on-tool-error behavior is not expressible as
a defaulted config field. Forking leaves v1 byte-for-byte unchanged. Extracting a
shared base is the right follow-up once v1 is between training runs; until then
this file duplicates Gym plumbing (model call and timeout, usage aggregation,
context-overflow truncation, tool dispatch, cookie threading) that a plumbing fix
would have to land in twice.

# Why the upstream loop is not imported

`model_library.agent.Agent` is async and `get_agent(parameters, llm=...)` accepts
an injected LLM, so importing it is technically possible. It would require an
adapter implementing `model_library.base.LLM.query` against model_library's
internal request/response types, it would move tool execution in-process and so
bypass the Gym resource server that owns the cache, session state, and `/verify`,
and it would have to translate `AgentResult` back into Responses-API items for
Gym's rollout format and RL trajectories. That trades four small mirrored facts
for a dependency on a large private API, so the loop is reimplemented here and
everything else is imported.

# Upstream parity surface

Every parity-relevant value upstream exposes at module level is imported, so it
cannot drift silently:

| Constant | Imported from |
|---|---|
| `UPSTREAM_DONE_TOOL` | `finance_agent.tools.SubmitFinalResult.name` |
| `UPSTREAM_MAX_TIME_SECONDS` | `finance_agent.get_agent.MAX_TIME_SECONDS` |
| `UPSTREAM_VALID_TOOLS` | `finance_agent.tools.VALID_TOOLS` |
| `UPSTREAM_ABORT_TOOL_ERRORS` | `finance_agent.exceptions.RetryExhaustedError` |

`UPSTREAM_NO_TOOL_CALL_NUDGE` is the one value that must be copied, because
upstream builds it as an inline literal inside a closure in
`get_agent._before_query`. `tests/test_app.py::TestUpstreamParity` asserts the
copy still appears verbatim in upstream's source, so a reword fails a test at
bump time rather than silently changing results.

The same tests assert that this component's `finance-agent` and `model-library`
pins match `resources_servers/finance_agent_v2/requirements.txt` exactly, so the
agent and the tools it drives can never be built from two different commits.

## Bumping the upstream pin

1. Update the pin in **both** `requirements.txt` files to the same commit.
2. Run `pytest responses_api_agents/finance_agent_v2/tests -q`.
3. For each parity failure, read that symbol's upstream diff and decide whether
   to follow it or pin the old behavior via config.
4. Re-baseline, and record the new SHA next to the scores — a run is only
   comparable to another run made against the same upstream commit.

# Configuration

Defaults reproduce upstream v2 policy; every one can be overridden.

| Key | Default | Notes |
|---|---|---|
| `max_steps` | `null` | Upstream sets no turn cap. Set it to bound a smoke run. |
| `max_time_seconds` | `3600.0` | Hard stop, checked before each model call. The resource server's `max_rollout_time_seconds` is a softer budget that only makes tools return an error asking the model to submit. |
| `done_tools` | `["submit_final_result"]` | Remaining calls in the batch are skipped when one fires. |
| `no_tool_call_nudge` | v2 text | Injected when a turn produces prose and no tool call. |
| `abort_on_tool_error_types` | `["RetryExhaustedError"]` | Error types that end the rollout instead of being fed back to the model. Set to `[]` to disable. |
| `max_tool_calls_per_turn` | `null` | Upstream supports it; v2 leaves it unset. Capped calls still get a result item because providers require one per call. |
| `model_call_timeout` / `tool_call_timeout` | `null` | Gym-side operational knobs with no upstream equivalent. |
| `truncate_on_overflow` | `false` | Eval only — training needs the full trajectory for reward assignment. |

# Known divergences from upstream

Faithful within the constraints of Gym's architecture, but not identical:

- **Time accounting.** Upstream subtracts retry overhead from the elapsed
  reading. This loop does not retry model calls, so wall clock is the same
  measurement.
- **Context overflow.** Upstream raises a typed
  `MaxContextWindowExceededError`; the model here is reached over HTTP, so
  overflow is detected by matching the provider's error text.
- **History compaction.** Upstream supports compaction as an alternative to
  truncation. v2's `AgentConfig` does not enable it, so only truncation is
  implemented.
- **Stop reason.** `StopReason` mirrors `model_library.agent.AgentStopReason` and is
  written to `response.metadata` as `stop_reason` (plus a `steps` count), so it
  reaches the rollout file through `/verify` without a resource-server change.
  `metadata` is the Responses API's own string-keyed side channel, which is why it
  survives the round trip. Use it to tell a trajectory that ran out of time or turns
  from one that answered and was judged wrong — under dealbreaker-gated scoring both
  land at reward 0.0. Runs collected before this was added have no `stop_reason`, and
  `scripts/report_run.py` shows those as `(not recorded)`.
