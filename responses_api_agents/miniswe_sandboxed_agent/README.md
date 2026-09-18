# Sandboxed mini-SWE

Generic mini-SWE 2.4.6 `DefaultAgent` execution on a caller-owned `AsyncSandbox`.
`harness.py` exposes `MiniSWEHarness`, `HarnessContext`, `MiniSWEConfig`, and
`HarnessOutcome`. The caller supplies the sandbox, task instruction, execution
user and working directory, setup budget, optional MCP/skills configuration,
artifact directory, and an async model-query callback. The harness imports no
benchmark code and has no dataset, provisioning, verification, or sandbox lifecycle logic.

The synchronous mini-SWE loop uses a bridge to async model and sandbox operations.
Cancellation closes pending I/O and joins the worker before returning its outcome,
response, and trajectory metadata. The caller owns subsequent collection and
cleanup. `app.py` is a thin Gym collector adapter: it forwards `/run` to the
configured resources runner, preserving session identity and model-call capture
routing, and returns its verification response.

The adapter loads system and instance prompts from the pinned package's `mini.yaml`
and exposes mini-SWE's native `bash` tool through Gym's Responses API. The version
and prompts follow [Artificial Analysis's TB4 methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking).
The prompt defines completion as a successful command whose first output
line is `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, matching mini-SWE's convention.
The Python defaults remain `step_limit=0`, `step_timeout_sec=600`, and `cost_limit=0`;
the caller bounds total execution time. The TB4 benchmark sets a 500-step limit and
30-second command timeout, overridable with `++tb4_max_steps=...` and
`++tb4_step_timeout_sec=...`. Commands receive the environment defaults from `mini.yaml`;
system information in the prompt comes from the task sandbox.
Every step persists the native mini-SWE trajectory, including observations.
The adapter preserves Responses output items (including reasoning and tool calls)
when replaying history and returns observations with their matching call IDs.

Full command observations are deliberately retained: we do not use `mini.yaml`'s
first/last 5,000-character truncation. AA's intent regarding that upstream default
is unclear. There is no context compaction or summarization. Execution uses
`DefaultAgent` without interactive confirmations and keeps cost limits disabled.

Task skills are exposed by their supplied directory. For MCP tasks, setup installs
`mcp==1.29.0` into a task-local virtual environment, discovers the declared tools,
and adds their schemas and invocation command to the prompt. The CLI supports
stdio, SSE, and streamable HTTP; calls execute inside the main sandbox so service
names retain their task-network meaning. A persistent MCP session preserves state
across calls. MCP tools are visible as schemas and CLI instructions in the task
prompt and invoked through native `bash` calls, rather than registered as separate
model tools. Image tool results become multimodal model inputs. This changes the evaluation profile
relative to native mini-SWE and must be disclosed in score comparisons.

Use [the TB4 mini-SWE profile](../../benchmarks/terminal_bench_4/miniswe.yaml) with
[TB4 resources](../../resources_servers/terminal_bench_4/README.md). Existing
`mini_swe_agent_2` SWE-bench behavior remains unchanged. Coverage is a validation
claim, not implied by selecting this configuration.

The TB4 profile retains one repeat; clients seeking AA's three-repeat protocol
must explicitly set `++num_repeats=3`. Verifier timeout handling is unchanged:
timeouts remain infrastructure failures pending evidence of their frequency on
real workloads. These decisions are detailed in the benchmark README and mean
the profile is not an exact reproduction of AA's evaluation.

## Rollout observability

The harness returns `ng_agent_observations` and canonical `ng_trajectory` turns.
TB4 sends the resources session ID as `x-session-id` on every Gym model request,
so capture can assign failed attempts and retries to the same invocation even
when there is no response ID. Successful responses retain exact response refs;
decisions are recorded before mini-SWE parses them, including rejected output.

Tool observations measure each dispatched bash call independently with monotonic
duration, wall-clock bounds, and outcome. Combined Responses output now includes
tool results, including terminal submission. The native mini-SWE transcript and
the model-visible prompts retain their existing behavior. Execution stays serial.
Intermediate canonical turns remain unresolved; agent submission is not verifier
success. Missing aggregate usage is preserved as unknown, never a partial or empty
zero sum.

`benchmarks/terminal_bench_4/smoke.py` now retains `model_calls/`,
`evaluator_rollouts.jsonl`, and Gym's `quality_summary.json` for offline inspection.
The separate harness capability checker currently lives in EFB's
`harness_capabilities` package. A passing artifact gate does not establish the
RFC's complete behavioral qualification or parallel-execution capability.
