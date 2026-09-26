# Sandboxed mini-SWE

Generic mini-SWE 2.4.6 `DefaultAgent` execution on a caller-owned `AsyncSandbox`.
`harness.py` exposes `MiniSWEHarness`, `HarnessContext`, `MiniSWEConfig`, and
`HarnessOutcome`. The caller supplies the sandbox, task instruction, execution
user and working directory, setup budget, optional MCP/skills configuration,
artifact directory, and Gym's model-server URL. The harness imports no
benchmark code and has no dataset, provisioning, verification, or sandbox lifecycle logic.

The mini-SWE `DefaultAgent` and its shell commands run inside the task sandbox.
Setup installs the pinned package into an isolated Python 3.13 environment and
uploads `sandbox_runner.py`. The task image needs `python3`, `bash`, and `setsid`,
plus network access to download Python and mini-SWE dependencies during setup.
Gym downloads uv for the sandbox's architecture and uploads it, so bootstrap
also works in task images without system CA certificates.
The agent calls the configured Gym model server directly at its rollout-prefixed
`/v1/responses` URL, as OpenCode does. That URL retains model-call capture and
training token capture through Gym's model wrapper; `x-session-id` correlates
failed calls to the resource session. The sandbox saves its model and tool history
as an artifact that the harness downloads after the single agent command exits.
For a remote sandbox, the model server must have an address reachable from that
sandbox. If Gym's configured server address is host loopback, set
`sandbox_model_base_url` to the sandbox-reachable HTTP(S) address of the same Gym
model server. The TB4 profile exposes this as `++tb4_sandbox_model_base_url=...`.
The address may end in `/v1`; the agent preserves its path prefix and appends the
run's rollout and token-capture paths before `/v1`.

`app.py` owns the Gym `/run` loop: it calls the configured resources server's
`/seed_session`, connects to the returned sandbox, and stores agent state under
the inbound request's client session ID, with the resource session ID inside the
state. It passes the same request and original create-params to `responses()`,
which retrieves the state and executes the sandbox runner. `/run` sends the response and
termination to `/verify` after execution stops and releases its session state and
transport. The resources server retains provisioning, renewal, grading, and
sandbox destruction. Retried runs share one execution; seeded cookies are
forwarded to verification.

`/v1/responses` requires an initialized mini-SWE session; it neither provisions
nor verifies a benchmark. Separate agent-session endpoints are not needed for
this `/run` flow. In full swapping, an environment-server request to
`/v1/responses` will replace the direct method call.

Cancellation stops the runner and its tool process groups before verification.
The adapter retains completed Responses items and downloads the native partial
trajectory when available. Runtime metadata records the sandbox hostname, process
ID, and Python executable so execution placement can be checked.

`models.py` defines the agent's local view of the HTTP protocol; it imports no
resources implementation. Seeding supplies `session_id`, `sandbox_descriptor`,
`sandbox_provider`, `instruction`, and optional `task_id`, `user`,
`agent_timeout_sec`, `mcp_servers`, and `skills_dir`. A failed seed can supply
`termination`; a completed session can supply `verified_response` for replay.
Task-specific `/run` fields are forwarded unchanged. Verification receives the
response and agent execution status; its result only needs Gym's
`BaseVerifyResponse` fields. Additional benchmark result fields pass through
unchanged. Other harnesses can implement the same HTTP exchange without importing
mini-SWE or TB4 code.

Agent shutdown uses one `shutdown_timeout_sec` budget for finishing an in-flight
seed request, stopping the sandbox runner, and requesting verification or cleanup. If the
seed response remains unavailable, it cancels the local request and returns;
the resources server's seeded-session deadline cleans up the abandoned sandbox.

Agent configuration owns `model_server`, `harness`, `agent_max_timeout_sec`, and
`artifacts_dir`. Harness setup (including reconnect and working-directory discovery)
has a separate 360-second budget. Execution uses the smaller of the official task
budget and the configured cap. The benchmark retains the `tb4_max_steps`,
`tb4_step_timeout_sec`, and `tb4_agent_max_timeout_sec` overrides; custom nested
overrides must now target `terminal_bench_4_miniswe.responses_api_agents.miniswe_sandboxed_agent`.
Agent trajectories default to `results/miniswe_sandboxed_agent/<session_id>/`.
The TB4 profile selects `results/terminal_bench_4/agent/<session_id>/`, overridable
with `tb4_agent_artifacts_dir` or a run's `artifact_directory`. When
`tb4_jobs_dir` is set, agent artifacts default to its `agent/` subdirectory so the
configured run directory captures artifacts from both servers.

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
Output-limit truncation (`incomplete_details.reason=max_output_tokens`) is passed
to mini-SWE as `finish_reason="length"`, enabling its native “Respond more concisely”
reminder when tool calls are missing or malformed. Valid tool calls still execute.
The adapter preserves Responses output items (including reasoning and tool calls)
when replaying history and returns observations with their matching call IDs.

Command observations use `mini.yaml`'s JSON format and first/last 5,000-character
limit, including timeout metadata. Full command output remains available in the
native trajectory's `extra.raw_output`; the model sees the bounded observation.
There is no context compaction or summarization. Execution uses
`DefaultAgent` without interactive confirmations and keeps cost limits disabled.

Length-limited responses use mini-SWE's concise-response recovery prompt. If all
consecutive format errors are length-limited, the terminal status is
`OutputTokenLimitExceeded`. For vLLM, enable
`policy_model.responses_api_models.vllm_model.propagate_context_overflow_errors: true`
so an overfull input is reported as `ContextWindowExceeded` instead of a synthetic
empty completion. The agent stops without retrying that input, and the caller can
still verify its partial work. Other model API errors remain infrastructure errors.
These changes affect benchmark trajectories and results compared with the prior
unbounded-observation profile.

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

Enable `observability_enabled: true` in Gym's run configuration to collect
`ng_agent_observations` and canonical `ng_trajectory` turns. Direct harness
callers pass `observability_enabled=True`; collection is disabled by default.
Responses without an ID retain their turn and record a
`model_call_reference_unavailable` gap.
The agent sends the resources session ID as `x-session-id` on every Gym model request,
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

The agent unit tests use an unrelated resource schema. `tests/test_server.py`
retains the TB4 composition tests in the server test suite for CI discovery; those
tests require both servers' dependencies.
