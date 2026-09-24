# Sandboxed mini-SWE

Generic mini-SWE 2.4.6 `DefaultAgent` execution on a caller-owned `AsyncSandbox`.
`harness.py` exposes `MiniSWEHarness`, `HarnessContext`, `MiniSWEConfig`, and
`HarnessOutcome`. The caller supplies the sandbox, task instruction, execution
user and working directory, setup budget, optional MCP/skills configuration,
artifact directory, and an async model-query callback. The harness imports no
benchmark code and has no dataset, provisioning, verification, or sandbox lifecycle logic.

The mini-SWE `DefaultAgent` and its shell commands run inside the task sandbox.
Setup installs the pinned package into an isolated Python 3.13 environment and
uploads `sandbox_runner.py`. The task image needs `python3`, `bash`, and `setsid`,
plus network access to download Python and mini-SWE dependencies during setup.
Gym downloads uv for the sandbox's architecture and uploads it, so bootstrap
also works in task images without system CA certificates.
Model requests travel over an atomic JSON file relay to the Gym agent server,
which forwards them to the configured model server with the existing capture and
session correlation. The sandbox does not need direct access to model credentials.
Tool observations and the native trajectory return over the same sandbox transport.

Use `episode.py` and the generic
[agent configuration](configs/miniswe_sandboxed_agent.yaml) with a
`single_agent` EnvironmentServer. Bind `agent_server`, `resources_server`, the
model server, and the sandbox provider in the run configuration, then submit
`SingleAgentEpisodeRequest` to the **EnvironmentServer** `/run` endpoint.
The native agent needs only its model reference; benchmark data stays in Resources.
The standalone YAML retains the existing `resources_server.name: ???` selector
for CLI agent composition. The environment configuration fills that selector;
native session execution uses the sandbox handed over by EnvironmentServer.

The native lifecycle is:

1. Resources `/seed_session` creates and prepares the task sandbox.
2. Agent `/v1/agent_sessions` connects to `SandboxAccess` and installs the runtime.
   It publishes the session cookie only after setup succeeds.
3. Agent `/ng-rollout/{capture_key}/v1/responses` activates the installed mini-SWE
   loop once in the supplied workdir and as the task user.
4. Agent `/v1/agent_sessions/close` confirms process cleanup, removes its runtime
   files, disconnects, and returns observations.
5. Resources `/verify` grades the task; `/close_session` releases owner resources.

Setup, execution, verification, and sandbox destruction have separate owners.
No host CLI execution is used by native sessions. Setup checks Linux, architecture,
`python3`, `bash`, and `setsid`; failures retain installer output in episode diagnostics.
Runtime files, HOME, and caches are isolated under a unique `/tmp` directory outside
the task repository. Task dependencies are not replaced. Only runtime settings and
explicit task MCP configuration enter the sandbox; model and provider credentials
stay in Gym.

The adapter accepts a string or text-only task messages. `instructions`, sampling,
reasoning, and output limits are forwarded to each model call. `max_output_tokens`
is a **per-call** limit; `harness.step_limit` and `agent_max_timeout_sec` bound the
activation. Unsupported stateful inputs, custom tools, and request options fail
before launch. A session permits one activation; duplicate or concurrent activation
returns 409. Cookie ownership, episode identity, and the capture route must match.

One Linux child subreaper supervises each activation. Close waits for the worker
and all tool descendants, including double-forked and detached children, before
reporting success. A missing launch handle or cleanup receipt cannot establish
successful cleanup. Such failures block grading and retain state for close retries.
Concurrent closes serialize; successful close receipts remain available for
`closed_session_retention_sec` (default 300 seconds), after which stale cookies
return 404. Failed sessions remain process-local until recovery or server restart;
owner/provider expiry handles crashes. Cleanup is not a security boundary against
hostile sandbox code. Supervisor overhead has not been measured at scale.

Cancellation retains completed Responses items and the available native partial
trajectory. Runtime metadata includes sandbox hostname, PID, UID, Python executable,
and harness version. Missing aggregate usage remains unknown. Inference smokes do
not establish training token-ID/logprob support.

The legacy `app.py` entrypoint remains available for existing callers using the
agent `/run` and the legacy resource seed/verify exchange. New sandbox integrations
should use the native EnvironmentServer flow above.

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
`mcp==1.29.0` into the isolated runtime. Activation starts the MCP client under the
same supervisor, discovers tools, and adds their schemas and invocation command
to the prompt. The CLI supports
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
The agent sends its agent session ID as `x-session-id` on every Gym model request,
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
