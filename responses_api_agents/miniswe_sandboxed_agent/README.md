# Sandboxed mini-SWE

Run pinned mini-SWE 2.4.6 directly in a resource-owned Linux sandbox, using its
native CLI, `DefaultAgent`, `LocalEnvironment`, and `litellm_response` model.
Use [the TB4 profile](../../benchmarks/terminal_bench_4/miniswe.yaml) with
[TB4 resources](../../resources_servers/terminal_bench_4/README.md).

`harness.py` supplies execution and trajectory conversion; it imports no benchmark
code. `/run` seeds the resources server, connects to its sandbox, stores session
state, calls `responses()`, and sends the result to resources `/verify` after
stopping agent processes. Resources owns provisioning, grading, and destruction.
`/v1/responses` requires a connected session and owns working-directory discovery,
config and model routing, harness setup, and execution. It neither provisions nor
grades. Retried responses share setup and execution for the same message.
Retried `/run` calls share one execution and forward the seeded resource cookies.

## Setup and model access

Task images need `bash`, `uname`, `tar`, and `setsid`; preinstalled Python, pip,
curl, and CA certificates are unnecessary. Setup detects the sandbox architecture
and libc using shell commands, then installs Python 3.13.12 and uv 0.10.12 as the
task user. Both archives are downloaded by Gym and cached under
`$TMPDIR/nemo-gym-miniswe-assets` (or `/tmp`), shared across tasks. Concurrent
initial setup requests share the downloads, and Python archives are checksum
verified. Downloads are pinned separately for x86-64/ARM64 and glibc/musl.
The sandbox never downloads these assets from GitHub; PyPI access is still needed
for the isolated mini-SWE environment. LiteLLM uses its bundled model-cost map.

The native agent calls Gym's rollout-prefixed `/v1/responses` endpoint directly.
Gym's model wrapper retains model-call and training-token capture; every request
includes the resource session's `x-session-id` header. Set
`sandbox_model_base_url` when the configured model address is not reachable from
the sandbox. The TB4 override is `++tb4_sandbox_model_base_url=...`; an optional
`/v1` suffix is accepted and any reverse-proxy path is preserved.

The caller supplies the task instruction, task user and working directory,
setup/execution budgets, optional skills/MCP configuration, and artifact path.
Setup has a separate 360-second budget. Execution uses the smaller of the task's
budget and `agent_max_timeout_sec`. Default artifacts are written to
`results/miniswe_sandboxed_agent/<session_id>/`; TB4 uses its configured jobs'
`agent/` directory. `artifact_directory` can override the location per run.

## Native behavior and observability

The CLI loads `mini.yaml` from the pinned package and overlays the task/model
configuration. Defaults remain unlimited steps, a 600-second command timeout,
and disabled cost limits. TB4 selects 500 steps and a 30-second command timeout;
`++tb4_max_steps`, `++tb4_step_timeout_sec`, and `++tb4_agent_max_timeout_sec`
override those values. Completion follows mini-SWE's
`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` convention. Native observations use the
package's first/last 5,000-character bounds and preserve raw output in trajectories.
There is no context compaction. Native format-error recovery and exception names
are retained, including `RepeatedFormatError` and `ContextWindowExceededError`.
Gym disables extra SDK/model retries; the model server retains its retry policy.

`trajectory.json` is the unmodified native transcript; `agent.log` contains CLI
stdout/stderr. With `observability_enabled: true`, Gym projects saved decisions,
reasoning, tool results, return codes, completion timestamps, usage, and native
model response references into `ng_agent_observations` and `ng_trajectory`.
LiteLLM's response-ID envelope is decoded to join Gym's original capture IDs.
Missing aggregate usage stays unknown. Submission is separate from verifier success.

Removing the custom wrapper has these limits:

- Native mini-SWE records tool completion timestamps, but no start times or
  durations. Gym emits `tool_timing_unavailable` gaps rather than estimating them.
- Submission exits before a final tool observation is saved. The decision and
  native submission text remain available; that tool's execution observation is
  incomplete, with no output or error evidence. This known mini-SWE gap fails
  TE-5 (`tool.terminal` and `tool.outcome`) and the `gym-p0/v1` conformance gate,
  even when the task succeeds. mini-SWE needs to persist a terminal tool result
  correlated with the submit call before exiting; Gym's projection must then
  retain that evidence. The observability test asserts this gap until it is fixed.
- Native trajectories are saved after each step. A forced kill during a model
  request or command can lose the active step, or leave a partially written
  trajectory. Gym model-server capture remains independent of that file.
- Wrapper-specific runtime metadata and `OutputTokenLimitExceeded` normalization
  are removed. Native termination names and CLI logs are retained.
- MCP tools remain available through a persistent task-local CLI session, but
  native Responses observations do not promote MCP image blobs to multimodal
  inputs. Such results remain command text subject to native output bounds.

Cancellation stops processes carrying the run's inherited environment marker,
including native shell commands in separate process groups, and registered MCP
process groups before verification. Resources must quiesce the sandbox if cleanup
fails. Shutdown uses one `shutdown_timeout_sec` budget for seed completion,
execution cleanup, and verification; resources expires abandoned seed sessions.

`benchmarks/terminal_bench_4/smoke.py` retains model captures, rollout JSONL,
health summaries, and source hashes. `--host` selects a service address reachable
from remote sandboxes. Three-step smokes validate execution and verification,
not task-solving accuracy. The TB4 profile retains one repeat; the three-repeat
protocol requires `++num_repeats=3`. Selecting this profile does not establish
benchmark coverage or exact methodology parity.
