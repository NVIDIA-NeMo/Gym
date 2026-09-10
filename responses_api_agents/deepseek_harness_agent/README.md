# DeepSeek Harness agent

Runs the upstream DeepSeek Harness Python SDK **0.1.5rc1** inside a task sandbox
through Gym's `AsyncSandbox` API. The resources server prepares the task and
verifies the resulting workspace. The agent contains no benchmark-specific image,
path, patch-extraction, or scoring logic.

The default `sdk` profile retains the upstream harness. `sdk-minimal` selects the
upstream persistent-shell-only profile for connectivity checks; it is a different
agent configuration and should be reported separately in benchmark results.

## Runtime and sandbox requirements

- A resources server whose `/seed_session` returns `sandbox_descriptor`, accepted
  by `AsyncSandbox.connect()`, or the existing `sandbox_handle` opaque-id field.
  The latter is supported by the OpenSandbox execution path used by SWE-bench.
- A matching, connectable Gym sandbox provider. Direct Sandbox API integration
  does not imply that every provider implements reconnecting to another server's
  task sandbox.
- Linux task images with Python 3.10+, `venv`, and `pip`. The published runtime
  wheels require glibc 2.28+ on x86-64 or arm64. System Node.js is unnecessary.
- A Gym model server reachable from inside the sandbox and compatible with
  DSH's DeepSeek Chat Completions dialect, including SSE and reasoning/tool calls.

Preinstall `deepseek-harness-sdk==0.1.5rc1` into the sandbox image for repeated
evaluations. Otherwise the runner installs that exact SDK and its matching
runtime wheel into a per-rollout virtual environment outside the task repository;
the image then needs package-index access. `python_executable` can select an
existing Python environment containing the pinned SDK.

Each rollout has its own `DSH_HOME`, session id, and artifacts directory. Gym
provides task-container isolation; the runner sets `DSH_PERMISSION_MODE` to
`danger-full-access` **inside that sandbox** to avoid nested kernel sandbox
requirements and unattended approval prompts.

With `stop_sandbox: true`, the agent closes the sandbox after verification,
exceptions, or cancellation. Resources may also stop it during verification;
cleanup failures are logged without replacing the verification result or original
error. Set it to `false` only when the
resources server owns cleanup, including failed runs. The SDK closes its runtime
on normal completion/errors; execution deadlines and process-tree termination
also depend on the selected sandbox provider.

## Configure and run

Compose `configs/deepseek_harness_agent.yaml` with a model, sandbox provider, and
resources server. Set `model` to the actual model id, not the Gym server instance
name. For example, the existing SWE-bench topology can be configured with:

```bash
gym env start \
  --config responses_api_models/openai_model/configs/openai_model.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --config resources_servers/swebench/configs/swebench.yaml \
  --config responses_api_agents/deepseek_harness_agent/configs/deepseek_harness_agent.yaml \
  ++deepseek_harness_agent.responses_api_agents.deepseek_harness_agent.resources_server.name=swebench_resources_server \
  ++deepseek_harness_agent.responses_api_agents.deepseek_harness_agent.model=YOUR_MODEL_ID \
  '++swebench_resources_server.resources_servers.swebench.allowed_agents=[deepseek_harness_agent]'
```

Set `policy_base_url`, `policy_api_key`, and `policy_model_name` in `env.yaml` or
as CLI overrides, plus the OpenSandbox connection settings as for the existing
`opencode_sandboxed_agent`. Send a benchmark row to this agent's `/run` endpoint;
task metadata is forwarded unchanged to `/seed_session` and `/verify`.
`/responses` is the execution step within that prepared `/run` context.

Use `openai_model` for hosted Chat Completions endpoints to preserve their native
`reasoning_content` fields through Gym's SSE bridge. The `vllm_model` converter
wraps reasoning in `<think>` text, which DSH records as assistant text.

Supported input is one text user task (a string or one user message). Conversation
replay, multimodal input, and caller-supplied tools/instructions are rejected.
DSH owns its prompts and tool roster. `max_output_tokens` overrides the configured
per-model-call `max_tokens`; `timeout_s` bounds sandbox runner execution, including
dependency installation. `max_concurrency` limits active rollouts per server worker.

## Results and validation

The verifier's response/reward is preserved. Additional fields are
`dsh_finish_reason`, `dsh_error`, and `dsh_artifacts`. A model token-limit outcome
remains distinct from a runner error. Review these fields alongside reward: a
partially completed workspace may still receive a verifier score.

The artifact directory contains input configuration, raw SDK notifications in
`events.jsonl`, the runner's `result.json`, and stdout/stderr when available.
Notifications are flushed while running, including child sessions and compaction
events. Artifacts are retrieved before sandbox cleanup, also after execution
errors. A terminated runner may leave a partial final JSONL record; that is
reported as an incomplete event log.

Gym's response contains the committed **root-session** reasoning, assistant text,
tool calls/results, and root usage. Child output does not overwrite the root
response. Cache reads/writes are included in Gym input-token totals; missing
detail counts remain unknown. The raw event log is authoritative for context
injections, compaction, and subagent behavior.

This is an evaluation integration. Training token IDs/logprobs are not synthesized
from text. Gym currently buffers model completions before emitting SSE; align
timeouts with the longest model call. Its `external_staging` training-capture path
rejects streaming requests and needs separate integration work.

```bash
pytest responses_api_agents/deepseek_harness_agent/tests
ruff check responses_api_agents/deepseek_harness_agent
ruff format --check responses_api_agents/deepseek_harness_agent
```

Install `deepseek-harness-sdk==0.1.5rc1` in the test environment to enable the SDK
smokes. Those use the real published runtime and real shell tools with a
deterministic model endpoint, for both profiles. They validate execution and event
translation, not real-model quality or benchmark/provider compatibility. A real
rollout with the intended model and sandbox provider is still required before
claiming benchmark validation.

A real-model smoke on 2026-09-10 passed for both profiles with DeepSeek V4 Flash,
Gym's `openai_model`, and an Enroot task container: read and repair a Python
function, execute tool calls, export the trajectory, and pass independent Python
assertions. This exercises `/responses` on a prepared sandbox; it does not
validate the resource-server `/run` handoff or a benchmark. Enroot itself does
not implement the reconnect capability required by `/run`.

Tracking: [#2577](https://github.com/NVIDIA-NeMo/Gym/issues/2577),
[#2950](https://github.com/NVIDIA-NeMo/Gym/issues/2950).
