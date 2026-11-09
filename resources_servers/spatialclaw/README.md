# SpatialClaw resource server

**Internal-only experimental integration.** The pinned SpatialClaw source is hosted
on NVIDIA GitLab and requires authorized repository access. Its `LICENSE` §3.3
restricts use to non-commercial scientific research. Gym's Apache-2.0 license
does not change those external terms. No native source is vendored here. Public
availability and compatible licensing require a maintainer/licensor decision;
this integration is not a publicly reproducible or certified benchmark.

This resource server loads the benchmark evaluator selected by a pinned
SpatialClaw dataset configuration. It uses `evaluate_single` for per-row
verification and the benchmark's full `evaluate` method for aggregate metrics.

Set `SPATIALCLAW_ROOT` to the pinned source checkout and
`SPATIALCLAW_DATA_ROOT` to its shared `data/` directory. Benchmark presets
under `benchmarks/spatialclaw` provide `dataset_config` and compose this
server with the shared SpatialClaw agent.

All 20 suite configurations use this resource server, including Video-MME and
Video-MME-v2. Benchmark-specific extraction and aggregation remain inside the
evaluator selected by SpatialClaw's native `BenchmarkFactory`.

Native metrics require `num_repeats: 1`; larger repeat counts are rejected.
Video-MME-v2 is evaluated in native dataset order, retaining only complete
four-question groups. If a row is masked or absent, its entire group is excluded.
`spatialclaw/num_evaluated_groups` and `spatialclaw/num_dropped_groups` report
coverage relative to the loaded native dataset, including groups outside a
requested subset. Row-level pass metrics still cover every measured row.
Timeouts, invalid model requests, and agent execution errors score zero; only
clear infrastructure failures are masked. Model HTTP failures are journaled
before native nodes catch them, including calls made in Jupyter kernels.

## Shared task sessions

`single_agent_turn` owns the episode lifecycle. Typed Resources seeds prepare the
native sample's media and initialize a persistent native Python kernel without
calling the policy model. The native SpatialClaw adapter uses a private HTTP
kernel protocol; OpenCode uses the same task through the `get_context` and
`execute` MCP tools. Initial images and later visual feedback use Gym's
`tool_output_images` contract. Ground-truth answers are never injected into this
context. Direct stateless verification remains available for historical rollouts.

The native adapter finalizes the prediction selected by its native workflow,
including its existing fallback behavior. OpenCode submits through
`ReturnAnswer(...)` in the task kernel; final assistant prose does not replace a
submission. A response marked `harness_step_limit_reached` scores zero by default,
even if an earlier cell submitted an answer. Set `score_step_limit_answer: true`
only for an explicitly different budget policy. The measured limit-hit fraction
is reported as `spatialclaw/step_limit_reached`.

Optional `sandbox_provider`, `sandbox_config`, `sandbox_image`,
`sandbox_workdir`, and `sandbox_network_access` settings lend a Resources-owned
CLI sandbox to OpenCode. Only sampled frames, reference images, and an answer-free
`task.json` are staged in a unique session child of `sandbox_workdir`. Resources
removes that child and stops the sandbox at close and shutdown; the harness only
borrows it. `model_and_tools` permits the model-server and this
resource server's hosts, while `inherit` keeps the provider's network settings.
The local development profile runs native Python and GPU-service calls on the
resource-server host; lending a CLI sandbox does not isolate that Python kernel.
The native AST checks are execution guards, not a hostile-code security boundary.

All auxiliary VLM requests use Gym's rollout-prefixed model route. If the Resources
runtime made such calls, the returned response leaves usage unknown and carries
`spatialclaw_resource_usage_unknown: "true"`, because the harness's own token
counts cover only its own model calls. Evaluation execution does not establish
training token-history compatibility or harness-conformance certification.

Automatic `gym eval reverify` is unsupported for this session-based track: the
accepted prediction belongs to the Resources session and may differ from the
harness's final prose. Saved rollouts retain `prediction` and native metrics;
responses and token history are preserved verbatim. The direct `/verify` API
still accepts historical native response bodies for explicit offline scoring.
