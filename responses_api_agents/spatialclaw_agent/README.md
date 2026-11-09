# SpatialClaw agent

**Internal-only experimental integration.** The pinned SpatialClaw source is hosted
on NVIDIA GitLab and requires authorized repository access. Its `LICENSE` §3.3
restricts use to non-commercial scientific research. Gym's Apache-2.0 license
does not change those external terms. No native source is vendored here. Public
availability and compatible licensing require a maintainer/licensor decision;
this integration is not a publicly reproducible or certified benchmark.

This is an evaluation-only NeMo Gym adapter for the multi-step SpatialClaw visual-spatial agent. It keeps Gym's standard rollout and verifier contract while SpatialClaw owns the planning, persistent Jupyter kernel, visual queries, and Reconstruct/SAM3 tool loop.

The adapter pins SpatialClaw commit `946ac114dfcabf9df997629bfa8b6f2f66da1425`, the same commit used by `third_party/spatial_claw` on the SpatialClaw VLMEvalKit branch. It clones that revision into an ignored source cache on first use. To reuse an existing checkout instead, set `SPATIALCLAW_ROOT`; startup verifies that its `HEAD` is exactly the pinned commit.

All policy calls go through the configured Gym `policy_model` server. The adapter replaces SpatialClaw's native OpenAI transport with Gym's shared aiohttp transport, including calls made by `vlm.locate` and `vlm.ask_with_thinking` inside the Jupyter kernel.

## Runtime prerequisites

- `ffmpeg` or the `imageio-ffmpeg` fallback must be available for video frame extraction.
- Reconstruct and SAM3 remain external SpatialClaw GPU services. Launch them from the same pinned checkout, or set `SPATIALCLAW_GPU_SERVER_REGISTRY` to that checkout's live `spatial_agent/logs/gpu_server.json`.
- Set `SPATIALCLAW_DATA_ROOT` to the pinned checkout's shared benchmark data directory.
- The default role configuration is `nemotron-3-nano-omni-30b-a3b-reasoning-spatialclaw-think65536`; set `SPATIALCLAW_MODEL_CONFIG` to select another model-role JSON from the pinned checkout.
- Keep agent-level `concurrency` at 1 because the pinned SpatialClaw revision uses a process-global configuration singleton. Scale with separate Gym agent replicas instead of concurrent rollouts in one process.

Prepared Gym rows pass their prompt and portable media metadata to the selected
SpatialClaw dataset configuration. The adapter supports image paths, grouped images,
reference images, and single- or multi-video paths. Scoring and aggregate metrics are
proxied to the preset's Gym resource server. The complete suite lives under
`benchmarks/spatialclaw`; the agent itself is not tied to Video-MME.

This adapter intentionally does not expose SpatialClaw's shell/Pyxis matrix runner
through Gym. Gym owns row scheduling, retry/resume behavior, verification, and
aggregation.

## Agent sessions

The session path is an in-process native SpatialClaw harness at the Evaluation
capability level. It accepts exactly one required `direct_http` grant to a
SpatialClaw Resources session; it rejects borrowed sandboxes, required MCP grants,
and continuation requirements with terminal 422 responses. Optional unsupported
MCP grants are skipped. All session workers use `num_workers: 1`.

The Environment Server owns the episode lifecycle. The adapter does not seed,
verify or close Resources during activation. It reads Resources' canonical task
context and runs the original native graph with a remote connection to the
resource-owned Python kernel. Model calls, including kernel VLM queries, retain
the Gym rollout route. Native force-termination/fallback semantics remain part of
the native harness; the final native prediction is committed to Resources through
the trusted direct HTTP connection before verification.

One activation is allowed per session. Identical retries share or replay its task;
conflicting bodies, wrong rollout URLs and unknown/closed session cookies return
409. A disconnected HTTP caller does not cancel shared work. Close cancels pending
work within its cleanup budget and returns observations, or an explicit gap when
activation produced none. It disconnects from the kernel; Resources owns final
kernel shutdown. Infrastructure/model outcomes use Gym's `AgentOutcome` contract
so the Environment Server selects retry, failure or verification.

Sessionless `/v1/responses` remains a local development path; compatibility `/run`
requires an explicit Resources binding. Neither path provides execution isolation.
The session integration requires shared filesystem access between agent and
Resources. It does not support training/token-history capture or crash-resumable
native workflow state. A worker restart must restart the episode.
