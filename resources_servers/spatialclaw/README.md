# SpatialClaw

This environment wraps the canonical SpatialClaw workflow as a NeMo Gym
Responses agent and uses SpatialClaw's benchmark classes as the verifier.
NeMo Gym owns rollout collection and metric reporting; SpatialClaw remains the
source of truth for prompt construction, frame sampling, tools, answer parsing,
and dataset-level scoring.

The implementation has three layers:

1. `prepare_data.py` converts a SpatialClaw benchmark to portable Gym JSONL.
   Videos remain `input_video` side-channel items rather than base64 frame dumps.
2. `responses_api_agents/spatialclaw_agent` decodes and caches frames, then runs
   `SpatialAgentWorkflow` against Gym's policy-model endpoint.
3. This resource server performs native per-sample verification and calls the
   benchmark's `evaluate()` method for aggregate metrics. This preserves
   Video-MME-v2's four-question group scoring.

## Required external inputs

- `SPATIALCLAW_ROOT`: a clean checkout of the original
  [NVlabs/SpatialClaw](https://github.com/NVlabs/SpatialClaw) implementation.
  The adapter targets revision `b062f82962549391a02cc70e022641c1f4d8ac2b`.
  Install its agent dependencies in the agent environment and launch its GPU
  tool service with the matching source and checkpoints.
- `SPATIALCLAW_DATA_ROOT`: the staged benchmark data root.
- A SpatialClaw GPU tool server that provides `Reconstruct` and `SAM3`. Start it
  through SpatialClaw's launcher or co-schedule it in the evaluation job. The
  model endpoint itself is supplied by the selected Gym model server.
- `SPATIALCLAW_FRAME_CACHE_ROOT`: optional persistent shared frame cache. This
  should be on storage visible to all agent workers. Cache keys include the
  video identity and frame-sampling settings, and extraction is file-locked so
  the four Video-MME-v2 questions do not decode the same video concurrently.
- `SPATIALCLAW_WORKSPACE_ROOT`: optional location for per-rollout SpatialClaw
  logs. Workspaces are retained by default for score-reproduction audits.

Model-specific `mm_processor_kwargs` are not enabled implicitly. Put them in
the pinned SpatialClaw model config or set the agent's
`video_mm_processor_kwargs` override when a serving stack requires them.

The original `SpatialAgentWorkflow` owns planning, Python execution, tools,
feedback, and answer parsing. Gym supplies the policy endpoint and aiohttp
transport beneath SpatialClaw's LLM client, including clients injected into
Jupyter kernels. One rollout runs per agent process because upstream prompt
and tool modules share a configuration singleton.

The benchmark entries use upstream `videomme.json` and `videommev2.json` and
their original defaults: 32 keyframes, 1 FPS, the code executor, planning,
Reconstruct, and SAM3. Any sampling or tool overrides define a different
evaluation protocol and must be recorded with its results.

SpatialClaw remains an external dependency under its own upstream license;
its source and model/tool weights are not vendored into Gym.

Do not mark the environment `verified: true` until the SpatialClaw checkout is
committed and pinned, the exact model/dataset configs are recorded, and Gym
scores reproduce an accepted canonical SpatialClaw run from the same raw
predictions.

## Metrics

Each rollout receives SpatialClaw's native `evaluate_single()` reward. During
aggregation, Gym runs `evaluate()` separately for every rollout repeat and
reports both per-repeat and mean native metrics under
`native/<dataset-config>/...`.

For a partial Video-MME-v2 run, only complete canonical four-question groups
are aggregated. The `coverage` metric reports what fraction of the full dataset
entered that repeat's native aggregate.
