# SpatialClaw Video-MME

This benchmark runs Video-MME through the canonical SpatialClaw agent workflow
and native SpatialClaw scorer. It uses
`spatial_agent/config/dataset/videomme.json` (32 keyframes, 1 FPS
cap, Reconstruct + SAM3).

Set `SPATIALCLAW_ROOT` to the pinned SpatialClaw checkout and stage Video-MME
under `SPATIALCLAW_DATA_ROOT` before preparing or running the benchmark.

The dataset config requires `Reconstruct` and `SAM3`, so a compatible
SpatialClaw GPU tool server must be reachable through its normal
`gpu_server.json` registry. Gym serves the policy model; it does not replace the
SpatialClaw tool server. Keep `SPATIALCLAW_FRAME_CACHE_ROOT` on persistent
shared storage to avoid decoding each video again for the three repeats.

```bash
gym eval prepare --benchmark spatialclaw_videomme
gym eval run --model-type vllm_model --benchmark spatialclaw_videomme \
  --split benchmark --output results/spatialclaw_videomme.jsonl
```

Native aggregate metrics are emitted under
`native/videomme/`. The headline is `overall_accuracy`; duration,
domain, and task-type breakdowns are retained. Per-rollout SpatialClaw logs are
kept under `SPATIALCLAW_WORKSPACE_ROOT` for comparison with the canonical
runner.

This config remains `verified: false` until the approved SpatialClaw commit,
model config, GPU-tool images/checkpoints, and accepted reference score are
frozen and a full three-repeat parity run is attached.
