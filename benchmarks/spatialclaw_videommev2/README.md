# SpatialClaw Video-MME-v2

This benchmark runs Video-MME-v2 through the canonical SpatialClaw agent
workflow and its group-aware native scorer. It uses
`spatial_agent/config/dataset/videommev2.json` (32 keyframes, 1 FPS
cap, Reconstruct + SAM3).

Set `SPATIALCLAW_ROOT` to the pinned SpatialClaw checkout and stage
Video-MME-v2 under `SPATIALCLAW_DATA_ROOT` before preparing or running it.

The dataset config requires `Reconstruct` and `SAM3`, so a compatible
SpatialClaw GPU tool server must be reachable through its normal
`gpu_server.json` registry. Gym serves the policy model; it does not replace the
SpatialClaw tool server. Keep `SPATIALCLAW_FRAME_CACHE_ROOT` on persistent
shared storage because four questions share each video and the benchmark runs
three repeats.

```bash
gym eval prepare --benchmark spatialclaw_videommev2
gym eval run --model-type vllm_model --benchmark spatialclaw_videommev2 \
  --split benchmark --output results/spatialclaw_videommev2.jsonl
```

Native aggregate metrics are emitted under
`native/videommev2/`. The official headline is
`final_rating/total`; `overall_accuracy` is the same grouped rating normalized
to `[0, 1]`, not ordinary per-question accuracy. Partial runs aggregate only
complete canonical four-question groups and expose their dataset `coverage`.

This config remains `verified: false` until the approved SpatialClaw commit,
model config, GPU-tool images/checkpoints, and accepted reference score are
frozen and a full three-repeat parity run is attached.
