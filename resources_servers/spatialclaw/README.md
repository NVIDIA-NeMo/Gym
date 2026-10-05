# SpatialClaw resource server

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
