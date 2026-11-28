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
