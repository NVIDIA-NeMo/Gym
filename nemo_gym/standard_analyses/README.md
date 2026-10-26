# Standard evaluation analyses

Gym writes long-call, repeat-excess, effort-outlier and token-accumulation diagnostics after collection and health checks. Reports live beside the output JSONL in `<stem>_analyses/` and export to configured W&B and MLflow runs, including with `+upload_rollouts=false`. Set `+standard_analyses=false` to opt out.

Enable observability in the shared configuration used by both the servers and collector:

```yaml
observability_enabled: true
model_call_capture_dir: /path/shared/by/gym/processes/model-calls
```

Automatic diagnostics run only when canonical trajectories contain model calls. Shards collected with `disable_aggregation=true` defer analysis to `gym eval aggregate`, which also supports `merge_shards=false`.

Install `nemo-gym[wandb]` and configure the normal `wandb_project`, `wandb_name`, `wandb_api_key` and optional `wandb_entity` settings. Create the project workspace once:

```bash
gym eval analysis-workspace --entity YOUR_TEAM --project YOUR_PROJECT
```

Select one run for details or several runs for comparison. For collections with several benchmark/agent/model/configuration cohorts, the manifest lists each report's namespace; supply `--namespace efficiency/cohorts/HASH` when creating its workspace. Comparisons stay within a cohort. A rollout without calls inherits the collection's unambiguous policy identity; auxiliary model calls remain part of that policy's effort.

Unhealthy and masked rollouts are excluded from effort diagnostics. Unobserved health remains explicitly unknown. Availability reflects the evidence each analysis requires; unrelated observation gaps and health exclusions do not downgrade complete measurements. Repeat excess requires complete capture and a successful baseline, and describes observed differences rather than recoverable time.

The pass@k chart reads the environment's existing aggregate metrics, preserving percentages, continuous scores and configured failure/missing-rollout accounting. It does not rescore health-filtered rows or publish a second set of scalar scores. It is unavailable when the environment supplies no pass@k values or its aggregates span several analysis cohorts. Existing `perf_summary` values also export under `perf_summary/<environment label>/`.

Chart tables are explicit report data, with row counts and truncation flags. Aggregate metrics use all eligible records. Add analyses through a Gym PR: implement an `AnalysisResult` function, register it in `suite.py`, define its charts in `dashboard.py`, and add numerical, coverage and export-binding tests. Keep transcript contents and credentials out of reports.
