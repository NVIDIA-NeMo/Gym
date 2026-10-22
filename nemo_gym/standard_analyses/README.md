# Standard evaluation analyses

Gym computes these reports after rollout collection and logs their derived metrics, tables and charts to the configured W&B run. Reports also live beside the output JSONL in `<stem>_analyses/`, including a standalone HTML preview. They work with `+upload_rollouts=false`. Set `+standard_analyses=false` to opt out. Shards with `disable_aggregation=true` defer analysis; analyze their canonical combined output after collection.

For model-call and tool measurements, enable observability in the shared configuration used by both the servers and collector:

```yaml
observability_enabled: true
model_call_capture_dir: /path/shared/by/gym/processes/model-calls
```

Analysis coverage reflects the evidence captured by the selected model and agent adapters.

Install `nemo-gym[wandb]` to export to W&B. Configure `wandb_project`, `wandb_name`, `wandb_api_key` and, optionally, `wandb_entity` through the normal Gym config. The API key determines the publishing account; `wandb_entity` selects the destination team.

Create the project workspace once, then select one run for detail or multiple runs for comparison:

```bash
python -m nemo_gym.standard_analyses workspace \
  --entity YOUR_TEAM --project YOUR_PROJECT --output /tmp/analysis-workspace --save
```

The template includes charts and detail tables for long calls, excess effort across repeats, effort outliers, token accumulation, potentially redundant actions, action phases, compaction, pass@k, system flags, runtime tails and model/tool time. The workspace reads logged results; computation happens in Gym. Unknown measurements remain unavailable, with coverage and interpretation notes in the exported tables.

Each benchmark/agent/model cohort has its own report. For a collection containing several cohorts, the manifest supplies a distinct `namespace`; pass `--namespace` when creating a workspace for that cohort. No comparisons cross cohort boundaries. Charts based on detail rows inherit their row limits; aggregate counts use all eligible records.

For an existing local file:

```bash
python -m nemo_gym.standard_analyses analyze rollouts.jsonl --format gym \
  --run-id my-run --benchmark my-benchmark --config-id my-config --output /tmp/analysis
```

Add an analysis through a Gym PR: implement a function returning `AnalysisResult`, register it in `suite.py`, add chart specifications in `dashboard.py`, and add numerical, missing-data and export-binding tests under `tests/unit_tests/standard_analyses/`. Keep transcript contents, credentials and file contents out of exported results. Repeat excess is an observed comparison, action flags are inspection candidates, and compaction comparisons are descriptive rather than causal.
