# BIRD Benchmark

Execution-based text-to-SQL on BIRD dev, bound to the `bird_sql` resource server.

- **Tasks**: 1534 across 11 SQLite databases
- **Reward**: binary; unordered result-set equality on the per-`db_id` DB
- **Metrics**: overall + per-difficulty (simple / moderate / challenging)
  via `compute_subset_metrics(field="difficulty")`

## Preparation

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/birdbench/config.yaml]"
```

This downloads the BIRD `dev.zip` (≈1.4 GB) via
`resources_servers.bird_sql.setup_bird_sql.ensure_bird_sql()`, dumps each
database schema with truncated INSERTs via `sqlite3.Connection.iterdump()`,
and writes `data/birdbench_benchmark.jsonl`.

The output schema matches `nemo_skills/dataset/birdbench/prepare.py`
(`question`, `gt_sql`, `sql_context`, `difficulty`, `db_id`, `id`).

## Prompt

`prompts/default.yaml` is a char-for-char copy of
`nemo_skills/prompt/config/generic/text_to_sql.yaml` so Skills and Gym runs
send the same prompt.
