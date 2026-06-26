# BIRD Benchmark

Execution-based text-to-SQL on BIRD dev, bound to the `bird_sql` resource server.

- **Tasks**: 1534 across 11 SQLite databases
- **Reward**: binary; unordered result-set equality on the per-`db_id` DB
- **Metrics**: overall + per-difficulty (simple / moderate / challenging)
  via `compute_subset_metrics(field="difficulty")`

## Preparation

```bash
gym eval prepare --benchmark birdbench
```

This downloads the BIRD `dev.zip` (≈1.4 GB) via
`resources_servers.bird_sql.setup_bird_sql.ensure_bird_sql()`, dumps each
database schema with truncated INSERTs via `sqlite3.Connection.iterdump()`,
and writes `data/birdbench_benchmark.jsonl`. Each row has
`question`, `gt_sql`, `sql_context`, `difficulty`, `db_id`, and `id`.

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --benchmark birdbench
```

Requires `policy_base_url` / `policy_api_key` / `policy_model_name` in
`env.yaml` (or passed as CLI overrides).

## Collecting rollouts

```bash
gym eval run --no-serve \
    --agent birdbench_bird_sql_simple_agent \
    --input benchmarks/birdbench/data/birdbench_benchmark.jsonl \
    --output results/birdbench_rollouts.jsonl \
    --num-repeats 4
```

For a 5-example smoke test against the resource server's `example.jsonl`,
see `resources_servers/bird_sql/README.md`.

## Prompt

Standard text-to-SQL prompt in `prompts/default.yaml`: the model reasons
step-by-step and returns the final SQL inside a ` ```sql ``` ` block.
