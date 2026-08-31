# BIRD Benchmark

Execution-based text-to-SQL on BIRD dev, bound to the `bird_sql` resource server.

- **Tasks**: 1534 across 11 SQLite databases
- **Reward**: binary; unordered result-set equality on the per-`db_id` DB
- **Metrics**: overall + per-difficulty (simple / moderate / challenging)
  via `compute_subset_metrics(field="difficulty")`

## Preparation

```bash
pip install bm25s nltk
gym eval prepare --benchmark birdbench
```

This downloads the BIRD `dev.zip` (≈1.4 GB) via
`resources_servers.bird_sql.setup_bird_sql.ensure_bird_sql()` and writes
`data/birdbench_benchmark.jsonl`. Each row has `question`, `gt_sql`,
`sql_context`, `difficulty`, `db_id`, and `id`.

`sql_context` is a per-table, per-column schema (data type, description,
example values), followed by a `#### Foreign key` section. Descriptions come
from BIRD's `database_description/<table>.csv`, aligned to real column names
via `original_column_name`. Example values combine a baseline per-column
sample with per-question BM25 retrieval (via `bm25s`, pure Python, no JVM)
against the question text. See `build_db_values.py` for the full retrieval
and rendering design.

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
    --prompt-config benchmarks/birdbench/prompts/default.yaml \
    --output results/birdbench_rollouts.jsonl \
    --temperature 0 \
    --num-repeats 4
```

`--no-serve` skips the data-preparation step that normally applies
`prompt_config` automatically, so `--prompt-config` must be passed explicitly
here -- otherwise rows won't have `responses_create_params` and collection
will fail with `KeyError: 'responses_create_params'`.

For a 5-example smoke test against the resource server's `example.jsonl`,
see `resources_servers/bird_sql/README.md`.

## Prompt

`prompts/default.yaml`: the model reasons step-by-step and returns the final
SQL inside a ` ```sql ``` ` block. Explains the `sql_context` format (the
`#### Tables`/`#### Foreign key` structure, per-column `data_type`/
`description`/`values`, and the `TYPE (primary key)` marker) before the
schema itself, so the model has the format legend in context before reading
the actual data.
