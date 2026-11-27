# BIRD SQL Resource Server

Execution-based text-to-SQL verification on the BIRD dev split.

Binary reward: `1.0` if the predicted SQL produces the same result set as the
ground-truth query on the per-`db_id` SQLite database, `0.0` otherwise. No LLM
judge.

## Dataset

- **1534 tasks** across 11 SQLite databases from the BIRD dev split
- Each task has a `difficulty` label: `simple`, `moderate`, or `challenging`

## Database download

SQLite databases are downloaded automatically on first server startup:

```
https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
```

They are cached at `resources_servers/bird_sql/.bird_sql/dev_20240627/dev_databases/`
and re-used on subsequent runs.

To trigger the download manually:

```python
from resources_servers.bird_sql.setup_bird_sql import ensure_bird_sql
ensure_bird_sql()
```

## Spark SQL dialect (experimental)

Set `dialect: spark` for a model that writes Spark SQL. Its query runs on a local PySpark
copy of the databases, while the BIRD gold query still runs, untouched, on the original
SQLite database; only the two result sets are compared (floats rounded, dates as strings).
No gold SQL is transpiled. Only a single `SELECT`/`WITH` model query is executed (anything
else scores as a pred error) so the shared warehouse cannot be modified, and a timed-out
query has its Spark job cancelled.

On first startup each SQLite database is written to Parquet under
`<bird_sql_dir>/spark_warehouse/` and registered as a Spark database named after its
`db_id`. Requires a JDK 17 plus `pyspark`, `sqlglot`, `pandas` and `pyarrow` (see
`benchmarks/birdbench/README.md`); the default SQLite dialect needs none of them.

## PostgreSQL dialect (experimental)

Set `dialect: postgres` for a model that writes PostgreSQL. Its query runs on a PostgreSQL
server that holds one schema per `db_id`, while the BIRD gold query still runs, untouched,
on the original SQLite database; only the two result sets are compared, as for Spark. No
gold SQL is transpiled. The model's query runs as a single prepared statement (so
`COMMIT; ...` cannot leave the transaction) in a read-only transaction as a read-only role,
with a server-side `statement_timeout`. A connection or server failure is masked as a
verifier error (`unknown_error`), not scored as the model's mistake.

Each server process keeps a pool of `max_concurrency` connections (opened on the first
grade), so the PostgreSQL server needs `max_connections` above `max_concurrency` times the
number of server processes. Every transaction rolls back, so nothing a query sets carries
over to the next grade on that connection. Waiting for a connection does not count against
the query's timeout.

The grader connects with `postgres_dsn`, a read-only role. When `postgres_admin_dsn` is set,
startup loads any missing `db_id` from the SQLite files (Spark's column typing; text columns
use the `"C"` collation so ordering matches SQLite) and creates that role from
`postgres_dsn`'s user and password. Without it, startup only checks that the data is
loaded. Tables keep BIRD's names; column names are lower-cased, so an unquoted column
name works as it does in SQLite, while mixed-case table names (17 tables in 4 dev
databases) still need double quotes. Each loaded `db_id` records its layout version, and
startup stops with an error on one loaded with mixed-case column names until it is dropped
and loaded again. `psycopg` and `psycopg_pool` are dependencies of this server (`pyproject.toml`); the default SQLite
dialect never imports them.

## Example usage

### Running servers

```bash
gym env start \
    --model-type vllm_model \
    --resources-server bird_sql
```

Requires `policy_base_url` / `policy_api_key` / `policy_model_name` in
`env.yaml` (or passed as CLI overrides).

### Collecting rollouts (5-example smoke test)

```bash
gym eval run --no-serve \
    --agent bird_sql_simple_agent \
    --input resources_servers/bird_sql/data/example.jsonl \
    --output results/bird_sql_rollouts.jsonl \
    --num-repeats 1
```

For a full BIRD dev run, see `benchmarks/birdbench/README.md`.

## Verification flow

1. Extract SQL from the model output (last ` ```sql ... ``` ` code block, with
   comments stripped and whitespace collapsed).
2. Execute the predicted SQL against the per-`db_id` SQLite database.
3. Execute the ground-truth SQL against the same database.
4. Compare result sets via unordered set equality.
5. Return reward `1.0` on match, `0.0` otherwise (no SQL block, a result
   mismatch, or a predicted query that fails or times out).

When the verifier itself cannot measure the answer (the database file is missing
or corrupt, the ground-truth query fails or times out, the executor crashes), the
row keeps the model's answer and is masked instead of scored: `mask_sample=true`,
`failure_kind=verifier_error`, with `failure_reason` naming the cause
(`database_missing`, `gold_execution_error`, `gold_execution_timeout`,
`unknown_error`). These are `failure_reason` values, not failure kinds in the
registry.

## Input JSONL format

Each task must provide:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "<question + schema dump>"}
    ]
  },
  "question": "...",
  "gt_sql": "SELECT ...",
  "sql_context": "CREATE TABLE ...;",
  "db_id": "california_schools",
  "difficulty": "simple",
  "id": 0
}
```

`question` / `gt_sql` / `db_id` are used by the server; `difficulty` / `id`
are passed through to metrics and output.

## Metrics

Overall:

- `pass@k/accuracy` — fraction of tasks with at least one correct rollout in k tries
- `pass@1[avg-of-k]/accuracy` — mean accuracy across k rollouts, averaged over tasks

Per-difficulty subsets (via `compute_subset_metrics(field="difficulty")`):

- `simple/pass@1[avg-of-k]/accuracy`
- `moderate/pass@1[avg-of-k]/accuracy`
- `challenging/pass@1[avg-of-k]/accuracy`
