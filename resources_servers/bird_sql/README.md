# BIRD SQL Resource Server

Execution-based text-to-SQL verification on the BIRD dev split.

Binary reward: `1.0` if the predicted SQL produces the same result set as the
ground-truth query on the per-`db_id` SQLite database, `0.0` otherwise. No LLM
judge. Both queries run in a verifier sandbox, never on the resources server (see
[Verifier sandbox](#verifier-sandbox)).

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

## Verifier sandbox

With the default SQLite dialect, every grade starts a fresh sandbox from `sandbox_provider`, runs the gold query
and then the model's query in it, and stops it. The sandbox runs `sandbox_runner.py` with an uploaded copy of
`eval_utils.py` (through `run_verification_sandbox`), so a query is executed, timed out and compared by the same code
as before; this server only reads the verdict. The database is opened read-only, so a query that writes
(`DROP TABLE`) fails as the model's error instead of changing the database later grades read.

- **Provider.** `sandbox_provider: sandbox` names a top-level block from a provider config, for example
  `nemo_gym/sandbox/providers/docker/configs/docker.yaml`; pass it with `--config`. The server refuses to start
  without it.
- **Image and database.** `sandbox_config.image` (default `python:3.13.14-slim-trixie`) needs Python 3.11+ with
  `sqlite3`; its SQLite version decides which SQL functions a query can call, so keep it pinned. The row's database
  is uploaded into each sandbox (dev's largest is about 600 MB). To skip the upload, set
  `sandbox_config.databases_dir` to a directory of an image, or of a read-only provider mount, that holds
  `<db_id>/<db_id>.sqlite`.
- **Limits.** No network: `provider_options.network_policy` denies all egress on OpenSandbox and E2B; on Docker set
  `network: none` in the provider block. Each query keeps `sql_execution_timeout_s`. `sandbox_config.memory_limit_mib`
  caps the runner's address space, so a query that needs more memory fails as the model's error instead of killing
  the sandbox. `max_concurrency` bounds the sandboxes running at once.
- **Failures.** A sandbox that cannot start, run the runner or report a result is retried on a fresh sandbox
  `inconclusive_verification_retries` times (default 2). If every attempt fails, the row is masked with
  `failure_reason: sandbox_error`, `evaluation_completed: false` and `failure_kind` `provider_unavailable` or
  `verifier_error`, and counted in the `sandbox_error` metric. Nothing falls back to running the query on the server.

The Spark and PostgreSQL dialects below are not sandboxed yet: the model's query runs on their engines.

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
    --resources-server bird_sql \
    --config nemo_gym/sandbox/providers/docker/configs/docker.yaml
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
2. Start a verifier sandbox with the per-`db_id` SQLite database.
3. In the sandbox, execute the ground-truth SQL, then the predicted SQL, against that database.
4. Compare result sets via unordered set equality.
5. Return reward `1.0` on match, `0.0` otherwise (no SQL block, a result
   mismatch, or a predicted query that fails or times out).

When the verifier itself cannot measure the answer (the database file is missing
or corrupt, the ground-truth query fails or times out, the executor crashes), the
row keeps the model's answer and is masked instead of scored: `mask_sample=true`,
`failure_kind=verifier_error`, with `failure_reason` naming the cause
(`database_missing`, `gold_execution_error`, `gold_execution_timeout`,
`unknown_error`). A verifier sandbox that fails on every attempt is masked the
same way with `sandbox_error` (see [Verifier sandbox](#verifier-sandbox)). These
are `failure_reason` values, not failure kinds in the registry.

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
