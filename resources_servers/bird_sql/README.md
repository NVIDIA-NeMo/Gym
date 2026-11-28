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

## Snowflake dialect (experimental, local stand-in)

Set `dialect: snowflake` for a model that writes Snowflake SQL. By default there is no
Snowflake account behind it: the query runs on [fakesnow](https://github.com/tekumara/fakesnow),
which transpiles Snowflake SQL with `sqlglot` and runs it on DuckDB, over a read-only copy of
the databases (`<snowflake_db_dir>/BIRD.db`, one schema per `db_id`, built on first start in
about 16 s for dev). As for Spark and PostgreSQL, the BIRD gold query still runs, untouched,
on the original SQLite database, and only the two result sets are compared. Tables and
columns keep BIRD's original names, created quoted, so they are case-sensitive as they would
be on Snowflake.

DuckDB is more lenient than Snowflake in places, so before running the query the grader
refuses, as Snowflake would, anything but a single query, identifiers that do not resolve
under Snowflake's rules (an unquoted name means its upper-case spelling; a column must belong
to its own table, and an alias doesn't cover a column inside its own definition), a bare
function in `FROM` (DuckDB's `range()` or `query()`; Snowflake wraps table functions in
`TABLE(...)`), and a few SQLite-only spellings (`IIF`, `INSTR`, `GROUP_CONCAT`, `STRFTIME`,
`JULIANDAY`, `TOTAL`, `LIMIT offset, count`), `SUM`/`AVG` of a boolean expression (including
an `IFF`, `CASE` or `COALESCE` of booleans), a `SELECT DISTINCT` ordered by something it
doesn't select, an aggregate `FILTER (WHERE ...)` clause, and six functions DuckDB has and
Snowflake doesn't (`DIVIDE`, `EPOCH_MS`, `LIST_VALUE`, `REGEXP_MATCHES`, `STRING_AGG`,
`STRPTIME`). This check is static and approximate; on a real account, Snowflake refused every
query it refuses (see the calibration below). It matches function names as written: a quoted
name (`"DIVIDE"(a, b)`) or a call in a query with a CTE of the same name (`WITH divide AS
(...) SELECT DIVIDE(a, b)`) gets past it, and a table alias with a column list that has one of
these names (`FROM t AS divide (a, b)`) is refused although Snowflake accepts it. Known
differences from Snowflake that the check does not cover, each checked on an account:

- `DATEADD` and `DATE_TRUNC` on a date return a timestamp, so the result gains `00:00:00`;
  `TO_CHAR` ignores its format when `sqlglot` cannot infer the argument's type (a
  subquery's column, a timestamp).
- `DATEDIFF` and `DATEADD` on BIRD's `VARCHAR` date columns fail; Snowflake casts the string.
  `YEAR`, `MONTH`, `EXTRACT` and `DATE_PART` on a `VARCHAR` fail on Snowflake too, and an
  explicit `TO_DATE(...)` or `::DATE` works on both.
- DuckDB refuses implicit text-to-number conversion (`'2026' - '1990'`, `varchar_col > 0`)
  where Snowflake casts row by row and errors only on text that isn't a number (3 of 11 dev
  golds that compare text with a number match on Snowflake).
- Wrong Snowflake SQL that runs: other functions DuckDB has and Snowflake doesn't, `TO_CHAR` of
  a `VARCHAR` with a format, division by zero (it returns `inf` instead of an error), `SUM`/`AVG`
  of another boolean-valued function, and a `SELECT DISTINCT` ordered by an aggregate. The
  static check leaves the last two alone rather than risk refusing SQL Snowflake accepts.

A query that fails on the stand-in scores as the model's error; a failure of the stand-in
itself (a closed connection, an I/O error) is masked as a verifier error (`unknown_error`).
Queries run single-threaded so float sums are repeatable. A query's timeout counts from when
it starts, not from when it waited for a thread or a connection; a timed-out query is
interrupted, and its slot stays taken until its thread has stopped. A result over 5,000,000
rows (18 times the largest dev gold result) isn't converted and scores as the query's failure.
DuckDB's file and network access is off and locked, so model SQL can't read files through
table functions such as `read_csv` or `sqlite_scan`. `fakesnow` and
`snowflake-connector-python` are dependencies of this server (`pyproject.toml`), pinned to
`fakesnow==0.12.1` because the grader reads two of its private attributes; the default SQLite
dialect never imports them.

`snowflake_engine: snowflake` runs the same grader against a real account instead, through
`snowflake_connection_name` in the connector's `connections.toml` (no credentials in this
config), with the data already loaded in the same layout. Each connection runs
`USE SECONDARY ROLES NONE`, so only the connection's own role applies: a user's default
secondary roles are often `ALL`, which would let a session on a read-only role write through
the user's other roles. That read-only role is also what contains `TABLE(...)` functions the
static check lets through, such as `TABLE(INFORMATION_SCHEMA.QUERY_HISTORY())`. Each query
opens its own connection, so it logs in each time: use key-pair (or another non-interactive)
authentication, and expect about a second of extra latency per query; that time doesn't count
toward the query's timeout. At startup, the grader checks that its role can see every table of
every database, and stops if one is missing.

To set up an account:

1. As an account admin, create a database `BIRD`, a warehouse (X-Small, with auto-suspend) and
   a read-only role for the grader, granted to the user the grader connects as. Grant that
   role `USAGE` on the warehouse: with secondary roles off, it can't borrow another role's.
2. Add two connections to the connector's `connections.toml`: an admin one, with the warehouse
   and a role that owns `BIRD`, and the grader's, with the read-only role and the warehouse.
3. Load the data and grant the grader's role read access:
   `resources_servers/bird_sql/.venv/bin/python -m benchmarks.birdbench.snowflake_load_account
   <admin connection> --grant-to <role>` (`--split train` for train). Dev takes about 7
   minutes. Run it again after any reload: FUTURE grants need MANAGE GRANTS, which a
   database owner such as SYSADMIN usually lacks, so the script grants on the tables itself.
4. Serve with `snowflake_engine: snowflake` and `snowflake_connection_name: <grader connection>`.

**Calibration on a real account** (BIRD dev, 2026-11-19, with
`benchmarks/birdbench/snowflake_gold_check.py --snowflake-connection`). sqlglot's translations
of the 1,534 dev golds match on 1,439 on Snowflake and on 1,413 on the stand-in. For the 1,532
whose gold runs on SQLite, the two agree on pass or fail for 1,498 (97.8%). 30 pass only on
Snowflake, mostly arithmetic on text that Snowflake casts (20). 4 pass only on the stand-in:
ties under `LIMIT` (2), float precision (1) and a text value cast to a number (1). Snowflake
refused every query the static check refuses, and the check refuses none of the 2,681
distinct queries Snowflake ran in the calibration (gold translations, probes and a model's
answers).

On a model's own SQL (Nemotron 3.5 Lightning with thinking on, all of dev, 2026-11-20), the
stand-in and Snowflake agree on pass or fail for 1,523 of 1,532 questions (99.4%). 8 pass only
on the stand-in: 5 ratios where Snowflake's fixed-point division changes the sixth decimal,
which is where the grader compares (a precision limit, not something the check can see), 2
ties under `LIMIT` and 1 date format Snowflake parses strictly. 1 passes only on Snowflake
(arithmetic on text that holds a number).

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
