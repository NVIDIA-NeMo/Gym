# Spider 2.0-Lite Resource Server

Execution-based text-to-SQL evaluation on Spider 2.0-Lite, a real-world enterprise SQL benchmark.
Binary reward: 1.0 if the predicted SQL produces a result set equivalent to the gold, 0.0 otherwise.
No LLM judge required.

Each row sets `engine`. `sqlite` is the default and runs against a local database file. `bigquery` runs the predicted SQL as a BigQuery job. Both engines use the same result-set comparison.


|                | SQLite                      | BigQuery                                               |
| -------------- | --------------------------- | ------------------------------------------------------ |
| `engine`       | `sqlite` (default)          | `bigquery`                                             |
| Prepared tasks | 135                         | 205                                                    |
| Databases      | 30 local `.sqlite` files    | 69 datasets, queried live                              |
| What is scored | `gold_sql` or `gold_result` | `gold_result` only                                     |
| Credentials    | none                        | application-default credentials with BigQuery Job User |


## SQLite databases

SQLite databases are downloaded lazily on the first SQLite request (~500 MB from Google Drive) and cached at `resources_servers/spider2_lite/.spider2_lite/sqlite/` for subsequent runs. BigQuery-only workloads never trigger the download.

To trigger the download manually:

```python
from resources_servers.spider2_lite.setup_spider2 import ensure_spider2_lite
ensure_spider2_lite()
```

BigQuery does not download a database. The prompt schema is taken from the Spider2 `DDL.csv` files at prepare time, and the predicted SQL runs against live BigQuery.

## Verification flow

1. Extract SQL from model output (searches for a `sql`  block, then a generic code block, then raw SQL).
2. Dispatch on `engine`.
  - `sqlite`: execute the predicted SQL against `{db_id}.sqlite`. If the row has `gold_sql`, execute that too and compare the two result sets. Otherwise compare the predicted rows to `gold_result`.
  - `bigquery`: execute the predicted SQL on BigQuery and compare the rows to `gold_result`. `gold_sql` on the row is not executed. A missing credentials file, a missing quota project, or a missing `google-cloud-bigquery` install returns `failure_reason: bigquery_unavailable` and reward 0.0 for that task. The server stays up.
3. Compare result sets with column-vector matching (the same method as the official Spider 2.0-Lite `evaluate.py`).
4. Return reward 1.0 on match, 0.0 otherwise.



## Input JSONL format

Every row needs `instance_id`, `db_id`, `question`, `engine`, `ignore_order`, and `condition_cols`. `engine` defaults to `sqlite` when omitted. A SQLite row must include `gold_sql` or `gold_result`. A BigQuery row must include `gold_result`.

`gold_result` is `list[list[list]]`: outer list = alternative answer sets, middle = rows, inner = values.

**SQLite,** `gold_sql` (24 of the 135 tasks publish gold SQL; those rows also carry `gold_result`):

```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "You are a SQL expert..."},
      {"role": "user", "content": "<DATABASE_SCHEMA>\n...\n</DATABASE_SCHEMA>\n\n<QUESTION>\n...\n</QUESTION>"}
    ]
  },
  "instance_id": "local022",
  "db_id": "IPL",
  "question": "Retrieve the names of players who scored no less than 100 runs...",
  "engine": "sqlite",
  "gold_sql": "SELECT DISTINCT p.player_name FROM ...",
  "ignore_order": true,
  "condition_cols": []
}
```

**SQLite,** `gold_result` (all 135 tasks):

```json
{
  "responses_create_params": {"input": [...]},
  "instance_id": "local022",
  "db_id": "IPL",
  "question": "...",
  "engine": "sqlite",
  "gold_result": [["SR Watson", "V Kohli"]],
  "ignore_order": true,
  "condition_cols": []
}
```

**BigQuery.** The system prompt asks for BigQuery Standard SQL with tables fully qualified as `project.dataset.table`. Scoring uses `gold_result`. A published `gold_sql` string may be stored on the row and is not executed.

```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "You are a SQL expert. ... Fully qualify tables as `project.dataset.table`."},
      {"role": "user", "content": "<DATABASE_SCHEMA>\n...\n</DATABASE_SCHEMA>\n\n<QUESTION>\n...\n</QUESTION>"}
    ]
  },
  "instance_id": "bq002",
  "db_id": "ga360",
  "question": "...",
  "engine": "bigquery",
  "gold_result": [[["value"]]],
  "ignore_order": true,
  "condition_cols": []
}
```

Committed smoke sets, 5 rows each:


| File                                                         | Contents             |
| ------------------------------------------------------------ | -------------------- |
| `resources_servers/spider2_lite/data/example.jsonl`          | SQLite smoke tasks   |
| `resources_servers/spider2_lite/data/bigquery_example.jsonl` | BigQuery smoke tasks |


`example.jsonl` is the file checked by `gym env test +should_validate_data=true`, so it must keep exactly 5 rows and stay in sync with `example_metrics.json` and `example_rollouts.jsonl`.

## Configuration


| Field                          | Default                                                 | Description                                                                                                                          |
| ------------------------------ | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `spider2_lite_dir`             | `resources_servers/spider2_lite/.spider2_lite`          | Path to cached databases (relative to repo root or absolute)                                                                         |
| `sqlite_max_concurrency`       | `32`                                                    | Max concurrent SQLite executions                                                                                                     |
| `sqlite_execution_timeout_s`   | `30.0`                                                  | Per-query SQLite timeout in seconds                                                                                                  |
| `bigquery_credentials_path`    | `~/.config/gcloud/application_default_credentials.json` | ADC JSON for the BigQuery client. Read only when a row sets `engine: bigquery`                                                       |
| `bigquery_project`             | `null`                                                  | Billing project override in `configs/spider2_lite.yaml`. When `null`, the server reads `quota_project_id` from the credentials file. |
| `bigquery_execution_timeout_s` | `120.0`                                                 | Per query BigQuery timeout in seconds                                                                                                |
| `bigquery_max_concurrency`     | `8`                                                     | Max concurrent BigQuery query jobs                                                                                                   |




## BigQuery credentials

`google-cloud-bigquery` is imported only when a row sets `engine: bigquery`, so a SQLite-only run does not need it.

Create application-default credentials and set the quota project to the GCP project where the identity has **BigQuery Job User**:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project <PROJECT_ID>
```

Jobs are billed to `<PROJECT_ID>`.

`resources_servers/spider2_lite/scripts/run_eval.sh --engine bigquery` exits before the eval if `~/.config/gcloud/application_default_credentials.json` is missing.

## Prepared datasets

`benchmarks/spider2_lite/prepare.py` writes one sidecar per engine and copies the selected one to `benchmarks/spider2_lite/data/spider2_lite_benchmark.jsonl`.


| File                                                                  | Contents                        |
| --------------------------------------------------------------------- | ------------------------------- |
| `benchmarks/spider2_lite/data/spider2_lite_sqlite_validation.jsonl`   | 135 SQLite tasks                |
| `benchmarks/spider2_lite/data/spider2_lite_bigquery_validation.jsonl` | 205 BigQuery tasks              |
| `benchmarks/spider2_lite/data/spider2_lite_combined_validation.jsonl` | 340 tasks, SQLite then BigQuery |


`prepare_dataset.py` skips a BigQuery example that has no gold result file, no eval metadata, or no schema directory.

Rebuild the sidecars:

```bash
python resources_servers/spider2_lite/scripts/prepare_dataset.py \
    --spider2-dir /path/to/Spider2/spider2-lite \
    --output-dir benchmarks/spider2_lite/data \
    --engine all
```

`--engine` is `sqlite`, `bigquery`, or `all` (default). `all` also writes the combined file.

Validate example rows:

```bash
gym dataset collate \
    --resources-server spider2_lite \
    --output-dir /tmp/prepare \
    --mode example_validation
```

To run an eval on one engine, see `benchmarks/spider2_lite/README.md`.

## Known limitations

- 24 of 135 SQLite tasks have publicly available gold SQL. The other 111 are scored from pre-computed `gold_result` rows.
- BigQuery scoring never re-runs gold SQL.

