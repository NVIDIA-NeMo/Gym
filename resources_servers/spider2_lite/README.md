# Spider 2.0-Lite Resource Server

Execution-based text-to-SQL evaluation on Spider 2.0-Lite, a real-world enterprise SQL benchmark.
Binary reward: 1.0 if the predicted SQL produces a result set equivalent to the gold, 0.0 otherwise.
No LLM judge required.

## Dataset

- **135 SQLite tasks** from the local subset of Spider 2.0-Lite
- 30 enterprise databases (IPL, Pagila, California_Traffic_Collision, etc.)
- 24 tasks have publicly available gold SQL; all 135 have pre-computed gold result CSVs

## Database download

SQLite databases are downloaded automatically on first server startup (~500 MB from Google Drive).
They are cached at `resources_servers/spider2_lite/.spider2_lite/sqlite/` and re-used on subsequent runs.

To trigger the download manually:

```python
from resources_servers.spider2_lite.setup_spider2 import ensure_spider2_lite
ensure_spider2_lite()
```

## Verification flow

1. Extract SQL from model output (searches for ` ```sql ``` ` block, then generic code block, then raw SQL)
2. Execute predicted SQL against the SQLite database
3. Compare result set against gold using column-vector matching (mirrors official Spider 2.0-Lite `evaluate.py`)
4. Return reward 1.0 on match, 0.0 otherwise

## Input JSONL format

Each line must include either `gold_sql` (execute-and-compare) or `gold_result` (pre-computed rows):

**`gold_sql` mode** (24 tasks with public gold SQL):
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
  "gold_sql": "SELECT DISTINCT p.player_name FROM ...",
  "ignore_order": true,
  "condition_cols": []
}
```

**`gold_result` mode** (all 135 tasks using pre-computed CSVs):
```json
{
  "responses_create_params": {"input": [...]},
  "instance_id": "local022",
  "db_id": "IPL",
  "question": "...",
  "gold_result": [["SR Watson", "V Kohli"]],
  "ignore_order": true,
  "condition_cols": []
}
```

`gold_result` is `list[list[list]]`: outer list = alternative answer sets, middle = rows, inner = values.

## Configuration

| Field | Default | Description |
|---|---|---|
| `spider2_lite_dir` | `resources_servers/spider2_lite/.spider2_lite` | Path to cached databases (relative to repo root or absolute) |
| `max_concurrency` | `32` | Max concurrent SQL executions |
| `sql_execution_timeout_s` | `30.0` | Per-query timeout in seconds |

## Dataset preparation

Validate example data:
```bash
ng_prepare_data "+config_paths=[resources_servers/spider2_lite/configs/spider2_lite.yaml]" \
    +output_dirpath=/tmp/prepare +mode=example_validation
```

## Known limitations

- Only 24 of 135 local tasks have publicly available gold SQL. The remaining 111 tasks require pre-computed gold result CSVs (embedded in the train/validation JSONL via `gold_result`).
- SQLite databases must be downloaded on first run; no offline support.
