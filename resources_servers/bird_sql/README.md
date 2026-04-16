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

## Verification flow

1. Extract SQL from the model output (last ` ```sql ... ``` ` code block, with
   comments stripped and whitespace collapsed — matches the NeMo Skills BIRD
   evaluator's `CODEBLOCK` extraction path).
2. Execute the predicted SQL against the per-`db_id` SQLite database.
3. Execute the ground-truth SQL against the same database.
4. Compare result sets via unordered set equality.
5. Return reward `1.0` on match, `0.0` otherwise (or on any extraction /
   execution error).

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

These map directly to the NeMo Skills BIRD metrics (`total_acc`, `simple_acc`,
`moderate_acc`, `challenging_acc`).
