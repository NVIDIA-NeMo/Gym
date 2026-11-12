# BIRD SQL Agentic Resources Server

Multi-turn text-to-SQL on BIRD dev. The model gets a read-only `execute_sql` tool for the task's
SQLite database, so it can inspect tables, check values, and fix errors before giving its final
answer in a `sql` code block. Scoring is the same as [`bird_sql`](../bird_sql/README.md): the last
`sql` block is executed and compared with the gold query by unordered result-set equality.

## The `execute_sql` tool

- Runs on a read-only connection (`mode=ro`), so model-generated SQL cannot change the database.
- Accepts one `SELECT`, `WITH`, or `PRAGMA table_info(...)` statement per call.
- Stops queries after `sql_execution_timeout_s` inside SQLite itself (progress handler).
- Returns up to `max_result_rows` rows (cells cut to `max_cell_chars`) plus the total row count.
- Returns SQL errors as the tool result instead of raising, so the model can fix its query.
- Is bound to the task's `db_id` in `/seed_session`; the model cannot choose another database.

Verify output adds `num_tool_calls` and `num_sql_errors`; aggregate metrics add their means.

## Example usage

The BIRD databases are shared with `bird_sql` and are downloaded automatically on first server startup (see [`bird_sql`](../bird_sql/README.md#database-download)).

```bash
gym env test --resources-server bird_sql_agentic
gym env start --resources-server bird_sql_agentic --model-type openai_model
gym eval run --no-serve --agent bird_sql_agentic_simple_agent \
    --input resources_servers/bird_sql_agentic/data/example.jsonl \
    --output results/bird_sql_agentic_example.jsonl
```

`data/example.jsonl` is produced from `bird_sql`'s examples by `create_examples.py`, which only swaps
the system prompt and adds the tool declaration.

## Limitations

- **No BIRD evidence.** Like `bird_sql`, the prompt does not include BIRD's per-question
  `evidence` hints, so questions that depend on a domain formula are harder than in the official
  BIRD setting.
- **Strict scoring.** Exact result-set matching marks some essentially correct answers as wrong,
  for example when the query returns an extra column or uses a precomputed column whose values
  were rounded differently. The verifier is kept identical to `bird_sql` so results stay comparable.
- **Little incentive to explore.** The prompt still contains the full schema and sample rows
  inherited from `bird_sql`, so untrained models often answer after a single tool call or none.

## Future work

- `include_evidence` option to add BIRD evidence to the prompt (off by default).
- `schema_only` prompt option without sample rows, so inspecting the data requires the tool.
- A `tested_final_query` metric: whether the model ran its final SQL with the tool before answering.
- A diagnostic `relaxed_match` metric that ignores extra columns and small rounding differences,
  to measure false negatives. The reward would still use exact matching.

## Licensing

- Code: Apache 2.0
- Data: BIRD dev, Creative Commons Attribution-ShareAlike 4.0 International
