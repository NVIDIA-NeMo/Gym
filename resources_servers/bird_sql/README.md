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

## Catalog tools

`catalog_app.py` is a second entrypoint: `bird_sql` plus three tools over
[Auto Ontology](https://github.com/NVIDIA/auto-ontology)'s catalog of the BIRD databases, served as `direct_http`
routes. `benchmarks/birdbench/catalog_tools.yaml` uses it.

| Route | Auto Ontology function it calls |
|---|---|
| `POST /retrieve` (`query`, `kinds`) | `react.retrieve.retrieve` |
| `POST /run_sql` (`sql`) | `react.run_sql.run_sql` |
| `POST /submit` (`sql`, `response`, `assumptions`) | `react.loop.submit` |

**One sandbox per episode.** At the seed the server starts a sandbox from `sandbox_config.image` and starts the
catalog's Postgres in it. It also starts the episode's tool process, `catalog_tool_server.py`, which the sandbox gets
through `SandboxSpec.files`. That process keeps one Auto Ontology `ToolContext` for the episode, so a tool call sees
what earlier calls found. Each route call runs `catalog_tool_server.py call` in the sandbox. `verify` stops the
sandbox, and so does closing the session or shutting the server down.

**Grading.** `verify` grades the SQL of the last `submit` call that the server answered with `Accepted.`. A query in
the answer's text is not graded. Grading reads only the rollout, so a saved rollout can be graded again without a
sandbox.

**The catalog image.** The image must have:

- a Python with Auto Ontology installed (`catalog_python`);
- a folder whose `.env` sets `CONNECTION_STRINGS` to the BIRD databases the catalog was built from, and the embedding
  endpoint and model it used (`catalog_workdir`);
- a command that starts the catalog's Postgres (`catalog_start_command`), with the `POSTGRES_*` settings in the
  image's environment.

The defaults match the image that
[auto-ontology-eval](https://github.com/NVIDIA/auto-ontology-eval)'s `scripts/build_agent_image.sh` builds, which
ingests and compiles the BIRD dev databases with Auto Ontology's own pipeline. `catalog_environment` passes settings
that do not belong in an image, such as the embedding endpoint's key (`DEFAULT_MODELS_API_KEY`).

| Setting | Default | Meaning |
|---|---|---|
| `sandbox_provider`, `sandbox_config` | required | The sandbox of each episode: a provider block of the global config, and the image |
| `max_tool_calls` | 24 | Tool calls an episode may make. Past it, only `submit` is served |
| `setup_timeout_s`, `tool_timeout_s`, `cleanup_timeout_s` | 300, 120, 60 | Limits on the seed, each tool call, and stopping the sandbox |
| `sandbox_ttl_s` | 7200 | How long the provider keeps a sandbox that was never stopped |
