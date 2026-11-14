# Spider 2.0-Lite benchmark

The benchmark scores the same resource server on one of three dataset selections. `prepare()` writes a per-engine JSONL sidecar and copies the one for `<engine>` to `benchmarks/spider2_lite/data/spider2_lite_benchmark.jsonl`, which is the file the Hydra config reads.

| `<engine>` | Tasks | Credentials |
|---|---|---|
| `sqlite` (default) | 135 | none |
| `bigquery` | 205 | gcloud application-default credentials. See the resource server README |
| `all` | 340 (SQLite, then BigQuery) | same as `bigquery` |

## Prepare

```bash
gym eval prepare --benchmark spider2_lite \
    "++prepare_script_args={engine:'<engine>'}"
```

`<engine>` has to be the split you intend to score. A later `gym eval run` with `++reuse_existing_data_preparation=true` reads whatever is already in `spider2_lite_benchmark.jsonl` and does not switch engines.

## Run

The output filename matches `resources_servers/spider2_lite/scripts/run_eval.sh`: `results/benchmarks/spider2_lite_<model-slug>_<engine>.jsonl`, where `<model-slug>` is the model name with `/` replaced by `_`. Including both the model and the engine keeps repeat runs from clobbering each other.

SQLite, after a `sqlite` prepare:

```bash
gym eval run \
    --model-type vllm_model \
    --benchmark spider2_lite \
    --output results/benchmarks/spider2_lite_<model-slug>_sqlite.jsonl \
    --split benchmark \
    --model-url <> \
    --model-api-key <> \
    --model <> \
    --resume \
    ++overwrite_metrics_conflicts=true \
    ++spider2_lite_benchmark_resources_server.resources_servers.spider2_lite.sqlite_max_concurrency=8 \
    ++reuse_existing_data_preparation=true
```

BigQuery, after a `bigquery` prepare. Create credentials first (`gcloud auth application-default login` and `gcloud auth application-default set-quota-project <PROJECT_ID>`). The server runs at most `bigquery_max_concurrency` query jobs at once (default 8).

```bash
gym eval run \
    --model-type vllm_model \
    --benchmark spider2_lite \
    --output results/benchmarks/spider2_lite_<model-slug>_bigquery.jsonl \
    --split benchmark \
    --model-url <> \
    --model-api-key <> \
    --model <> \
    --resume \
    ++overwrite_metrics_conflicts=true \
    ++spider2_lite_benchmark_resources_server.resources_servers.spider2_lite.bigquery_max_concurrency=8 \
    ++reuse_existing_data_preparation=true
```

`resources_servers/spider2_lite/scripts/run_eval.sh --engine sqlite|bigquery|all` does this in one step: it prepares the selected engine, checks for the BigQuery credentials file when the engine needs it, and writes the output path above.

```bash
resources_servers/spider2_lite/scripts/run_eval.sh \
    --engine bigquery \
    --model-url <> \
    --model-api-key <> \
    --model <>
```

`--smoke` runs the committed 5-row example file for that engine instead of the full validation set. `--engine all` concatenates the two example files.
