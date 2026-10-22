# BIRD Benchmark

Execution-based text-to-SQL on BIRD, bound to the `bird_sql` resource server. Two choices define a run:

| Choice | Values | What it controls |
|---|---|---|
| **dialect** | `sqlite` (default), `spark` | the SQL the model writes, the prompt, and the engine that runs it |
| **split** | `dev` (default), `train` | the questions and databases: `dev` is the 1534-task benchmark, `train` is for generating training data |

Reward is binary: unordered result-set equality between the model's query and BIRD's gold query. Metrics are overall
plus per-difficulty (simple / moderate / challenging; `train` has no difficulty labels).

## The three processes

| Process | Command | Needs a model? | Produces |
|---|---|---|---|
| **prepare** | `python -m benchmarks.birdbench.prepare` | no | the questions file: one row per task with `question`, `gt_sql`, and `sql_context` (schema plus example values) |
| **eval** | `eval.sh --model <name>` | yes | `rollouts.jsonl` (one row per rollout) and `rollouts_aggregate_metrics.json` (pass@1, per-difficulty and failure metrics) |
| **collect** | `python -m benchmarks.birdbench.collect collect` | yes | `raw_choices.jsonl` (one row per rollout) and `grouped.jsonl` (one record per task, with logprobs) for training |

- **prepare** runs first and the other two consume its output; they run it for you if the file is missing.
- **eval** measures a model, through Gym's servers. **collect** generates training data, calling your endpoint directly
  because the `vllm_model` server pins `top_logprobs=0`, which hosted endpoints may reject. Both score every answer with
  the same `bird_sql` reward.
- All three take the same `--dialect` and `--split` (the choices above). `prepare` and `eval` default to `dev`;
  `collect` defaults to `train`.

## Installation

None of these are `pyproject.toml` dependencies; install them into the environment you run `gym` from.

```bash
uv pip install bm25s nltk        # needed by every `prepare` (BM25 retrieval for sql_context)
```

**Spark dialect** additionally needs a JDK, plus `pyspark` (runs the model's SQL) and `sqlglot` (checks it is a single
`SELECT`/`WITH`) in the **server's own venv** (`resources_servers/bird_sql/.venv`, created by `gym` on its first
start), not the one you run `gym` from. `prepare` itself needs no Java.

```bash
# macOS (Homebrew); on Linux install any JDK 17 and set JAVA_HOME. JAVA_HOME must be set in the shell that starts servers.
brew install openjdk@17
export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"
uv pip install --python resources_servers/bird_sql/.venv/bin/python "pyspark>=3.5,<4" sqlglot
```

Spark runs in local mode inside the server process; no cluster is needed. Re-run the `uv pip install` line if `gym`
rebuilds the server venv.

## Dialects and splits

| | sqlite | spark |
|---|---|---|
| Prompt | `prompts/default_sqlite.yaml` | `prompts/default_spark.yaml` |
| `sql_context` rendered as | SQLite schema | Spark schema: `` `db_id`.`table` `` names, backtick-quoted identifiers, Spark column types |
| Model's SQL runs on | the SQLite databases | a local PySpark copy (Parquet under `resources_servers/bird_sql/.bird_sql/spark_warehouse[_train]/`, built on first start) |

| | dev | train |
|---|---|---|
| Questions | BIRD `dev.json` | the **uberset**: BIRD `train.json` with every row that ReViSQL re-verified ([`bird-verified-train.json`](https://github.com/uiuc-kang-lab/ReViSQL/blob/main/data/bird-verified-train.json)) replaced by its verified version, matched by `question_id`; same size as `train.json` |
| Databases | BIRD dev, ≈1.4 GB, downloaded on first use | BIRD train, ≈9 GB, downloaded on first use to `resources_servers/bird_sql/.bird_sql/train/` |
| `difficulty` | simple / moderate / challenging | `null` |
| BM25 corpus scan | every distinct value | at most 2000 per column (the databases are multi-GB; only a few example values per column are shown anyway) |

- **Gold SQL is never transpiled.** In both dialects it runs on the original SQLite database; only the two result sets
  are compared (floats rounded to 6 places, dates compared as strings).
- **Spark execution.** ANSI mode is on, as in Spark 4 / Databricks (invalid casts and divide-by-zero raise errors). Only
  a single `SELECT`/`WITH` query is executed; anything else scores as an execution error. Spark has 4 local cores, so
  use a lower server `max_concurrency` than SQLite's 32
  (`+birdbench_bird_sql_resources_server.resources_servers.bird_sql.max_concurrency=8`). A numeric column whose data
  is not cleanly numeric, and date/datetime columns, are `STRING`.
- **Caveat.** Because the gold is the answer under SQLite semantics, a correct Spark query can differ from it when the
  gold relies on SQLite behavior (bare `GROUP BY` columns, integer division, case-insensitive `LIKE`, loose
  text/number comparison). `spark_gold_check.py` estimates about 8% of dev (121 of 1534) are affected.

## Preparing data

`prepare` writes `data/birdbench_<dialect>_<split>.jsonl` (e.g. `birdbench_sqlite_dev.jsonl`), downloading the
databases first if needed:

```bash
python -m benchmarks.birdbench.prepare                                    # dev,   sqlite (the defaults)
python -m benchmarks.birdbench.prepare --dialect spark                    # dev,   spark
python -m benchmarks.birdbench.prepare --split train                      # train, sqlite
python -m benchmarks.birdbench.prepare --split train --dialect spark      # train, spark
```

Other flags: `--train-dir <dir>` to use an already-extracted BIRD `train/` (with `train.json` and `train_databases/`)
instead of downloading, `--no-evidence`, and `--dscp`.

Each row has `question`, `gt_sql`, `sql_context`, `difficulty`, `db_id` and `id`. `sql_context` is a per-table,
per-column schema (type, description, example values) plus a `#### Foreign key` section. Example values combine a
per-column sample with per-question BM25 hits on the question text; see `build_db_values.py`.

- **Evidence.** BIRD's `evidence` (literal-value hints such as "triple type bonds refers to bond_type = '#'") is
  prepended to each question; `--no-evidence` turns that off.
- **Column descriptions** come from `--dscp`, an ordered list of `name`, `col_dscp` (column_description) and `val_dscp`
  (value_description). The default, `name_or_col_dscp`, shows exactly one of name or description (name if non-blank);
  e.g. `--dscp name,col_dscp,val_dscp` shows all three.
- Through `gym eval prepare` these are `prepare_script_args.{dialect,split,dscp,include_evidence}`.

## Files

The dataset is **never repeated**: every file below has one row per task (`dev`: 1534). Several rollouts per task come
from `--num-repeats` only. Do not raise the dataset's `num_repeats` in `config.yaml`: `gym eval run` multiplies the
two, so the prepared files would grow to 1534 x `num_repeats` rows.

| File | Rows | Written by |
|---|---|---|
| `data/birdbench_<dialect>_<split>.jsonl` | 1 per task | prepare |
| `data/bird_verified_train.json` | cached download | prepare, train split |
| `data/..._prepare.jsonl`, `<output>/rollouts/preprocessed_datasets/benchmark.jsonl` | 1 per task | `gym eval run` (applies the prompt) |
| `results/<split>_<dialect>/<model>/rollouts.jsonl`, `rollouts_aggregate_metrics.json` | 1 per task x `--num-repeats` | eval |
| `train_data/<split>_<dialect>/<model>/{raw_choices,grouped}.jsonl` | 1 per rollout, 1 per task | collect |

## Evaluating a model

`eval.sh` runs one model end to end: it prepares the data if missing, runs `gym eval run`, prints a failure
breakdown, and kills leftover server processes on exit.

**The model** is an OpenAI-compatible endpoint that you already serve, e.g. a vLLM server you started (local or on a
cluster node), or a hosted API. Gym does not download or launch it. `--model` is the name that endpoint uses for the
model, `--model-url` its base URL (e.g. `http://localhost:8000/v1`) and `--model-api-key` its key. Instead of flags, set
`policy_model_name`, `policy_base_url` and `policy_api_key` in `env.yaml`; flags override it. `collect` uses the same
`env.yaml` values (or `--policy-model-name`, `--policy-base-url`, `--policy-api-key`).

```bash
./benchmarks/birdbench/eval.sh --model <name>                                 # dev,   sqlite (the defaults)
./benchmarks/birdbench/eval.sh --model <name> --dialect spark                 # dev,   spark
./benchmarks/birdbench/eval.sh --model <name> --split train                   # train, sqlite
./benchmarks/birdbench/eval.sh --model <name> --split train --dialect spark   # train, spark
```

`--limit 5` gives a smoke test and `--num-repeats N` N rollouts per task. Results go to
`benchmarks/birdbench/results/<split>_<dialect>/<model>/` (override with `--output_dir`). The script header documents every option;
other flags pass through to `gym eval run`.

`rollouts.jsonl` has one row per rollout (prompt, response, reward, verifier fields like `extracted_sql` and
`failure_reason`) and no grouped file; `collect.py group` makes one.

## Collecting training data

`collect` runs the model on every task `--num-repeats` times and writes one training record per task:
`{id, db_id, context, completions: [{completion, reward, reference_logprob}]}`, where `reference_logprob` is the
summed per-token logprob of the generation.

```bash
python -m benchmarks.birdbench.collect collect --num-repeats 8                    # train, sqlite (the defaults)
python -m benchmarks.birdbench.collect collect --num-repeats 32                   # more rollouts per task
python -m benchmarks.birdbench.collect collect --num-repeats 8 --dialect spark    # train, spark
python -m benchmarks.birdbench.collect collect --num-repeats 8 --split dev        # dev,   sqlite
python -m benchmarks.birdbench.collect collect --num-repeats 4 --limit 20         # smoke test on 20 tasks
```

Output goes to `benchmarks/birdbench/train_data/<split>_<dialect>/<model>/` (`--output-dir` overrides):
- `raw_choices.jsonl`: one row per rollout, the raw API response, saved before any processing.
- `grouped.jsonl`: one record per task, with all its completions.

Notes:
- A repeat that fails after 5 retries is dropped with a warning, so a task can have fewer than `--num-repeats`
  completions.
- Saved completions are `<think>reasoning</think>answer<stop token>`. `<think>` is added to the saved text only (it is
  never generated, so it has no logprob); the stop token is absent if `--max-tokens` cut the model off.
- `--sql-concurrency` defaults to 16. At 32, threads contending on SQLite's progress handler starved the GIL and rewards
  collapsed (0/32 correct vs 31/32 at 16, same data). Use about 4 for Spark.
- `--dialect spark` needs a JDK and pyspark: run with `resources_servers/bird_sql/.venv`'s python.
- `collect` writes no aggregate metrics; use `eval.sh` for those.

### Rarely needed

```bash
# Re-score saved raw responses, with no API calls (after a fix to the text reconstruction, reward or logprob logic)
python -m benchmarks.birdbench.collect reprocess --output-dir <collect run dir>

# Convert an eval.sh / `gym eval run` rollouts file to the grouped format
# (reference_logprob is set only if return_token_id_information reached the policy model)
python -m benchmarks.birdbench.collect group --rollouts <eval output>/rollouts.jsonl --output grouped.jsonl

# Second pass on the tasks no rollout solved: select them (pooled over any number of grouped files, matched by id),
# then collect again on just those
python -m benchmarks.birdbench.collect select-hard --prepared-jsonl data/birdbench_sqlite_train.jsonl \
    --grouped-jsonl <run dir>/grouped.jsonl --output-jsonl data/birdbench_sqlite_train_hard.jsonl
python -m benchmarks.birdbench.collect collect --input-jsonl data/birdbench_sqlite_train_hard.jsonl --num-repeats 128
```
