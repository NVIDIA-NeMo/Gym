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
- All three take the same `--dialect` and `--split` (the choices above).

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
| Prompt | `prompts/default.yaml` | `prompts/spark.yaml` |
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

`prepare` writes one jsonl per (split, dialect) pair, downloading the databases first if needed:

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
| `data/birdbench_<dialect>_benchmark.jsonl` (dev), `data/birdbench_train_<dialect>.jsonl` (train) | 1 per task | prepare |
| `data/bird_verified_train.json` | cached download | prepare, train split |
| `data/..._prepare.jsonl`, `<output>/rollouts/preprocessed_datasets/benchmark.jsonl` | 1 per task | `gym eval run` (applies the prompt) |
| `<output>/rollouts.jsonl`, `rollouts_aggregate_metrics.json` | 1 per task x `--num-repeats` | eval |
| `results/collect_<split>_<dialect>_<model>/{raw_choices,grouped}.jsonl` | 1 per rollout, 1 per task | collect |

### Prompts

`prompts/default.yaml` has the model reason step by step and return the final SQL in a ` ```sql ``` ` block. It
explains the `sql_context` format (`#### Tables` / `#### Foreign key`, per-column `data_type`/`description`/`values`,
the `TYPE (primary key)` marker) before the schema. `prompts/spark.yaml` is the same prompt naming Spark SQL as the
engine, with no Spark-specific hints: the point is to test the model's own Spark SQL knowledge.

## Evaluating a model

`eval.sh` runs one model end to end: it prepares the data if missing, runs `gym eval run`, prints a failure
breakdown, and kills leftover server processes on exit. Set the endpoint in `env.yaml` (`policy_base_url` /
`policy_api_key` / `policy_model_name`) or pass `--model-url` / `--model-api-key`.

```bash
./benchmarks/birdbench/eval.sh --model <name>                                 # dev,   sqlite (the defaults)
./benchmarks/birdbench/eval.sh --model <name> --dialect spark                 # dev,   spark
./benchmarks/birdbench/eval.sh --model <name> --split train                   # train, sqlite
./benchmarks/birdbench/eval.sh --model <name> --split train --dialect spark   # train, spark
```

`--limit 5` gives a smoke test and `--num-repeats N` N rollouts per task. Results go to
`results/<model>[_<dialect>][_<split>]/` (override with `--output_dir`). The script header documents every option;
other flags pass through to `gym eval run`.

`rollouts.jsonl` has one row per rollout (prompt, response, reward, verifier fields like `extracted_sql` and
`failure_reason`) and no grouped file; `collect.py group` makes one.

### Driving gym directly

Everything `eval.sh` sets up is a Hydra override, needed for any combination other than dev + sqlite:

```bash
DIALECT=spark SPLIT=train
PREPARED=benchmarks/birdbench/data/birdbench_${SPLIT}_${DIALECT}.jsonl   # dev: birdbench_${DIALECT}_benchmark.jsonl
PROMPT=benchmarks/birdbench/prompts/${DIALECT}.yaml                      # sqlite: default.yaml
BIRD=birdbench_bird_sql_resources_server.resources_servers.bird_sql
OVERRIDES=(
    "+prepare_script_args.dialect=${DIALECT}" "+prepare_script_args.split=${SPLIT}"
    "+${BIRD}.dialect=${DIALECT}" "+${BIRD}.split=${SPLIT}"
    "+${BIRD}.spark_warehouse_dir=resources_servers/bird_sql/.bird_sql/spark_warehouse_train"   # train + spark only
    "++birdbench_bird_sql_simple_agent.responses_api_agents.simple_agent.datasets=[{name:birdbench,type:benchmark,jsonl_fpath:${PREPARED},prompt_config:${PROMPT},prepare_script:benchmarks/birdbench/prepare.py,num_repeats:1}]"
)

gym eval prepare --benchmark birdbench "${OVERRIDES[@]}"
gym eval run --benchmark birdbench --model-type vllm_model --split benchmark \
    --output results/birdbench_rollouts.jsonl --temperature 1 --num-repeats 1 "${OVERRIDES[@]}"
```

Hydra cannot override one list item, so the whole `datasets` list is replaced; `gym eval prepare` fails with a
`ConfigError` unless `prepare` writes exactly the dataset's `jsonl_fpath`. To start the servers yourself, use
`gym env start --model-type vllm_model --benchmark birdbench "${OVERRIDES[@]}"`, then `gym eval run --no-serve
--agent birdbench_bird_sql_simple_agent --input ${PREPARED} --prompt-config ${PROMPT} --output ... --num-repeats 1`.
`--no-serve` skips the step that applies the prompt, so `--prompt-config` is required, otherwise rows lack
`responses_create_params` and collection fails with `KeyError: 'responses_create_params'`.

`gym eval run` stops each server by killing only its launcher shell, which leaves the real server (and, for Spark, its
~2 GB JVM) running as an orphan. `eval.sh` reaps those on exit; if you call `gym` directly, look for entries with parent
PID 1 (`ps -axo pid,ppid,command | grep "\.venv/bin/python app.py"`) and `kill` them. For a 5-example smoke test against
the server's own `example.jsonl`, see `resources_servers/bird_sql/README.md`.

## Collecting training data

`collect.py` writes one training record per task: `{id, db_id, context, completions: [{completion, reward,
reference_logprob}]}`, where `reference_logprob` is the summed per-token logprob of the generation.

```bash
python -m benchmarks.birdbench.collect collect --split train --dialect sqlite --num-repeats 32 --limit 20   # drop --limit for all
python -m benchmarks.birdbench.collect reprocess --split train --output-dir <collect run dir>
python -m benchmarks.birdbench.collect group --rollouts <eval.sh output>/rollouts.jsonl --output grouped.jsonl
python -m benchmarks.birdbench.collect select-hard --prepared-jsonl data/birdbench_train_sqlite.jsonl \
    --grouped-jsonl <run dir>/grouped.jsonl --output-jsonl data/birdbench_train_sqlite_hard.jsonl
python -m benchmarks.birdbench.collect collect --split train --input-jsonl data/birdbench_train_sqlite_hard.jsonl \
    --num-repeats 128                  # second pass on the tasks nobody solved
```

- **`collect`** writes `results/collect_<split>_<dialect>_<model>/` (`--output-dir` overrides) with `raw_choices.jsonl`
  (the raw API response per rollout, saved before any processing) and `grouped.jsonl` (per task). A repeat that fails
  after 5 retries is dropped with a warning, so a task can have fewer than `--num-repeats` completions.
- **`reprocess`** rebuilds `grouped.jsonl` from `raw_choices.jsonl` with no API calls, for after a fix to the text
  reconstruction, reward or logprob logic.
- **`group`** converts an `eval.sh` / `gym eval run` rollouts file; `reference_logprob` is set only if
  `return_token_id_information` reached the policy model.
- **`select-hard`** keeps the prepared rows that no collected rollout solved, pooled over any number of grouped files,
  matched by `id`.
- **Saved completions** are `<think>reasoning</think>answer<stop token>`. `<think>` is added to the saved text only
  (it is never generated, so it has no logprob); the stop token is absent if `--max-tokens` cut the model off.
- **Concurrency.** `--sql-concurrency` defaults to 16: at 32, threads contending on SQLite's progress handler starved
  the GIL and rewards collapsed (0/32 correct vs 31/32 at 16, same data). Use about 4 for Spark.
- **Spark** needs a JDK and pyspark: run with `resources_servers/bird_sql/.venv`'s python.

`collect` writes no aggregate metrics; use `eval.sh` for those.
