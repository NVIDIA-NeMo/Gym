# BIRD Benchmark

Execution-based text-to-SQL on BIRD dev, bound to the `bird_sql` resource server.

- **Tasks**: 1534 across 11 SQLite databases
- **Reward**: binary; unordered result-set equality between the model's query and the BIRD gold query
- **Metrics**: overall + per-difficulty (simple / moderate / challenging)
  via `compute_subset_metrics(field="difficulty")`
- **Dialects**: the model writes `sqlite` (default) or `spark` SQL; `eval.sh --dialect` switches
  (see [SQL dialects](#sql-dialects))
- **Splits**: `dev` (default; the 1534-task benchmark) or `train` (for generating training data);
  `eval.sh --split` switches (see [Splits](#splits))
- **Rollouts**: one per task by default (1534). The dataset is never repeated
  (see [Files and row counts](#files-and-row-counts))

## Quick start

```bash
# SQLite (default)
./benchmarks/birdbench/eval.sh --model <name> --output_dir benchmarks/birdbench/results/<name>

# Spark SQL (after the Spark setup in Installation)
./benchmarks/birdbench/eval.sh --model <name> --dialect spark --output_dir benchmarks/birdbench/results/<name>-spark
```

Add `--limit 5` for a smoke test (5 different tasks). `eval.sh` is a thin wrapper over `gym eval prepare` /
`gym eval run`: it prepares the data for the chosen dialect if missing, runs the evaluation, and cleans up leftover
server processes when it exits. Its header documents every option (`--max-output-tokens`, `--temperature`,
`--num-repeats`, `--concurrency`, `--model-url`, ...), and `eval_models.sh` runs it for a list of models. Any other
flag is passed through to `gym eval run`.

## Installation

None of the packages below are `pyproject.toml` dependencies, so install them
separately into the environment you run `gym` from.

**SQLite (default dialect)** -- BM25 retrieval for `sql_context`:

```bash
uv pip install bm25s nltk
```

**Spark SQL dialect** -- needs a JDK, plus `pyspark` (executes the model's SQL) and `sqlglot` (checks that the
model's SQL is a single `SELECT`/`WITH` query) in the **server's own venv**, not the one you run `gym` from. The
servers run in `resources_servers/bird_sql/.venv`, which `gym` creates on its first start. `prepare` needs only
`bm25s`, `nltk` (above) plus `pandas` and `pyarrow` (already installed with Gym), and no Java.

```bash
# macOS (Homebrew). openjdk@17 is keg-only, so point JAVA_HOME at it instead of relying on a symlink.
# JAVA_HOME must be set in the shell that starts the servers (they inherit it).
brew install openjdk@17
export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"
# Linux: install any JDK 17 (e.g. `apt install openjdk-17-jdk`) and set JAVA_HOME.

uv pip install --python resources_servers/bird_sql/.venv/bin/python "pyspark>=3.5,<4" sqlglot
"$JAVA_HOME/bin/java" -version   # sanity check; should report 17
```

Spark runs in local mode inside the server process, so no cluster or daemon is needed. If `gym` rebuilds the server
venv (for example after a Gym upgrade), re-run the `uv pip install` line.

## SQL dialects

`config.yaml` is the SQLite default. A dialect changes four things together, which is what `eval.sh --dialect`
sets up for you:

| Dialect | Prepared dataset | Prompt | The model writes | The server executes the model's SQL on |
|---|---|---|---|---|
| `sqlite` (default) | `data/birdbench_sqlite_benchmark.jsonl` | `prompts/default.yaml` | SQLite SQL | the SQLite databases |
| `spark` | `data/birdbench_spark_benchmark.jsonl` | `prompts/spark.yaml` | Spark SQL | a local PySpark copy of the databases |

Dataset files are always named `data/birdbench_<dialect>_benchmark.jsonl`, so dialects never overwrite each other.

To drive `gym` directly instead of `eval.sh`, pass the same overrides to `gym eval prepare`, `gym env start` and
`gym eval run` (nothing is needed for SQLite):

```bash
DIALECT=spark
OVERRIDES=(
    "+prepare_script_args.dialect=${DIALECT}"
    "+birdbench_bird_sql_resources_server.resources_servers.bird_sql.dialect=${DIALECT}"
    "++birdbench_bird_sql_simple_agent.responses_api_agents.simple_agent.datasets=[{name:birdbench,type:benchmark,jsonl_fpath:benchmarks/birdbench/data/birdbench_${DIALECT}_benchmark.jsonl,prompt_config:benchmarks/birdbench/prompts/${DIALECT}.yaml,prepare_script:benchmarks/birdbench/prepare.py,num_repeats:1}]"
)
```

They set, in order: what `prepare` renders and which file it writes; which engine the server runs; and the dataset
(Hydra cannot override one list item, so the whole `datasets` list is replaced). `gym eval prepare` fails with a
`ConfigError` unless `prepare` writes exactly the dataset's `jsonl_fpath`.

How the two dialects differ:

- **Gold SQL is never transpiled.** In both dialects it runs on the original SQLite database, and only the two result
  sets are compared (unordered set equality; floats rounded to 6 places, dates compared as strings).
- **Prompt and schema.** `prepare` renders `sql_context` for the dialect. For Spark: Spark column types, `` `db_id`.`table` ``
  names (the `db_id` is the Spark schema) and backtick-quoted identifiers. A numeric column whose SQLite data is not
  cleanly numeric is shown (and stored) as `STRING`; date/datetime columns are `STRING`, as in SQLite.
  `prompts/spark.yaml` names Spark SQL as the engine and describes the schema format. It gives no Spark-specific
  hints, because the point is to test the model's own Spark SQL knowledge.
- **Spark execution.** ANSI mode is on, as in Spark 4 / Databricks (invalid casts and divide-by-zero raise errors).
  Only a single `SELECT`/`WITH` model query is executed; anything else scores as an execution error. On first start
  the server writes each SQLite database to Parquet under `resources_servers/bird_sql/.bird_sql/spark_warehouse/`
  (a minute or two) and reuses it afterwards. Spark runs on 4 local cores, so you may want a lower server
  `max_concurrency` than SQLite's 32
  (`+birdbench_bird_sql_resources_server.resources_servers.bird_sql.max_concurrency=8`).
- **Caveat.** Because the gold is the answer under SQLite semantics, a correct Spark query can occasionally return a
  different result when the gold relies on SQLite-specific behavior (bare `GROUP BY` columns, integer division,
  case-insensitive `LIKE`, loose text/number comparison). `spark_gold_check.py` estimates how many tasks are affected
  (about 8%, 121 of 1534, differ or error when the gold is mechanically transpiled to Spark with ANSI mode on).

## Splits

`eval.sh --split dev|train` (default `dev`) picks which BIRD split's questions and databases are used. It combines
freely with `--dialect`.

| Split | Questions | Databases | Prepared dataset |
|---|---|---|---|
| `dev` (default) | BIRD `dev.json` (1534 tasks) | BIRD dev (≈1.4 GB download) | `data/birdbench_<dialect>_benchmark.jsonl` |
| `train` | the "uberset" (below), one task per `train.json` row | BIRD train (≈9 GB download) | `data/birdbench_train_<dialect>.jsonl` |

- **The uberset** is BIRD's `train.json` with every row that ReViSQL re-verified
  ([`bird-verified-train.json`](https://github.com/uiuc-kang-lab/ReViSQL/blob/main/data/bird-verified-train.json))
  replaced by its verified version, matched by `question_id`. It has the same size as `train.json`. Train rows carry
  no `difficulty` (it is `null`).
- **Databases** are downloaded on first use by `ensure_bird_train()` (to `resources_servers/bird_sql/.bird_sql/train/`).
  To use an already-extracted copy, call `prepare(split="train", train_dir=<dir with train.json and train_databases/>)`.
- **Corpus cap.** Train databases are multi-GB, so `prepare` scans at most 2000 distinct values per column for BM25
  retrieval there (`TRAIN_CORPUS_LIMIT_PER_COLUMN`); dev scans everything.
- **Server.** `eval.sh --split train` sets the `bird_sql` server's `split: train`, which makes it verify against
  `train_databases`, and (for Spark) gives train its own `spark_warehouse_train/` so it does not mix with dev's.
- **Never repeated, as for dev.** Use `--num-repeats` for several rollouts per task.

When driving `gym` directly, add `+prepare_script_args.split=train`, `+birdbench_bird_sql_resources_server.resources_servers.bird_sql.split=train`
and a `datasets` override pointing at the train file (see `eval.sh` for the exact list).

### Leftover processes

`gym eval run` stops each server by killing only its launcher shell, which leaves the real server (and, for
`bird_sql`, its ~2 GB Spark JVM) running as an orphan. `eval.sh` reaps orphaned servers of this repo when it exits. If
you call `gym` directly, look for entries with parent PID 1 and `kill` them:
`ps -axo pid,ppid,command | grep "\.venv/bin/python app.py"`.

## Preparation

Dataset prep needs `bm25s` and `nltk` (see Installation above):

```bash
gym eval prepare --benchmark birdbench                          # SQLite
gym eval prepare --benchmark birdbench "${OVERRIDES[@]}"        # other dialect (OVERRIDES: see SQL dialects)
```

This downloads the BIRD `dev.zip` (≈1.4 GB) via
`resources_servers.bird_sql.setup_bird_sql.ensure_bird_sql()` and writes
`data/birdbench_<dialect>_benchmark.jsonl`. Each row has `question`, `gt_sql`,
`sql_context`, `difficulty`, `db_id`, and `id`.

`sql_context` is a per-table, per-column schema (data type, description,
example values), followed by a `#### Foreign key` section. Descriptions come
from BIRD's `database_description/<table>.csv`, aligned to real column names
via `original_column_name`. Example values combine a baseline per-column
sample with per-question BM25 retrieval (via `bm25s`, pure Python, no JVM)
against the question text. See `build_db_values.py` for the full retrieval
and rendering design.

Each column's description is built from `prepare_script_args.dscp`: a
comma-separated, ordered list of BIRD's `column_name` (`name`),
`column_description` (`col_dscp`), and `value_description` (`val_dscp`)
fields, concatenated as prose (blank fields dropped, a period appended to
each kept field unless it already ends with one). E.g. `dscp: name,col_dscp`
includes both fields, name first. The default, `name_or_col_dscp`, is a
single fallback field -- name if non-blank, else description -- matching
this benchmark's original (pre-`dscp`) behavior of showing exactly one of
the two, never both. Override from the CLI with, e.g.:

```bash
gym eval prepare --benchmark birdbench "+prepare_script_args.dscp='name,col_dscp,val_dscp'"
```

By default, each entry's BIRD `evidence` field (literal-value hints, e.g.
"triple type bonds refers to bond_type = '#'") is prepended to its question.
Set `prepare_script_args.include_evidence: false` (pre-declared in
`config.yaml`) to disable this, e.g.:

```bash
gym eval prepare --benchmark birdbench +prepare_script_args.include_evidence=false
```

## Files and row counts

The dataset is **never repeated**: `num_repeats` on the dataset in `config.yaml` is 1, so every file below has one
row per task.

| File | Rows | Written by |
|---|---|---|
| `data/birdbench_<dialect>_benchmark.jsonl` | 1534 | `prepare` / `gym eval prepare` |
| `data/birdbench_<dialect>_benchmark_prepare.jsonl` and `<output>/rollouts/preprocessed_datasets/benchmark.jsonl` | 1534 | `gym eval run`'s data-preparation step (applies the prompt) |
| `<output>/rollouts.jsonl` | 1534 x `--num-repeats` (default 1, so 1534) | the rollout collector |

`--num-repeats N` on `gym eval run` (or `eval.sh`) is the only way to take several rollouts per task. Do not raise the
dataset's `num_repeats` instead: `gym eval run` multiplies the two, and the prepared files then grow to
1534 x `num_repeats` rows (about 600 MB at 16). Files from runs made before this was set to 1 (24,544 rows) are stale
and safe to delete; the next run rewrites them.

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --benchmark birdbench \
    "${OVERRIDES[@]}"      # omit for SQLite (OVERRIDES: see SQL dialects)
```

Requires `policy_base_url` / `policy_api_key` / `policy_model_name` in
`env.yaml` (or passed as CLI overrides).

## Collecting rollouts

`eval.sh` (see Quick start) is the easiest way. To drive `gym` directly:

```bash
gym eval run --benchmark birdbench --model-type vllm_model --split benchmark \
    --output results/birdbench_rollouts.jsonl \
    --temperature 1 --num-repeats 1 \
    "${OVERRIDES[@]}"      # omit for SQLite
```

Or against servers you started yourself with `--no-serve`, which skips the data-preparation step that normally
applies `prompt_config` automatically, so `--prompt-config` must be passed explicitly here -- otherwise rows won't
have `responses_create_params` and collection will fail with `KeyError: 'responses_create_params'`:

```bash
gym eval run --no-serve \
    --agent birdbench_bird_sql_simple_agent \
    --input benchmarks/birdbench/data/birdbench_sqlite_benchmark.jsonl \
    --prompt-config benchmarks/birdbench/prompts/default.yaml \
    --output results/birdbench_rollouts.jsonl \
    --temperature 1 \
    --num-repeats 1
```

(For Spark use `birdbench_spark_benchmark.jsonl` and `prompts/spark.yaml`, with the servers started with
`"${OVERRIDES[@]}"`.)

For a 5-example smoke test against the resource server's `example.jsonl`,
see `resources_servers/bird_sql/README.md`.

## Collecting training data

`collect.py` turns any split/dialect into training records `{id, db_id, context, completions: [{completion, reward,
reference_logprob}]}`, one per task. `reference_logprob` is the summed per-token logprob of the generation, which is why
it calls the model endpoint itself with `logprobs: true, top_logprobs: 1` rather than going through the `vllm_model`
server (that one hardcodes `top_logprobs=0`, which hosted inference APIs may not support). Rewards are exactly the
`bird_sql` server's (SQLite, or Spark with `--dialect spark`).

```bash
# Policy endpoint from env.yaml (policy_base_url / policy_api_key / policy_model_name) or --policy-* flags.
python -m benchmarks.birdbench.collect collect --split train --dialect sqlite --num-repeats 32 \
    --reasoning-effort low --limit 20          # drop --limit for the full run

python -m benchmarks.birdbench.collect reprocess --split train --output-dir <collect run dir>
python -m benchmarks.birdbench.collect group --rollouts <eval.sh output>/rollouts.jsonl --output grouped.jsonl
python -m benchmarks.birdbench.collect select-hard --prepared-jsonl data/birdbench_train_sqlite.jsonl \
    --grouped-jsonl <run dir>/grouped.jsonl --output-jsonl data/birdbench_train_sqlite_hard.jsonl
python -m benchmarks.birdbench.collect collect --split train --input-jsonl data/birdbench_train_sqlite_hard.jsonl \
    --num-repeats 128                           # second pass on the tasks nobody solved
```

- **`collect`** prepares the split/dialect's data first if it is missing, then writes `raw_choices.jsonl` (every raw API
  response, saved before any processing) and `grouped.jsonl` into `results/collect_<split>_<dialect>_<model>/`
  (`--output-dir` overrides it). A repeat that still fails after 5 retries is dropped with a warning, so a task can end up
  with fewer than `--num-repeats` completions.
- **`reprocess`** rebuilds the grouped file from `raw_choices.jsonl` with no API calls, for when the text
  reconstruction, reward or logprob logic is fixed after an expensive run.
- **`group`** converts the rollouts of a `gym eval run` (e.g. `eval.sh`) into the same format. `reference_logprob` is
  only set if `return_token_id_information` reached the policy model.
- **`select-hard`** keeps the prepared rows (with their `sql_context`) that no collected rollout solved, pooled over
  any number of grouped files. Records are matched to rows by `id`.
- **Saved completions** are `<think>reasoning</think>answer<stop token>`. The `<think>` prefix is added for the saved
  text only; chat templates force it, so it is never generated and has no logprob. The stop token is present only when
  the model chose to stop (a cut-off by `--max-tokens` has none).
- **SQL concurrency.** `--sql-concurrency` defaults to 16: with 32 threads contending on SQLite's progress handler the
  GIL starved and rewards collapsed (0/32 correct at 32 vs 31/32 at 16, same data). Use about 4 for `--dialect spark`.
- **Spark** needs a JDK and pyspark, so run it with `resources_servers/bird_sql/.venv`'s python. It loads the split's
  databases into the same Spark warehouse the server uses (`spark_warehouse` for dev, `spark_warehouse_train` for train).

## Prompt

`prompts/default.yaml`: the model reasons step-by-step and returns the final
SQL inside a ` ```sql ``` ` block. Explains the `sql_context` format (the
`#### Tables`/`#### Foreign key` structure, per-column `data_type`/
`description`/`values`, and the `TYPE (primary key)` marker) before the
schema itself, so the model has the format legend in context before reading
the actual data.

`prompts/spark.yaml` is the same prompt for the `spark` dialect: it names Spark SQL as the engine and shows
schema-qualified, backtick-quoted names, with no Spark-specific instructions.
