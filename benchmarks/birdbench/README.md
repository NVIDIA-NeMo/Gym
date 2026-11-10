# BIRD Benchmark

Execution-based text-to-SQL on BIRD, bound to the `bird_sql` resource server. Two choices define a run:

| Choice | Values | What it controls |
|---|---|---|
| **dialect** | `sqlite` (default), `spark`, `postgres` | the SQL the model writes, the prompt, and the engine that runs it |
| **split** | `dev` (default), `train` | the questions and databases: `dev` is the 1534-task benchmark, `train` is BIRD's train split (not an evaluation set) |

Reward is binary: unordered result-set equality between the model's query and BIRD's gold query. Metrics are overall
plus per-difficulty (simple / moderate / challenging; `train` has no difficulty labels).

## The two processes

| Process | Command | Needs a model? | Produces |
|---|---|---|---|
| **prepare** | `uv run --project benchmarks/birdbench python -m benchmarks.birdbench.prepare` | no | the questions file: one row per task with `question`, `gt_sql`, and `sql_context` (schema plus example values) |
| **eval** | `eval.sh --model <name>` | yes | `rollouts.jsonl` (one row per rollout) and `rollouts_aggregate_metrics.json` (pass@1, per-difficulty and failure metrics) |

- **prepare** runs first and **eval** consumes its output; eval runs it for you if the file is missing.
- **eval** measures a model through Gym's servers and scores every answer with the `bird_sql` reward.
- Both take the same `--dialect` and `--split` (the choices above) and default to `sqlite` and `dev`.

## Installation

Python dependencies are declared and locked, so there is nothing to `pip install`:

- **prepare** runs in the benchmark's own locked project, `benchmarks/birdbench/pyproject.toml` (`bm25s`, `nltk`). `eval.sh`
  and `gym eval prepare` use it for you; to run `prepare` yourself, start it with
  `uv run --project benchmarks/birdbench` (see [Preparing data](#preparing-data)).
- **The server** runs in `resources_servers/bird_sql`'s locked environment, which `gym` builds on first start. It
  includes `pyspark` and `sqlglot` for the Spark dialect (`pyspark` runs the model's SQL, `sqlglot` checks that it is a
  single `SELECT`/`WITH`), and `psycopg` and `psycopg_pool` for the PostgreSQL dialect; the SQLite dialect never
  imports them. The preparation project also has `psycopg`: `prepare --dialect postgres` renders the column types the
  loader will create, without connecting.

What is left is outside Python:

- **`nltk` tokenizer data** (`punkt_tab`), used by `prepare`, is downloaded on first use. On a machine without internet
  access, run `python -m nltk.downloader punkt_tab` elsewhere and copy `~/nltk_data` over.
- **The Spark dialect needs a JDK 17.** `prepare` itself needs no Java; the server does, so `JAVA_HOME` must be set in
  the shell that starts it.

```bash
# Linux (Debian/Ubuntu). Any JDK 17 works, e.g. `conda install -c conda-forge openjdk=17` instead of apt.
sudo apt install -y openjdk-17-jdk
export JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(which java)")")")"   # must be set in the shell that starts servers
java -version                            # should report 17
```

On macOS: `brew install openjdk@17`, then `export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"`.

Spark runs in local mode inside the server process; no cluster is needed.

**PostgreSQL dialect** needs a PostgreSQL server. `BIRD_PG_DSN` is
the grader's DSN: a read-only role that the loader creates. `BIRD_PG_ADMIN_DSN` (optional) is a role that can create
schemas and roles; when it is set, the server loads any missing `db_id` on startup (dev takes about 20 s).

```bash
docker run -d --name bird-pg -p 127.0.0.1:5432:5432 -e POSTGRES_USER=bird -e POSTGRES_PASSWORD=<admin password> postgres:16
export BIRD_PG_ADMIN_DSN=postgresql://bird:<admin password>@127.0.0.1:5432/bird
export BIRD_PG_DSN=postgresql://bird_reader:<reader password>@127.0.0.1:5432/bird
```

## Dialects and splits

| | sqlite | spark | postgres |
|---|---|---|---|
| Prompt | `prompts/default_sqlite.yaml` | `prompts/default_spark.yaml` | `prompts/default_postgres.yaml` |
| `sql_context` rendered as | SQLite schema | Spark schema: `` `db_id`.`table` `` names, backtick-quoted identifiers, Spark column types | PostgreSQL schema: `"db_id"."table"` names, double-quoted identifiers, column names in lower case as loaded, the loaded column types |
| Model's SQL runs on | the SQLite databases | a local PySpark copy (Parquet under `resources_servers/bird_sql/.bird_sql/spark_warehouse[_train]/`, built on first start) | a PostgreSQL server you provide (one schema per `db_id`, loaded on first start when `BIRD_PG_ADMIN_DSN` is set) |

| | dev | train |
|---|---|---|
| Questions | BIRD `dev.json` | the **uberset**: BIRD `train.json` with every row that ReViSQL re-verified ([`bird-verified-train.json`](https://github.com/uiuc-kang-lab/ReViSQL/blob/main/data/bird-verified-train.json)) replaced by its verified version, matched by `question_id`; same size as `train.json` |
| Databases | BIRD dev, ≈1.4 GB, downloaded on first use | BIRD train, ≈9 GB, downloaded on first use to `resources_servers/bird_sql/.bird_sql/train/` |
| `difficulty` | simple / moderate / challenging | `null` |
| BM25 corpus scan | every distinct value | at most 2000 per column (the databases are multi-GB; only a few example values per column are shown anyway) |

- **Gold SQL is never transpiled.** In every dialect it runs on the original SQLite database; only the two result sets
  are compared (floats rounded to 6 places, dates compared as strings).
- **Spark execution.** ANSI mode is on, as in Spark 4 / Databricks (invalid casts and divide-by-zero raise errors). Only
  a single `SELECT`/`WITH` query is executed; anything else scores as an execution error. Spark has 4 local cores, so
  use a lower server `max_concurrency` than SQLite's 32
  (`+birdbench_bird_sql_resources_server.resources_servers.bird_sql.max_concurrency=8`). A numeric column whose data
  is not cleanly numeric, and date/datetime columns, are `STRING`.
- **PostgreSQL execution.** One schema per `db_id` in one database, with the Spark typing rules; text columns use the
  `"C"` collation so ordering matches SQLite. The model's query runs as one prepared statement (so `COMMIT; ...` cannot
  leave the transaction) in a read-only transaction as the read-only role, with `statement_timeout`, `search_path` set
  to the `db_id`, and parallel query off (a parallel float `SUM` changes from run to run), on one of `max_concurrency`
  pooled connections per server process; size the server's `max_connections` for all of them. Tables keep BIRD's
  names and column names are lower-cased, so an unquoted column name works as in SQLite; mixed-case table names
  (17 tables in 4 dev databases) still need double quotes. BIRD's evidence, shown with the question, spells columns in
  their original case: a model that copies such a name inside double quotes names a column that doesn't exist (61 of
  1,534 dev questions quote one, or name one that needs quotes anyway). `postgres_gold_check.py` grades sqlglot transpiles of
  the golds: 91% match after three mechanical fixes (identifier case, double-quoted strings, `REAL`).
- **Caveat.** Because the gold is the answer under SQLite semantics, a correct Spark query can differ from it when the
  gold relies on SQLite behavior (bare `GROUP BY` columns, integer division, case-insensitive `LIKE`, loose
  text/number comparison). `spark_gold_check.py` estimates about 8% of dev (121 of 1534) are affected.

## Preparing data

`prepare` writes `data/birdbench_<dialect>_<split>.jsonl` (e.g. `birdbench_sqlite_dev.jsonl`), downloading the
databases first if needed:

```bash
uv run --project benchmarks/birdbench python -m benchmarks.birdbench.prepare                                  # dev,   sqlite (the defaults)
uv run --project benchmarks/birdbench python -m benchmarks.birdbench.prepare --dialect spark                  # dev,   spark
uv run --project benchmarks/birdbench python -m benchmarks.birdbench.prepare --dialect postgres               # dev,   postgres (no server needed)
uv run --project benchmarks/birdbench python -m benchmarks.birdbench.prepare --split train                    # train, sqlite
uv run --project benchmarks/birdbench python -m benchmarks.birdbench.prepare --split train --dialect spark    # train, spark
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

## Evaluating a model

`eval.sh` runs one model end to end: it prepares the data if missing, runs `gym eval run`, prints a failure
breakdown, and kills leftover server processes on exit.

**The model** is an OpenAI-compatible endpoint that you already serve, e.g. a vLLM server you started (local or on a
cluster node), or a hosted API. Gym does not download or launch it. `--model` is the name that endpoint uses for the
model, `--model-url` its base URL (e.g. `http://localhost:8000/v1`) and `--model-api-key` its key. Instead of flags, set
`policy_model_name`, `policy_base_url` and `policy_api_key` in `env.yaml`; flags override it.

```bash
./benchmarks/birdbench/eval.sh --model <name>                                 # dev,   sqlite (the defaults)
./benchmarks/birdbench/eval.sh --model <name> --dialect spark                 # dev,   spark
./benchmarks/birdbench/eval.sh --model <name> --dialect postgres              # dev,   postgres
./benchmarks/birdbench/eval.sh --model <name> --split train                   # train, sqlite
./benchmarks/birdbench/eval.sh --model <name> --split train --dialect spark   # train, spark
```

`--limit 5` gives a smoke test and `--num-repeats N` N rollouts per task. Results go to
`benchmarks/birdbench/results/<split>_<dialect>/<model>/` (override with `--output_dir`). The script header documents every option;
other flags pass through to `gym eval run`.

`rollouts.jsonl` has one row per rollout (prompt, response, reward, verifier fields like `extracted_sql` and
`failure_reason`).
