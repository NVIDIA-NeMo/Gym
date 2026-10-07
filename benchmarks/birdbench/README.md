# BIRD Benchmark

Execution-based text-to-SQL on BIRD dev, bound to the `bird_sql` resource server.

- **Tasks**: 1534 across 11 SQLite databases
- **Reward**: binary; unordered result-set equality on the per-`db_id` DB
- **Metrics**: overall + per-difficulty (simple / moderate / challenging)
  via `compute_subset_metrics(field="difficulty")`

## Installation

None of the packages below are `pyproject.toml` dependencies, so install them
separately into the environment you run `gym` from.

**SQLite (default dialect)** -- BM25 retrieval for `sql_context`:

```bash
uv pip install bm25s nltk
```

**Spark SQL dialect (in development)** -- needs a JDK plus Python packages
(`pyspark` for execution, `sqlglot` to transpile the gold SQL):

```bash
# macOS (Homebrew). openjdk@17 is keg-only, so point JAVA_HOME at it
# instead of relying on a system-wide symlink:
brew install openjdk@17
export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"

# Linux: install any JDK 17 (e.g. `apt install openjdk-17-jdk`) and set JAVA_HOME.

uv pip install "pyspark>=3.5,<4" sqlglot pandas pyarrow setuptools
java -version   # sanity check; should report 17
```

`setuptools` is needed because PySpark 3.5 imports `distutils`, which Python
3.12+ no longer ships. If Spark fails with `PYTHON_VERSION_MISMATCH`, its
workers picked up a different Python than the driver; set
`PYSPARK_PYTHON` to the interpreter you run `gym` with.

Spark runs in local mode inside the server process, so no cluster or daemon is
needed.

## Preparation

Dataset prep needs `bm25s` and `nltk` (see Installation above):

```bash
gym eval prepare --benchmark birdbench
```

This downloads the BIRD `dev.zip` (≈1.4 GB) via
`resources_servers.bird_sql.setup_bird_sql.ensure_bird_sql()` and writes
`data/birdbench_benchmark.jsonl`. Each row has `question`, `gt_sql`,
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

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --benchmark birdbench
```

Requires `policy_base_url` / `policy_api_key` / `policy_model_name` in
`env.yaml` (or passed as CLI overrides).

## Collecting rollouts

```bash
gym eval run --no-serve \
    --agent birdbench_bird_sql_simple_agent \
    --input benchmarks/birdbench/data/birdbench_benchmark.jsonl \
    --prompt-config benchmarks/birdbench/prompts/default.yaml \
    --output results/birdbench_rollouts.jsonl \
    --temperature 1 \
    --num-repeats 4
```

`--no-serve` skips the data-preparation step that normally applies
`prompt_config` automatically, so `--prompt-config` must be passed explicitly
here -- otherwise rows won't have `responses_create_params` and collection
will fail with `KeyError: 'responses_create_params'`.

For a 5-example smoke test against the resource server's `example.jsonl`,
see `resources_servers/bird_sql/README.md`.

## Prompt

`prompts/default.yaml`: the model reasons step-by-step and returns the final
SQL inside a ` ```sql ``` ` block. Explains the `sql_context` format (the
`#### Tables`/`#### Foreign key` structure, per-column `data_type`/
`description`/`values`, and the `TYPE (primary key)` marker) before the
schema itself, so the model has the format legend in context before reading
the actual data.
