# BIRD Benchmark (dev 2025-11-06)

Execution-based text-to-SQL on the BIRD dev 2025-11-06 release, bound to the
`bird_sql` resource server.

- **Tasks**: 1534 across 11 SQLite databases
- **Reward**: binary; unordered result-set equality on the per-`db_id` DB
- **Metrics**: overall + per-difficulty (simple / moderate / challenging)
  via `compute_subset_metrics(field="difficulty")`

## Relation to `birdbench`

[`birdbench`](../birdbench/README.md) evaluates the 2024-06-27 BIRD dev release.
In the 2025-11-06 release
([`birdsql/bird_sql_dev_20251106`](https://huggingface.co/datasets/birdsql/bird_sql_dev_20251106)),
the BIRD team reviewed every dev task. They revised the question wording to
remove ambiguity, rewrote evidence to be concise and accurate, and corrected the
gold SQL. The question count and order, the difficulty labels, and the databases
are unchanged. This release is the "New Dev" column on the
[BIRD leaderboard](https://bird-bench.github.io/).

The two benchmarks share the databases, the `sql_context` rendering
(`benchmarks.birdbench.prepare.write_benchmark_rows`), the prompt
(`benchmarks/birdbench/prompts/default.yaml`), and the verifier, so a score
difference between them reflects the annotation changes only. Run them side
by side to compare a model on both releases.

The verifier keeps `bird_sql`'s 30s `sql_execution_timeout_s`, matching the
30s limit in BIRD's official EX evaluation. Three gold queries are slow on
SQLite: id 518 takes about 30s, id 1131 about 105s, and id 701 about 165s.
A gold timeout scores 0, so 701 and 1131 score 0 for every model, as they do
officially. Id 518 sits at the limit, and whether it can score depends on
machine load. Of the 1534 gold queries, the other 1530 match themselves under
the verifier. The exception is id 178, which averages floats and differs in
the eighth significant digit between runs.

## Preparation

```bash
uv pip install bm25s nltk
gym eval prepare --benchmark birdbench_dev1106
```

This downloads the databases through
`resources_servers.bird_sql.setup_bird_sql.ensure_bird_sql()`, which shares its
cache with `birdbench`, and the question annotations from HuggingFace at a pinned
revision. It writes `data/birdbench_dev1106_benchmark.jsonl` in the `birdbench`
row format: `question`, `gt_sql`, `sql_context`, `difficulty`, `db_id`, and `id`.

`prepare_script_args.include_evidence` and `prepare_script_args.dscp` take the
same values as in `birdbench`. See its README for details.

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --benchmark birdbench_dev1106
```

Requires `policy_base_url` / `policy_api_key` / `policy_model_name` in
`env.yaml` (or passed as CLI overrides).

## Collecting rollouts

```bash
gym eval run --no-serve \
    --agent birdbench_dev1106_bird_sql_simple_agent \
    --input benchmarks/birdbench_dev1106/data/birdbench_dev1106_benchmark.jsonl \
    --prompt-config benchmarks/birdbench/prompts/default.yaml \
    --output results/birdbench_dev1106_rollouts.jsonl \
    --temperature 1 \
    --num-repeats 4
```

As with `birdbench`, `--no-serve` skips the step that applies `prompt_config`,
so `--prompt-config` must be passed explicitly.
