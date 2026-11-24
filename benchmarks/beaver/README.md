# BEAVER Benchmark

[BEAVER](https://beaverbench.github.io/) ([paper](https://arxiv.org/abs/2409.02038)) is an enterprise text-to-SQL
benchmark: questions over real, anonymized data warehouses with hundreds of tables, gold queries in MySQL. This
benchmark asks for one MySQL query per question from the schema alone and scores it with BEAVER's official execution
accuracy, through the [`beaver_sql`](../../resources_servers/beaver_sql/README.md) resources server.

Integration profile `custom-gym-verifier`; agent `simple_agent`; scorer `beaver_sql`. Layout as `birdbench`.

## Measurement contract

**Provenance.** Questions: [`beaverbench/beaver-query`](https://huggingface.co/datasets/beaverbench/beaver-query) at
revision `3c8eb607`; databases: `beaver_db.zip` in
[`beaverbench/beaver-table`](https://huggingface.co/datasets/beaverbench/beaver-table) at revision `86f2c4ec`, each dump
pinned by SHA-256. Both MIT and gated: accept the terms on each dataset page, then `hf auth login`. Nothing is
committed here; `prepare.py` and the server download at run time.

| Split | Questions | Database |
|---|---|---|
| `dw` | 5,787 | `dw` (97 tables) |
| `dw_real` | 121 (from real query logs) | `dw` |
| `neutron` | 1,017 | `neutron` (175 tables) |
| `nova` | 1,053 | `nova` (109 tables, 2.1 GB dump) |

**Sample.** The default is 300 `dw` questions drawn with BEAVER's own sampling (`random.Random(77).sample`, as its
`data/download_hf.py`), the sample auto-ontology-eval runs. Samples nest: the first 100 of the 300 are the
100-question sample, and the first 300 of a 400-question sample are these 300. `sample=0` takes every question.

**Task contract.** One user turn, no tools (`prompts/schema_only.yaml`, auto-ontology-eval's schema-only control
prompt): the question; its domain knowledge as `### Evidence` when it has any; "Write a single mysql query that
answers the question."; and the database's `CREATE TABLE` statements from the dump (about 96,000 characters for `dw`).
The system prompt asks for the answer in a ```` ```sql ```` block and for exactly the requested output fields. The
evidence is BEAVER's annotated domain knowledge (222 of the first 400 `dw` questions have some). BEAVER's leaderboard
**Setting 0 gives no hints**: prepare with `include_evidence=false` for a number comparable with it.

**Scoring.** Reward 1 when the model's query and the gold query give the same result under BEAVER's `ex_acc`, else 0
(details in the server's README). Two headline numbers, as BEAVER reports:

- `pass@1[avg-of-k]/accuracy` over every task;
- `nonempty_gold/pass@1[avg-of-k]/accuracy` over tasks whose gold query returns rows. Any query that returns nothing
  (or fails) matches an empty gold result, so this is the stricter number.

Per-category scores (`category`: `real`, `complex query`, `domain-specific query`, `domain-specific complex query`)
are reported too. **Run conditions:** single-turn, no judge; 10 seconds per query; token budget and sampling are the
model's (eval command).

## Gold check

`gold_check.py` runs every gold query through the grader. On MySQL 8.0.46 with `--lower-case-table-names=1`, all
6,925 questions of `dw`, `dw_real` and `neutron`:

| Split | Rows | Empty | Error | Timeout |
|---|---|---|---|---|
| `dw` | 5,771 | 10 | 0 | 6 at 8 at a time, 0 one at a time |
| `dw_real` | 102 | 19 | 0 | 0 |
| `neutron` | 1,017 | 0 | 0 | 0 |

The median gold query takes 0.02 s. With MySQL's default case-sensitive table names (Linux), 950 of `neutron`'s gold
queries and 15 of `dw_real`'s fail, because they spell tables and aliases in another case than the dumps, which were
made on macOS. The server refuses such a MySQL server; see its README for the `docker run` line.

## Official evaluator check

`official_check.py` scores the same predictions with BEAVER's official `eval/utils/ex_acc.py` (downloaded at commit
`81a2a88a1`; the repository has no license file, so nothing from it is committed) and with the grader. Results are in
the PR that adds this benchmark.

## Preparing data and evaluating

```bash
hf auth login
gym eval prepare --benchmark beaver
# other settings, e.g. all of dw without hints; the file name follows the settings (prepare.output_fpath_for):
gym eval prepare --benchmark beaver '+prepare_script_args.sample=0' '+prepare_script_args.include_evidence=false' \
    '++beaver_agent.responses_api_agents.simple_agent.datasets.0.jsonl_fpath=benchmarks/beaver/data/beaver_dw_all_seed77_noevidence.jsonl'
```

Start MySQL (server README), then:

```bash
export BEAVER_MYSQL_DSN='mysql://beaver_reader:<password>@127.0.0.1:3306'
export BEAVER_MYSQL_ADMIN_DSN='mysql://root:<admin password>@127.0.0.1:3306'
gym eval run --benchmark beaver --model-type openai_model --split benchmark \
    --output results/beaver/rollouts.jsonl --num-repeats 1 \
    '+beaver_sql_resources_server.resources_servers.beaver_sql.mysql_dsn=${oc.env:BEAVER_MYSQL_DSN}' \
    '+beaver_sql_resources_server.resources_servers.beaver_sql.mysql_admin_dsn=${oc.env:BEAVER_MYSQL_ADMIN_DSN}'
```

For `neutron` or `nova` questions, add the database to the server's `databases` list.

## Files

| File | Purpose |
|---|---|
| `config.yaml`, `manifest.yaml` | Wiring and the benchmark's declaration |
| `prepare.py` | Questions, sample, evidence and schema into `data/beaver_<splits>_<n>_seed<seed>[_noevidence].jsonl` |
| `prompts/schema_only.yaml` | The prompt |
| `gold_check.py` | Runs every gold query through the grader and counts the outcomes |
| `official_check.py` | Scores predictions with the official evaluator and the grader, and counts disagreements |

## Credits and licenses

BEAVER: Chen et al., MIT (data). The grading rule follows BEAVER's official evaluator. The schema-only setup, prompt
and BEAVER seeding are adapted from the Auto Ontology team's
[auto-ontology-eval](https://github.com/NVIDIA/auto-ontology-eval) (Apache 2.0), which first ran BEAVER on Gym.
