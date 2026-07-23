# Legal Agent Bench Benchmark

This benchmark registers the existing
[Legal Agent Bench resource server](../../resources_servers/legal_agent_bench/README.md)
with Gym's benchmark catalog. It evaluates public Harvey LAB tasks through
the legal_agent_bench resource server's implementation.

Benchmark preparation reuses
the resource server's pinned task and skill caches and writes only a small,
gitignored benchmark index.

## Requirements

- Python 3.12 and the repository environment installed with `uv`
- Docker with a running daemon
- Authorized OpenAI-compatible policy and judge endpoints in the root
  `env.yaml`
- At least 10 GB of free working space

See the [resource-server README](../../resources_servers/legal_agent_bench/README.md)
for endpoint configuration, source and license details, cache locations, and
troubleshooting. The initial source download is several hundred MiB, and the
first rollout builds a document-tooling Docker image that can take several
minutes.

## Prepare

From the repository root, run:

```bash
gym eval prepare --benchmark legal_agent_bench
```

This validates or prepares the shared task and skill caches, then writes the
deterministic benchmark index to
`benchmarks/legal_agent_bench/data/legal_agent_bench_benchmark.jsonl`.
Repeated preparation reuses valid caches and does not download a second copy of
LAB.

## Run

For the standard one-shot workflow:

```bash
gym eval run \
  --model-type vllm_model \
  --benchmark legal_agent_bench \
  --split benchmark \
  --output results/legal_agent_bench_benchmark.jsonl \
  --concurrency 1
```

For a one-task smoke test, add `--limit 1`.

To manage the servers separately, start them first:

```bash
gym env start \
  --model-type vllm_model \
  --benchmark legal_agent_bench
```

Then run against them from a second activated terminal:

```bash
gym eval run --no-serve \
  --benchmark legal_agent_bench \
  --agent legal_agent_bench_benchmark_harbor_agent \
  --input benchmarks/legal_agent_bench/data/legal_agent_bench_benchmark.jsonl \
  --output results/legal_agent_bench_benchmark.jsonl \
  --concurrency 1 \
  --limit 1
```

## Scoring

The default `full_task` reward is LAB's official all-criteria score: a task
receives `1.0` only when every criterion passes. To use diagnostic criterion
pass rate instead, add this override to `gym env start` or the one-shot
`gym eval run` command:

```bash
+legal_agent_bench_benchmark_resources_server.resources_servers.legal_agent_bench.reward_mode=criteria_pass_rate
```

This changes only the reported reward; it does not change the tasks, agent, or
judge criteria.

## Test

Run the benchmark and resource-server tests with:

```bash
uv run pytest -q \
  benchmarks/legal_agent_bench/tests \
  resources_servers/legal_agent_bench/tests
```

Generated indexes, collation metrics, Harbor jobs, source documents, and skills
must not be committed.
