# LabBench2 direct-PDF retrieval

This benchmark runs LitQA3, FigQA2, and TableQA2 with OpenCode through Gym's
general Harbor bridge. Each task mounts only the benchmark-relevant scientific
papers at `/papers`; it does not provide a retrieval skill, extracted-text
cache, source hint, or gold answer to the agent. The agent may choose to extract
text, render pages, crop figures, run OCR, or use any other installed tool.

The source checkout calls the table task `tableqa2`, so this integration keeps
that upstream name. The environment implementation and detailed acquisition,
licensing, task-layout, and troubleshooting notes live in
[`environments/labbench2_pdf`](../../environments/labbench2_pdf/README.md).

## Configure endpoints

Put the policy and judge endpoints in the repository-root, gitignored
`env.yaml`:

```yaml
policy_base_url: https://your-policy-endpoint/v1
policy_api_key: your-policy-key
policy_model_name: your-policy-model

judge_base_url: https://your-judge-endpoint/v1
judge_api_key: your-judge-key
judge_model_name: your-judge-model
```

Harbor owns inference for this benchmark, so no Gym `--model-type` is needed.

## Prepare the full benchmark

From the repository root:

```bash
gym eval prepare --benchmark labbench2_pdf
```

Preparation downloads the pinned LabBench2 question snapshot, attempts to
resolve its DOI papers from public open-access sources, builds the Docker
runtime, skips questions missing any required PDF, and creates the Harbor task
registry. It writes Gym's generated index to
`benchmarks/labbench2_pdf/data/labbench2_pdf_benchmark.jsonl`; generated tasks
and PDF caches stay gitignored under `environments/labbench2_pdf/data/`.

To use a local paper snapshot instead of downloading PDFs:

```bash
gym eval prepare --benchmark labbench2_pdf \
  +prepare_script_args.papers_dir=/absolute/path/to/all_papers
```

You can also bypass both downloads:

```bash
gym eval prepare --benchmark labbench2_pdf \
  +prepare_script_args.questions_dir=/absolute/path/to/questions \
  +prepare_script_args.papers_dir=/absolute/path/to/all_papers
```

Preparation is deliberately repeatable: cached downloads are reused and the
generated task registry is replaced. Use
`+use_cached_prepared_benchmarks=true` to skip preparation when the generated
benchmark index already exists.

## Run

After preparation, collect the complete benchmark with:

```bash
gym eval run \
  --benchmark labbench2_pdf \
  --split benchmark \
  --output results/labbench2_pdf/rollouts.jsonl \
  --concurrency 1
```

For one rollout, add `--limit 1`. To run one real FigQA2 item specifically,
start the benchmark server and use the generated FigQA2 index:

```bash
gym env start --benchmark labbench2_pdf

sed -n '1p' environments/labbench2_pdf/data/tasks_docker/figqa2_input.jsonl \
  > /tmp/labbench2_figqa2_single.jsonl

gym eval run --no-serve \
  --agent labbench2_pdf_benchmark_harbor_agent \
  --input /tmp/labbench2_figqa2_single.jsonl \
  --output results/labbench2_pdf/figqa2_opencode_single.jsonl \
  --concurrency 1
```

The Gym rollout is written to the requested JSONL. Harbor's original ATIF
trajectory remains under
`results/labbench2_pdf/harbor_jobs/<job>/<trial>/agent/trajectory.json`; inspect
its tool calls to see whether the agent rendered, cropped, or otherwise read an
image. Image inspection is available but is not required by the task prompt.

## Rollouts

A single real FigQA2 task, `figqa2-0001-b9ba0817`, was collected through Gym
with OpenCode 1.18.35 and GPT-5-mini in Docker on 2026-10-09. The agent made
five tool calls: it listed `/papers`, tried two PDF searches, read
`/papers/10.1101_2023.10.16.561085.pdf` with OpenCode's native `read` tool,
and wrote and checked `/app/answer.txt`. Its answer, `PBMC (12k)`, matched the
reference exactly and received reward 1.0 without a judge request. This run
does not demonstrate explicit page rendering, image cropping, or PNG inspection.

The reviewable evidence is committed with the environment's example data:

- [Gym rollout](../../environments/labbench2_pdf/data/example_rollouts.jsonl)
- [Original Harbor ATIF trajectory](../../environments/labbench2_pdf/data/example_harbor_trajectory.json)
- [Verifier details](../../environments/labbench2_pdf/data/example_verifier_details.json)
- [Aggregate rollout metrics](../../environments/labbench2_pdf/data/example_rollouts_aggregate_metrics.json)
- [Rollout health report](../../environments/labbench2_pdf/data/example_rollout_health.json)

The run used 74,733 input tokens (43,904 cached) and 380 output tokens. This is
one representative smoke rollout, not a full benchmark baseline. Gym's health
report records one **unobserved** rollout and zero model-call captures; it is
not a health-check pass. The Gym conversion is marked lossy, so inspect the
original ATIF alongside the Gym row. Its source-trajectory reference has been
changed to the committed ATIF file; endpoint credentials, PDFs, rendered
images, agent databases, and bulk logs are not included in this evidence set.

To reproduce the run against an already prepared registry, start Gym with a
fresh Harbor jobs directory in one terminal:

```bash
gym env start --benchmark labbench2_pdf \
  ++labbench2_pdf_benchmark_harbor_agent.responses_api_agents.harbor_agent_general.harbor_jobs_dir=environments/labbench2_pdf/data/harbor_jobs/pr-figqa2-retry1/jobs
```

Once the server is ready, collect exactly one FigQA2 task in another terminal:

```bash
head -n 1 environments/labbench2_pdf/data/tasks_docker/figqa2_input.jsonl \
  > /tmp/labbench2_figqa2_single.jsonl

gym eval run --no-serve \
  --agent labbench2_pdf_benchmark_harbor_agent \
  --input /tmp/labbench2_figqa2_single.jsonl \
  --output environments/labbench2_pdf/data/harbor_jobs/pr-figqa2-retry1/rollouts.jsonl \
  --num-repeats 1 \
  --concurrency 1
```

Use a new directory name in both commands for each independent run to avoid
resuming an earlier Harbor trial. Stop the server after collection. The raw
job directory remains gitignored; only the small evidence exports above are
intended for review.

## Test

```bash
pytest -q benchmarks/labbench2_pdf/tests environments/labbench2_pdf/tests
```
