# CritPt Benchmark

Benchmark wrapper for [CritPt](https://huggingface.co/datasets/CritPt-Benchmark/CritPt), a
70-problem research-level physics benchmark. Each problem has a description and a Python
code template; the model must produce a precise numerical answer.

- **Tasks**: 70 physics problems
- **Reward**: aggregate accuracy from the [Artificial Analysis API](https://artificialanalysis.ai/documentation#critpt-api) (private test cases run server-side), distributed uniformly across the batch — every rollout shares the same float reward
- **Metrics**: `pass@1/accuracy` — fraction of problems the AA API accepts

The agent runs two LLM turns per problem:
1. **Turn 1** (`prompts/turn1.yaml`): step-by-step derivation ending in `Final Answer:`
2. **Turn 2**: populate the code template with the answer (the model sees its Turn 1 reasoning)

Turn 2's output is submitted to the AA API by `CritPtResourcesServer.verify()`.

## API key

The Artificial Analysis API key is read from `env.yaml`:

```yaml
artificial_analysis_api_key: <your-key>
```

The resources server config interpolates this via `${artificial_analysis_api_key}` — no key
in any committed file.

## Replay recovery

`resources_servers.critpt.replay` is a recovery tool for partially scored CritPt runs. CritPt
does not score each problem independently: the resources server buffers 70 generated code
submissions and sends them together to the Artificial Analysis API. If the API returns a
rate-limit error after model inference has already produced submissions, the rollout can fail
without a final AA score for that batch.

Replay reuses the submissions saved by the original run. It reads the cached submissions,
skips any submission IDs that already have a successful AA response, sends only unscored full
batches back to AA, and appends the recovered responses to the same cache. This lets you wait
for the AA quota to reset and recover scores without rerunning the model.

Replay is only possible if the resources server persisted the original submissions.

Before starting `ng_run`, set a cache directory:

```bash
export CRITPT_CACHE_DIR=results/critpt_cache
mkdir -p "$CRITPT_CACHE_DIR"
```

Each server launch writes into its own `<timestamp>-<pid>-<rand>` subdirectory under
`CRITPT_CACHE_DIR` (e.g. `results/critpt_cache/20260707-110622-48213-1a2b3c4d/`) so
independent runs never share cache files. The exact path is logged at startup:

```
CritPt cache for this run: <repo>/results/critpt_cache/20260707-110622-48213-1a2b3c4d
```

Each run's subdirectory contains two replay inputs:

- `submissions.jsonl`: one line per submission received by `verify()`
- `aa_responses.jsonl`: one line per successful AA scoring response
- `partial_metrics.json`: aggregate accuracy over the scored submissions **of that run only**

A relative `CRITPT_CACHE_DIR` is anchored to the repo root, not the server's
working directory (Gym runs each server from its own subdirectory). So
`CRITPT_CACHE_DIR=results/critpt_cache` always resolves to `<repo>/results/critpt_cache`
regardless of where the server ran. Pass the run's logged subdirectory as `--cache-dir`
when running replay after the quota resets. (Set `unique_cache_per_run: false` in the
resources server config to write directly into `CRITPT_CACHE_DIR` instead, e.g. to resume
a specific prior run's directory.)

## Prepare benchmark data

`CritPt-Benchmark/CritPt` is a public HuggingFace dataset (no auth required).

```bash
gym eval prepare --benchmark critpt
```

This invokes `benchmarks/critpt/prepare.py` (declared as `prepare_script` in `config.yaml`),
which downloads the dataset and writes the full 70-problem flat-field JSONL to
`benchmarks/critpt/data/critpt_benchmark.jsonl` (gitignored).

## Run servers

```bash
gym env start --benchmark critpt --model-type vllm_model
```

While `gym env start` is up, the CritPt resources server exposes a `GET /status` endpoint
that reports live batch-fill progress (e.g. `{"pending_batches":[47],"batch_size":70}`).

## Collect rollouts

With `gym env start` already up, point `gym eval run --no-serve` at the flat-field JSONL and pass the
Turn 1 prompt config so the framework materializes `responses_create_params.input` at
rollout time.

```bash
gym eval run --no-serve \
    --agent critpt_benchmark_agent \
    --input benchmarks/critpt/data/critpt_benchmark.jsonl \
    --output results/critpt_rollouts.jsonl \
    --num-repeats 1 \
    --prompt-config benchmarks/critpt/prompts/turn1.yaml \
    --temperature 0.0
```

Use `temperature: 0.0` to match the nemo-skills baseline and ensure reproducible scores.

## Replay after AA rate limits

If rollout collection fails because the AA scoring quota was exhausted, wait until the quota
resets and replay the unscored cached submissions. Point `--cache-dir` at the run's
subdirectory that the server logged at startup (`CritPt cache for this run: ...`), not the
base `CRITPT_CACHE_DIR`:

```bash
RUN_DIR="$CRITPT_CACHE_DIR/20260707-110622-48213-1a2b3c4d"  # from the server startup log
ARTIFICIAL_ANALYSIS_API_KEY="aa-xxxxx" <!-- pragma: allowlist secret --> \
  python -m resources_servers.critpt.replay --cache-dir "$RUN_DIR"
```

For multiple AA keys, pass a bracketed comma-separated list. Replay rotates to the next key
on each HTTP 429:

```bash
ARTIFICIAL_ANALYSIS_API_KEY="[aa-key-A,aa-key-B,aa-key-C]" <!-- pragma: allowlist secret --> \
  python -m resources_servers.critpt.replay --cache-dir "$RUN_DIR"
```

Replay is idempotent. It reads `submissions.jsonl`, skips any `submission_ids` already
present in `aa_responses.jsonl`, and appends only newly scored batches. If every configured
key is rate-limited, the command exits with code `3`; rerun it after the next quota reset.

AA public scoring requires full 70-submission batches. Any leftover short batch is left
unscored until enough cached submissions exist to fill a full batch. For a smoke run of
fewer than 70 problems, pass `--fire-after N` to pad the short batch up to 70 with empty
padding submissions and ship it anyway (matching the server's smoke-test `fire_after`); only the real
submissions are recorded as scored:

```bash
ARTIFICIAL_ANALYSIS_API_KEY="aa-xxxxx" <!-- pragma: allowlist secret --> \
  python -m resources_servers.critpt.replay --cache-dir "$RUN_DIR" --fire-after 5
```

### One-shot alternative

Runs prepare + servers + rollout collection and tears the servers down afterwards:

```bash
gym eval run \
    --model-type vllm_model \
    --benchmark critpt \
    --output results/benchmarks/critpt.jsonl \
    ++overwrite_metrics_conflicts=true \
    --split benchmark \
    --resume \
    ++reuse_existing_data_preparation=true \
    --model-url <your_endpoint> \
    --model-api-key <your_key> \
    --model <your_model> \
    --temperature 0.0
```

## Metrics

`pass@1/accuracy` is the headline metric.
