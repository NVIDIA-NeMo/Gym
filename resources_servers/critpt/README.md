# CritPt Resources Server

Evaluates research-level physics solutions on the
[CritPt](https://huggingface.co/datasets/CritPt-Benchmark/CritPt) benchmark by submitting
generated code to the [Artificial Analysis API](https://artificialanalysis.ai/documentation#critpt-api).

- Task type: multi-turn (two LLM calls per problem) + external batched API evaluator
- Domain: `other` (research physics)
- Tasks: **70** (PUBLIC mode requires submissions for all of them)
- Reward: aggregate accuracy from the AA API, distributed uniformly across the batch

> **Run commands and `env.yaml` setup**: see [`benchmarks/critpt/README.md`](../../benchmarks/critpt/README.md).

## Server Composition

CritPt uses a **custom two-turn agent** — `simple_agent` will not work:

- `responses_api_agents/critpt_agent` (Turn 1: solve; Turn 2: populate the code template)
- `responses_api_models/*` (typically `policy_model`)
- `resources_servers/critpt` (this server)

The full wiring lives in `benchmarks/critpt/config.yaml`, which chains all three plus the
dataset and Turn 1 / Turn 2 prompt configs.

## Batched Verification (unique to CritPt)

The AA API rejects single-problem submissions in PUBLIC mode — it requires all 70 problems in
one batched payload. The Gym framework expects per-rollout rewards via `verify()`, so this
server fakes the contract:

1. Each `verify()` call adds its submission to an in-memory dict keyed by `problem_id` and
   awaits a shared `asyncio.Future`.
2. When 70 unique `problem_id`s have accumulated, the last caller fires one POST to the AA
   API; the response resolves the future; all 70 waiters unblock together with the same
   aggregate accuracy as their `reward`.

This matches nemo-skills' behavior: aggregate accuracy is distributed uniformly across the
batch, so `pass@1` across the 70 problems equals the AA aggregate.

**Multi-repeat support (`num_repeats > 1`)**: each `verify()` joins the first pending batch
that doesn't already contain its `problem_id`, opening a new batch if every pending batch
already has it. With `num_repeats=N` and 70 problems, this produces N independent batches of
70 unique `problem_id`s — N AA API calls total, each scored as a separate run. Assumes
uniform `num_repeats` across problems (which `gym eval run --no-serve` enforces).

## Grading errors

Real runs send full batches of 70 real submissions only; a smoke run never calls AA (see below).

- **Quota or key errors stop the run.** AA answers HTTP 429 when the quota is exhausted and 401 or 403
  when the key is rejected or its account is not approved. Retrying cannot help, so every row of the
  batch fails like a judge failure: it lands in the failures file with its saved answer kept, typed
  `provider_quota_exhausted` (429; the reason carries the reset time from `Retry-After`: `AA quota
  exhausted, resets at <UTC time>; grade with replay or resubmit after the reset`) or
  `provider_credentials_rejected` (401/403). The run stops with that kind and does not retry. Before the
  reset, a recovery decision on that state is STOP; after it, `gym eval resubmit --recover` grades the
  saved answers again with no model calls. With a rejected key, fix the key (or get the account approved)
  first. `replay` grades the cached submissions the same way.
- **Partial grader errors are tolerated.** If AA reports `judge_error_count <= max_judge_errors` (server
  config, default `3`: the benchmark PIC's v1 value, about 5% of a 70 batch, to be tuned empirically), the batch
  is accepted and its accuracy is rescaled to the graded items:
  `accuracy * batch_size / (batch_size - judge_error_count)`, clamped to 1.0. This assumes AA's accuracy is
  correct items over the whole batch, so an ungraded item counts as incorrect. Every row records
  `judge_error_count` and `graded_items`. With more errors than that the batch fails as `judge_failed`
  (masked, graded again). `replay` takes the same threshold as `--max-judge-errors`.
- **A re-judge of fewer than `batch_size` rows cannot be graded alone.** AA only grades full batches and nothing
  is padded, so a smaller set (the last 23 rows of a run, say) waits for more rows. When
  `verify_timeout_seconds` passes, every waiter fails as `judge_failed` ("N saved answers pending, fewer than
  batch_size; grade them with replay together with other runs or after more rows arrive"): masked, saved answer
  kept. Set `verify_timeout_seconds` low for re-judge jobs, and `num_samples_in_parallel` to at least
  `batch_size`, or a full re-judge never fills a batch.
- **Any other AA failure** (a 4xx/5xx, a timeout) is `judge_failed`: the rows are masked and graded again.

## Dataset Format

Two JSONL files coexist with different shapes:

- **`benchmarks/critpt/data/critpt_benchmark.jsonl`** (full 70-row dataset, gitignored;
  produced by `prepare.py`). **Flat-field**: each row has `problem_id`, `problem` (Markdown
  question), `code_template` (Python stub), and `uuid`. Prompt templating happens at
  rollout time via the dataset entry's `prompt_config:
  benchmarks/critpt/prompts/turn1.yaml`.
- **`resources_servers/critpt/data/example.jsonl`** (5-row hand-curated fixture,
  committed). **Pre-materialized**: same flat fields plus `responses_create_params.input`
  already filled in from the Turn 1 template, matching the convention of every other
  paired server's example fixture. Because it's pre-materialized, callers must NOT pass
  `+prompt_config` when running rollouts against this file (the framework rejects rows
  that have both).

## Observability

Both signals surface in the run log, prefixed with the resources-server instance name —
`(critpt_resources_server)` when launched directly from this server's config, or
`(critpt_benchmark_resources_server)` when launched via `benchmarks/critpt/config.yaml`
(the inheriting instance):

- Per-`verify()` log line at WARNING level:
  `CritPt verify #<N>: batch <B> at <K>/70 submissions buffered (problem_id=...)`
  where `#<N>` is a monotonic counter of all verify calls received since startup (useful
  when tailing the log with `num_repeats > 1`, since per-batch counts interleave but `#N`
  always increases), `<B>` is the pending-batch index, and `<K>/70` is that batch's fill.
- Batch-fire log line (WARNING level):
  `CritPt batch full (70 submissions); firing AA API.`
- On a quota or key error, an ERROR line names the cause and the cache path. On any other AA API
  failure, the full exception + traceback is logged via `LOG.exception(...)` and every waiter in that
  batch is failed as `judge_failed`.

`GET /status` returns the live buffer count:

```bash
PORT=$(grep -E "critpt(_benchmark)?_resources_server.*Uvicorn running" <run.log> | head -1 | grep -oE '127\.0\.0\.1:[0-9]+' | cut -d: -f2)
curl -s http://127.0.0.1:$PORT/status
# {"pending_batches": [47], "batch_size": 70}
# (with num_repeats=N, the list grows up to N entries — one per concurrently-filling batch)
```

On HPC: bind is `127.0.0.1` on the compute node — curl from the same host, or
`ssh <node> "curl ..."`.

## Replay recovery

`resources_servers.critpt.replay` is a scoring-only recovery tool for CritPt runs that fail
after hitting Artificial Analysis rate limits. The live server writes model submissions before
shipping a full 70-submission batch to AA. If AA rejects the batch because quota is exhausted,
replay can later read those cached submissions, skip anything already scored, and send only the
unscored full batches back to AA without rerunning model inference.

Replay requires the live server to persist its cache. Set `CRITPT_CACHE_DIR` before starting
the server:

```bash
export CRITPT_CACHE_DIR=results/critpt_cache
```

When set, each server launch persists `submissions.jsonl`, successful `aa_responses.jsonl`
records, and `partial_metrics.json` under its own `<timestamp>-<pid>-<rand>` subdirectory of
`CRITPT_CACHE_DIR`, so independent runs never share (and pollute) each other's cache. The exact
path is logged at startup (`CritPt cache for this run: ...`). After the AA quota resets, point
replay at that run's subdirectory:

```bash
RUN_DIR="$CRITPT_CACHE_DIR/20260707-110622-48213-1a2b3c4d"  # from the server startup log
ARTIFICIAL_ANALYSIS_API_KEY="aa-xxxxx" <!-- pragma: allowlist secret --> \
  python -m resources_servers.critpt.replay --cache-dir "$RUN_DIR"
```

Set `unique_cache_per_run: false` on the resources server config to write directly into
`CRITPT_CACHE_DIR` instead (e.g. to resume a specific prior run's directory).

For the full benchmark workflow, see [`benchmarks/critpt/README.md`](../../benchmarks/critpt/README.md).

## Running servers

```bash
gym env start --benchmark critpt --model-type openai_model
```

## Smoke test (5 example problems)

**Note:** The AA API rejects sub-70 batches, and a smoke run must not call the AA server at all. To run e2e
on the 5 example problems, the resources server has one opt-in config knob:

- `fire_after: int` — fire the batch after this many real submissions arrive. The batch is **not** sent
  to AA: the submissions are written to the cache (when `cache_dir` is set) and every row comes back masked
  (`mask_sample: true`, no accuracy) with `failure_kind: judge_skipped` and a reason naming smoke
  mode. A smoke run therefore proves the wiring only; a real score needs a full run of 70 problems.

Defaults to `None`/unset, so production behavior is unchanged. Set it only for testing purposes:

```bash
gym env start --benchmark critpt --model-type openai_model     '++critpt_benchmark_resources_server.resources_servers.critpt.fire_after=5'
```

## Collecting rollouts

```bash
gym eval run --no-serve     --agent critpt_benchmark_agent     --input resources_servers/critpt/data/example.jsonl     --output results/critpt_smoke.jsonl     --num-repeats 1     --temperature 0.0
```

What this exercises: agent runs Turn 1 + Turn 2 on exactly 5 problems -> 5 verify() calls
arrive -> server fires the batch after the 5th, logs `smoke mode: batch of 5 submissions not sent to AA`
and resolves the 5 rollouts as not graded. No AA call is made.

## Tests

```bash
gym env test +entrypoint=resources_servers/critpt
```

Covers code extraction edge cases, partial/full/multi-batch buffering, the `/status`
endpoint, and the empty-code-still-counts-as-a-slot invariant.

## Licensing

- **Code**: Apache 2.0
- **Data**: CritPt dataset license (see [CritPt-Benchmark/CritPt](https://huggingface.co/datasets/CritPt-Benchmark/CritPt) on HuggingFace)
- **Evaluator**: Artificial Analysis API ToS
- **Dependencies**:
  - nemo_gym: Apache 2.0
