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
uniform `num_repeats` across problems (which `ng_collect_rollouts` enforces).

## Dataset Format

Flat-field JSONL (prompt templating happens at runtime via
`benchmarks/critpt/prompts/turn1.yaml`). Each row:

- `problem_id`: AA API submission key, e.g. `Challenge_1_main`
- `problem`: physics question (Markdown)
- `code_template`: Python function stub the model populates in Turn 2
- `uuid`: same as `problem_id`

## Observability

Both signals surface in the run log (prefixed `(critpt_resources_server)`):

- Per-`verify()` log line at WARNING level:
  `CritPt verify #<N>: batch <B> at <K>/70 submissions buffered (problem_id=...)`
  where `#<N>` is a monotonic counter of all verify calls received since startup (useful
  when tailing the log with `num_repeats > 1`, since per-batch counts interleave but `#N`
  always increases), `<B>` is the pending-batch index, and `<K>/70` is that batch's fill.
- Batch-fire log line (WARNING level):
  `CritPt batch full (70 submissions); firing AA API.`
- On AA API failure, full exception + traceback is logged via `LOG.exception(...)` and
  every waiter in that batch is failed with the same exception.

`GET /status` returns the live buffer count:

```bash
PORT=$(grep "critpt_resources_server.*Uvicorn running" <run.log> | grep -oE '127\.0\.0\.1:[0-9]+' | cut -d: -f2)
curl -s http://127.0.0.1:$PORT/status
# {"pending_batches": [47], "batch_size": 70}
# (with num_repeats=N, the list grows up to N entries — one per concurrently-filling batch)
```

On HPC: bind is `127.0.0.1` on the compute node — curl from the same host, or
`ssh <node> "curl ..."`.

## Tests

```bash
ng_test +entrypoint=resources_servers/critpt
```

Covers code extraction edge cases, partial/full/multi-batch buffering, the `/status`
endpoint, and the empty-code-still-counts-as-a-slot invariant.

## Licensing

- **Code**: Apache 2.0
- **Data**: CritPt dataset license (see [CritPt-Benchmark/CritPt](https://huggingface.co/datasets/CritPt-Benchmark/CritPt) on HuggingFace)
- **Evaluator**: Artificial Analysis API ToS
- **Dependencies**:
  - nemo_gym: Apache 2.0
