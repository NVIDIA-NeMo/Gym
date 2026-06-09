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
- On AA API failure, full exception + traceback is logged via `LOG.exception(...)` and
  every waiter in that batch is failed with the same exception.

`GET /status` returns the live buffer count:

```bash
PORT=$(grep -E "critpt(_benchmark)?_resources_server.*Uvicorn running" <run.log> | head -1 | grep -oE '127\.0\.0\.1:[0-9]+' | cut -d: -f2)
curl -s http://127.0.0.1:$PORT/status
# {"pending_batches": [47], "batch_size": 70}
# (with num_repeats=N, the list grows up to N entries — one per concurrently-filling batch)
```

On HPC: bind is `127.0.0.1` on the compute node — curl from the same host, or
`ssh <node> "curl ..."`.

## Running servers

```bash
ng_run "+config_paths=[benchmarks/critpt/config.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"
```

## Smoke test (5 example problems)

**Note:** The AA API rejects sub-70 batches. To run e2e on the 5 example problems, the resources server
has one opt-in config knob:

- `fire_after: int` — fire the batch after this many real submissions arrive, then pad
  up to `batch_size` (70) with empty dummies for the missing problem_ids (drawn from the
  hardcoded canonical CritPt problem list, `Challenge_<N>_main` for `N` in 1..70)

Defaults to `None`/unset, so production behavior is unchanged. Set it only for testing purposes:

```bash
ng_run "+config_paths=[benchmarks/critpt/config.yaml,responses_api_models/openai_model/configs/openai_model.yaml]" \
    '++critpt_benchmark_resources_server.resources_servers.critpt.fire_after=5'
```

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=critpt_benchmark_agent \
    +input_jsonl_fpath=resources_servers/critpt/data/example.jsonl \
    +output_jsonl_fpath=results/critpt_smoke.jsonl \
    +num_repeats=1 \
    "++responses_create_params={temperature: 0.0}"
```

What this exercises: agent runs Turn 1 + Turn 2 on exactly 5 problems -> 5 verify() calls
arrive -> server fires the batch after the 5th, pads with 65 empty dummies -> real AA
call -> aggregate distributed to the 5 real rollouts.

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
