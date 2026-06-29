# SciCode Resources Server

Verifies multi-step scientific code generation on the
[SciCode](https://huggingface.co/datasets/SciCode1/SciCode) benchmark. The agent submits the code it
generated for each sub-step; this server executes each sub-step's accumulated code against
ground-truth targets in `test_data.h5` and returns the reward.

- Task type: multi-step code generation (one LLM call per sub-step) + local code execution
- Domain: `coding`
- Tasks: 80 problems / 341 sub-steps (`validation` + `test`, the split nemo-skills calls `test_aai`)
- Reward: binary per problem — `1.0` iff every sub-step passes its tests

> **Run commands and `env.yaml` setup**: see [`benchmarks/scicode/README.md`](../../benchmarks/scicode/README.md),
> including how to stage the required `test_data.h5`.

## Server Composition

SciCode uses a **custom multi-step agent** — `simple_agent` will not work:

- `responses_api_agents/scicode_agent` (generates code sub-step by sub-step, accumulating prior code)
- `responses_api_models/*` (typically `policy_model`)
- `resources_servers/scicode` (this server)

The full wiring lives in `benchmarks/scicode/config.yaml`, which chains all three plus the dataset.
The agent + its `example` dataset are defined in this server's own `configs/scicode.yaml`.

## Execution (no Docker sandbox)

The agent sends a `solutions` dict — `{"<problem_id>.<step>": accumulated_code}` — to `/verify`. For
each scored sub-step the server builds a program (`accumulated_code` + HDF5/compare helpers + the
sub-step's sanitized test assertions) and runs it in a **Ray subprocess** (`scicode_integration/runner.py`);
exit code 0 = pass. The HDF5 target-loading and comparison helpers (`process_hdf5_to_tuple`,
`cmp_tuple_or_list`, …) are ported from nemo-skills' `scicode_utils.eval_prefix`; the only change is
that the `test_data.h5` path is injected from config rather than hardcoded.

Because the generated code runs in this server's own venv, `requirements.txt` pins `scipy<1.14`
(keeps `scipy.integrate.simps`, has Python 3.12 wheels) plus `numpy`, `matplotlib`, `h5py`, `sympy`.

**Scoring notes** (mirroring nemo-skills):
- Sub-steps absent from `solutions` (prefilled reference steps) are excluded from the denominator.
- Out-of-context sentinels (`_ran_out_of_context_`) are counted as failed sub-steps.
- `verify()` returns `reward` + `step_results`, `num_steps_passed`, `num_steps_total`,
  `problem_accuracy`. The headline `subtask_accuracy` (total passed / total over all rollouts) is
  computed on the **agent** (`/aggregate_metrics` runs there), not this server.

## Test data (required, staged manually)

`test_data.h5` (~1 GB) holds the numeric ground-truth targets. It is **not** auto-downloaded — stage
it from the official SciCode Google Drive and point `test_data_fpath` at it (see
[`benchmarks/scicode/README.md`](../../benchmarks/scicode/README.md)). If it's missing, `verify()`
fails fast with a clear error rather than scoring everything as wrong.

## Dataset Format

Two JSONL files coexist with different shapes:

- **`benchmarks/scicode/data/scicode_benchmark.jsonl`** (full 80-row dataset, gitignored; produced by
  `prepare.py`). Flat-field: each row has `problem_id`, `sub_steps`, `required_dependencies`, `uuid`.
- **`resources_servers/scicode/data/example.jsonl`** (5-row fixture, committed).

Unlike most benchmarks, SciCode's prompts are **not** materialized at rollout time — the agent builds
each sub-step's prompt itself from `sub_steps`. So rows carry an empty `responses_create_params.input`,
the dataset `prompt_config` is `null`, and you do **not** pass `+prompt_config` when collecting
rollouts.

## Running servers

```bash
gym env start --benchmark scicode --model-type openai_model
```

## Smoke test (5 example problems)

Requires `test_data.h5` staged. Use the benchmark config (so `test_data_fpath` is set) and the
benchmark agent:

```bash
gym eval run --no-serve \
    --agent scicode_benchmark_agent \
    --input resources_servers/scicode/data/example.jsonl \
    --output results/scicode_smoke.jsonl \
    --num-repeats 1 \
    --temperature 0.0
```

## Tests

```bash
gym env test +entrypoint=resources_servers/scicode
```

Covers test sanitization, the program builder, subprocess pass/fail/timeout, and the `verify()`
paths (no solutions, missing/unconfigured test data, all-pass, all-fail, prefilled-excluded,
out-of-context).

## Licensing

- **Code**: Apache 2.0
- **Data**: SciCode dataset (`SciCode1/SciCode`) + `test_data.h5` — Apache 2.0
- **Dependencies**:
  - nemo_gym: Apache 2.0
