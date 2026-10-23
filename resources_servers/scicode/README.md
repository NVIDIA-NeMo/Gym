# SciCode Resources Server

Verifies multi-step scientific code generation on the
[SciCode](https://huggingface.co/datasets/SciCode1/SciCode) benchmark. The agent submits the code it
generated for each sub-step; this server executes each sub-step's accumulated code against
ground-truth targets in `test_data.h5` and returns the reward.

- Task type: multi-step code generation (one LLM call per sub-step) + local code execution
- Domain: `coding`
- Tasks: 65 problems / 288 evaluated sub-steps (`test` only, matching the current AA Intelligence Index setup)
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

**Per-sub-step execution records (opt-in).** By default the server keeps only the pass/fail bit per
sub-step: the subprocess's stdout is never read and its stderr is discarded after the exit code is
known. Set `record_step_outputs: true` to add `step_outputs` to the verify response (and therefore to
every rollout row): one record per scored sub-step, aligned with `step_results`, with `step_id`,
`step_number`, `environment` (the grading interpreter whose run the record describes: under the OR rule
the one that passed, else the last one tried; `null` for a sentinel step), `status` (`executed`, or
`out_of_context` / `empty` for sentinels that never ran), `passed`, `returncode` (`null` on a timeout),
`timed_out`, `duration_s`, `exception_class` (the class named on the last traceback line, e.g.
`AssertionError`, `ImportError`, `numpy.linalg.LinAlgError`; `null` when there is none) and the last
`output_tail_chars` characters (default 2000; 0 keeps none, negative values are rejected) of `stdout_tail`
and `stderr_tail`. The per-environment
pass bits and error text of SciCode-Verified's OR grading stay in `step_environment_results` /
`step_environment_errors`; the record adds the execution details of the decisive run. The tails contain model-generated output and tracebacks, so the flag is off by
default and the rows grow by up to `2 * output_tail_chars` per sub-step when it is on. With
`record_step_code: true` as well, each record also carries `code`: the accumulated program the
sub-step's tests ran against (the agent's `solutions[step_id]`: the problem's dependency header and any
prefilled reference steps from the dataset, then the model's code for the previous sub-steps and this
one; `null` for a sentinel step that never ran). That mixes dataset text with model-generated code and
grows a row by the size of every scored program, so it is a second opt-in. The agent's raw per-step
generations are a separate concern: enable model-call observability to keep them.

```yaml
scicode:
  resources_servers:
    scicode:
      record_step_outputs: true
      output_tail_chars: 2000
      record_step_code: true
```

## Test data (required, staged manually)

`test_data.h5` (~1 GB) holds the numeric ground-truth targets. It is **not** auto-downloaded — stage
it from the official SciCode Google Drive and point `test_data_fpath` at it (see
[`benchmarks/scicode/README.md`](../../benchmarks/scicode/README.md)). If it's missing, `verify()`
fails fast with a clear error rather than scoring everything as wrong.

## Dataset Format

Two JSONL files coexist with different shapes:

- **`benchmarks/scicode/data/scicode_benchmark.jsonl`** (65-row test split, gitignored; produced by
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

Covers test sanitization, the program builder, subprocess pass/fail/timeout and the execution
details it reports (exit code, timeout, duration, bounded output tails, exception class), and the
`verify()` paths (no solutions, missing/unconfigured test data, all-pass, all-fail, prefilled-excluded,
out-of-context, `step_outputs` on and off, `code` on and off).

## Licensing

- **Code**: Apache 2.0
- **Data**: SciCode dataset (`SciCode1/SciCode`) + `test_data.h5` — Apache 2.0
- **Dependencies**:
  - nemo_gym: Apache 2.0
