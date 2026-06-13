# SciCode Benchmark

Benchmark wrapper for [SciCode](https://huggingface.co/datasets/SciCode1/SciCode), a scientific
code-generation benchmark. Each problem is decomposed into sub-steps; the model implements one
Python function per sub-step, with each sub-step building on the code it generated for previous
sub-steps. Generated code is executed against the problem's test cases.

- **Tasks**: 80 problems / 341 sub-steps (`validation` + `test` combined — the split nemo-skills calls `test_aai`)
- **Reward**: binary per problem — `1.0` only if every sub-step passes its tests
- **Metrics** (reported by the resources server): `subtask_accuracy` — the headline SciCode number,
  total sub-steps passed divided by total sub-steps across all rollouts (matches nemo-skills'
  `pass@1[avg-of-3]/subtask_accuracy`) — and `problem_accuracy` (= `mean/reward`, the whole problem
  passing)

A custom multi-step agent (`responses_api_agents/scicode_agent`) drives the per-sub-step
generation loop; the resources server (`resources_servers/scicode`) executes each sub-step's
accumulated code and runs its test cases.

## Test data (required manual download)

Scoring needs `test_data.h5` — the numeric ground-truth targets the test-case assertions compare
against. It is **not** downloaded automatically. Stage it from the official SciCode source (a
[Google Drive folder](https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR?usp=drive_link))
and save it as `benchmarks/scicode/data/test_data.h5` (the path the resources server reads via its
`test_data_fpath` config).

On a headless machine / cluster, download it with `gdown`:

```bash
uv pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR" \
    -O benchmarks/scicode/data
# If it lands under a subdirectory or a different name, move it to:
#   benchmarks/scicode/data/test_data.h5
```

Verify the download (~1 GB):

```bash
sha256sum benchmarks/scicode/data/test_data.h5
# expect: 48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890
```

You can also point `test_data_fpath` at an absolute path via config override.
If the file is missing, the resources server fails fast with a clear error rather than scoring everything as wrong.

## Prepare benchmark data

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/scicode/config.yaml]"
```

Downloads `SciCode1/SciCode` and writes `benchmarks/scicode/data/scicode_benchmark.jsonl`
(one row per problem, carrying the full `sub_steps` list). This does not fetch `test_data.h5` — see
above.

## Dependencies

The resources server executes generated SciCode solutions in its own venv, so its
`requirements.txt` pins `scipy<1.14` — the last range that keeps `scipy.integrate.simps` (used by
some test cases, removed in scipy 1.14) while still providing Python 3.12 wheels — plus `numpy`,
`matplotlib`, and `h5py`.

## Running servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/scicode/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

Requires `policy_base_url` / `policy_api_key` / `policy_model_name` in `env.yaml` (or passed as CLI
overrides).

## Collect rollouts

```bash
ng_collect_rollouts \
    +agent_name=scicode_benchmark_agent \
    +input_jsonl_fpath=benchmarks/scicode/data/example.jsonl \
    +output_jsonl_fpath=results/scicode_rollouts.jsonl \
    +num_repeats=1
```

Use `benchmarks/scicode/data/scicode_benchmark.jsonl` for the full run; `example.jsonl` is a 5-problem
smoke set.

## Licensing

Code: Apache 2.0
Data (`SciCode1/SciCode`, `test_data.h5`): Apache 2.0
