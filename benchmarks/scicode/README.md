# SciCode Benchmark

Benchmark wrapper for [SciCode](https://huggingface.co/datasets/SciCode1/SciCode), a scientific
code-generation benchmark. Each problem is decomposed into sub-steps; the model implements one
Python function per sub-step, with each sub-step building on the code it generated for previous
sub-steps. Generated code is executed against the problem's test cases.

- **Tasks**: 65 problems / 288 evaluated sub-steps (`test` only, matching the current AA Intelligence Index setup)
- **Reward**: binary per problem — `1.0` only if every sub-step passes its tests
- **Metrics** (reported by the agent): `subtask_accuracy` — the headline SciCode number,
  total sub-steps passed divided by total sub-steps across all rollouts — and `problem_accuracy`
  (= `mean/reward`, the whole problem passing)

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
gym eval prepare --benchmark scicode
```

Downloads `SciCode1/SciCode` and writes `benchmarks/scicode/data/scicode_benchmark.jsonl`
(one row per problem from the HuggingFace `test` split, carrying the full `sub_steps` list). This
does not fetch `test_data.h5` — see above.

The agent uses the current AA SciCode prompt in `benchmarks/scicode/prompts/background.yaml`.
The scientist-authored background in each sub-step is included in the prompt; the model is asked
to write the next function directly.

## Dependencies

The resources server executes generated SciCode solutions in its own venv, so its
`requirements.txt` pins `scipy<1.14` — the last range that keeps `scipy.integrate.simps` (used by
some test cases, removed in scipy 1.14) while still providing Python 3.12 wheels — plus `numpy`,
`matplotlib`, `h5py`, and `sympy`.

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --benchmark scicode
```

Requires `policy_base_url` / `policy_api_key` / `policy_model_name` in `env.yaml` (or passed as CLI
overrides).

## Collect rollouts

With the servers up, run the full benchmark:

```bash
gym eval run --no-serve \
    --agent scicode_benchmark_agent \
    --input benchmarks/scicode/data/scicode_benchmark.jsonl \
    --output results/scicode_rollouts.jsonl \
    --num-repeats 3 \
    --temperature 0.0
```

(For a quick smoke, point `--input` at `resources_servers/scicode/data/example.jsonl`
with `--num-repeats 1`.)

### One-shot alternative

Runs prepare + servers + rollout collection and tears the servers down afterwards — this is the
full-benchmark run that produces the headline `subtask_accuracy`. Requires `test_data.h5` staged
(see above).

```bash
gym eval run \
    --model-type vllm_model \
    --benchmark scicode \
    --split benchmark \
    --output results/benchmarks/scicode.jsonl \
    ++reuse_existing_data_preparation=true \
    ++overwrite_metrics_conflicts=true \
    --model-url <your_endpoint> \
    --model-api-key <your_key> \
    --model <your_model> \
    --temperature 0.0
```

`num_repeats: 3` comes from Gym's dataset config and can be overridden independently of the
AA-aligned problem set and prompt.

## Licensing

Code: Apache 2.0
Data (`SciCode1/SciCode`, `test_data.h5`): Apache 2.0
