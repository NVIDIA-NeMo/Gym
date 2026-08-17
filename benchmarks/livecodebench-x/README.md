# LiveCodeBench-X

Multilingual LiveCodeBench benchmark ported from NeMo Skills'
`nemo_skills/dataset/livecodebench-x`.

## What Is Different From `livecodebench`

- Source dataset: `nvidia/Nemotron-Multilinugual-Eval-LCB`
- Languages: `de`, `es`, `fr`, `ja`
- Versions: `v5` and `v6` (same problems, different release filters; both are
  emitted into a single combined JSONL).
- Each row preserves:
  - `task_id`: LCB-canonical task id, joinable to upstream LCB.
  - `release_version`: `"v5"` or `"v6"` — used by metric stratification and
    by reviewers who want to subset to one LCB release.
  - `subset_for_metrics`: language code (mirrors Skills' field for downstream
    per-language metric breakdown).
  - `target_language`: language code (same value as `subset_for_metrics`,
    kept for symmetry with Skills' JSONL).
- Prompting mirrors Skills' `generic/default` behavior: the language-specific
  instruction prefix is baked into each row's `question`, and the prompt
  template (`benchmarks/prompts/generic/default.yaml`) is a passthrough.

## Verification

This benchmark reuses the existing `code_gen` resource server, unmodified.
`code_gen.verify()` extracts code from the model output via LCB's
`extraction_utils.extract_code(LMStyle.OpenAIChat)` and runs it against
`verifier_metadata.unit_tests` using LCB's own `testing_util.py` fork. Test
cases (public + private) are baked into each row's `verifier_metadata` at
prepare time by joining on `task_id` against the canonical
`livecodebench/code_generation_lite` (revision `refs/pr/7`) — the same data
source the existing monolingual `livecodebench/v5_2408_2502` and
`v6_2408_2505` Gym benchmarks already use.

## Data Preparation

```bash
gym eval prepare --benchmark livecodebench-x
```

That writes `benchmarks/livecodebench-x/data/livecodebench-x_benchmark.jsonl`
(~19 GB; gitignored). The size comes from LCB's hidden test suites — a few
problems carry 100–200 MB of test data each, duplicated 4× by language. The
existing monolingual `livecodebench/v5_2408_2502` and `v6_2408_2505` benchmarks
have the same characteristic; `code_gen.verify()` already handles it.

For a smaller subset (e.g. one language × one version, ~300 rows) suitable
for local smoke-testing, invoke the prepare script directly with its argparse
flags — `gym eval prepare` calls `prepare()` with no kwargs and so cannot
forward these:

```bash
python benchmarks/livecodebench-x/prepare.py --languages de --versions v5
```

`--prompt_language en` swaps the target-language instruction prefix for an
English one (matches Skills' `--prompt_language en`):

```bash
python benchmarks/livecodebench-x/prepare.py --prompt_language en
```

## Quickstart

```bash
gym env start \
    --benchmark livecodebench-x \
    --model-type vllm_model
```

Then in another shell:

```bash
mkdir -p results/livecodebench-x
gym eval run --no-serve \
    --benchmark livecodebench-x \
    --model-type vllm_model \
    --agent livecodebench-x_code_gen_simple_agent \
    --input benchmarks/livecodebench-x/data/livecodebench-x_benchmark.jsonl \
    --output results/livecodebench-x/rollouts.jsonl \
    --num-repeats 4 \
    --prompt-config benchmarks/prompts/generic/default.yaml \
    --temperature 1.0 \
    --top-p 0.95 \
    --max-output-tokens 16384 \
    +num_repeats_add_seed=true
```

`--config` and `+prompt_config` are required: the prepared JSONL ships
raw benchmark rows (no `responses_create_params.input` baked in), and the
agent's dataset-level `prompt_config` is metadata for `gym env start` only — the
rollout CLI needs `+prompt_config=...` directly to apply the prompt template
before merging `responses_create_params` overrides. `mkdir -p` is needed
because `gym eval run --no-serve` does not create parent directories.
