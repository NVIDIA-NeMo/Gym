# IOI (International Olympiad in Informatics)

Gym benchmark for IOI'26, evaluated via `resources_servers/competitive_coding_challenges`.

This benchmark contributes:

- `prepare.py` — pulls everything from the official task archive
  (`github.com/ioi/task-archive`) and emits CCC-shaped artifacts: one row per
  (problem, subtask), plus a JSONL metadata file wrapped in CCC's competition
  shape and keyed by the task's short code, so a single `problem_id` string
  serves both metadata lookup and IOI's `graders/{problem_id}.cpp` filename
  convention.

- `config.yaml` — inherits the `competitive_coding_challenges` server +
  `competitive_coding_challenges_simple_agent` configs, overrides `test_file`
  to point at the benchmark's own data dir, and wires the `benchmark`-type
  dataset with its `prepare_script`.


## Metrics

The benchmark sets `grade_all_subtasks: true` and `partial_credit: true`, so every
rollout is graded on every subtask of its problem with partial credit, as IOI and NeMo Skills score it, and
the totals are comparable with NeMo Skills' CCC numbers. They are not comparable with totals from earlier
Gym versions, which gave a subtask full points when only the tests it shares with the row's own subtask
passed, and gave no partial credit. Both totals are on the 0–600 IOI'26 scale.

Emitted by `competitive_coding_challenges`:

- `total_score` — sum across problems of max per-subtask score pooled
  across rollouts (Skills' `pass@k` total).
- `pass@1[avg-of-k]/total_score` — per prompt row and subtask, the mean score
  over that row's rollouts; per subtask, the best row; summed (Skills'
  `pass@1[avg-of-k]` total).
- `per_problem_subtask_scores` — per-problem breakdown, each with
  `total.{score,max_score}` plus per-subtask `{score, max_score}`.
- Plus the standard pass@k/accuracy stats from `compute_pass_majority_metrics`.

## Sandbox

The CCC server compiles and runs every submission in a fresh sandbox from a Gym sandbox provider;
nothing runs on the server's host. Build the image, split this benchmark's metadata into one file per
problem, and write a provider config that mounts it, as
[the CCC README](../../resources_servers/competitive_coding_challenges/README.md#sandbox) describes.
After `gym eval prepare --benchmark ioi26`:

```bash
export CCC_SANDBOX_IMAGE=<registry>/ccc-verifier:<tag>
python -m resources_servers.competitive_coding_challenges.split_problem_metadata \
    benchmarks/ioi26/data/ioi26_metadata.json benchmarks/ioi26/data/problems
```

Then mount `benchmarks/ioi26/data/problems` read-only at `/data/ccc`, for example with the CCC README's
`ccc_docker.yaml` (`<problem_dir>` set to the absolute path of `benchmarks/ioi26/data/problems`).

## Running

```bash
gym eval prepare --benchmark ioi26

gym env start \
    --benchmark ioi26 \
    --config ccc_docker.yaml \
    --model-type vllm_model

gym eval run --no-serve \
  --agent ioi26_simple_agent \
  --input benchmarks/ioi26/data/ioi26_benchmark.jsonl \
  --prompt-config benchmarks/prompts/generic/default.yaml \
  --output results/ioi_rollouts.jsonl \
  --num-repeats 50 \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-output-tokens 131072 \
  +num_repeats_add_seed=true
```
