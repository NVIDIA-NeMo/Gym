# IOI (International Olympiad in Informatics)

Gym benchmark for IOI'26, evaluated via `resources_servers/competitive_coding_challenges`.

Six problems, 83 scored subtasks, 600 points:

| day | problems |
|---|---|
| 1 | `ballmachine`, `monuments`, `tiling` |
| 2 | `classroom`, `magiccity`, `partition` |

This benchmark contributes:

- `prepare.py` — pulls everything from the official task archive
  (`github.com/ioi/task-archive`) and emits CCC-shaped artifacts: one row per
  (problem, subtask), plus a JSONL metadata file wrapped in CCC's competition
  shape and keyed by the task's short code, so a single `problem_id` string
  serves both metadata lookup and IOI's `graders/{problem_id}.cpp` filename
  convention.

- `config.yaml` — inherits the `competitive_coding_challenges` server +
  `competitive_coding_challenges_simple_agent` configs, overrides `test_file` /
  `shared_dir` to point at the benchmark's own data dir, and wires the
  `benchmark`-type dataset with its `prepare_script`.

## Preparing

```bash
python benchmarks/ioi/prepare.py
```

That is the whole setup — no HuggingFace datasets, no manual downloads. It
clones only the parts of the archive it needs (a blobless, sparse checkout), so
the other years and this year's 106 MB of translated statements never cross the
network. The download is about **390 MB**, almost all of it private test data,
cached under `data/_task_archive/` and reused on later runs.

Requires `git` and `pikepdf`.

### Where the data comes from

| artifact | source |
|---|---|
| statement | the markdown attached inside each `en.pdf` |
| subtask scores, test lists | `subtasks/*.json` |
| private tests | `tests/*.in`, `tests/*.out` |
| graders, headers, manager, stub, checker | `graders/`, `checker/` |
| compile/run harness | `huggingface/ioi`, patched to C++20 |

**Statements are extracted, not converted.** Each `en.pdf` carries the setter's
own markdown as a PDF file attachment; `prepare.py` pulls those bytes out and
verifies them against the size and MD5 the PDF itself records. No text is
scraped off the rendered page, so LaTeX, tables and code fences survive intact.

**The compile harness is patched to `-std=gnu++20`.** IOI'26 graded with C++20,
while the upstream scripts still say `gnu++17` — left alone, valid submissions
would fail to compile and score zero. The substitution is applied at download
time rather than vendored, so upstream fixes still flow through.

Sample subtasks (`00-samples`, score 0) are dropped: they carry no weight and
CCC would otherwise score a zero-weight group.

## Metrics

Emitted by `competitive_coding_challenges`:

- `total_score` — sum across problems of max per-subtask score pooled across
  rollouts. On the 0–600 IOI'26 scale.
- `per_problem_subtask_scores` — per-problem breakdown, each with
  `total.{score,max_score}` plus per-subtask `{score, max_score}`.
- Plus the standard pass@k/accuracy stats from `compute_pass_majority_metrics`.

## Sandbox prerequisite (local)

The CCC server compiles and runs candidate solutions inside the NeMo Skills
sandbox over HTTP. Bring one up locally before running the benchmark.

There is no published image — build it from the NeMo-Skills repo:

```bash
git clone --depth 1 https://github.com/NVIDIA-NeMo/Skills.git /tmp/NeMo-Skills \
  && docker build -t nemo-skills-sandbox \
       -f /tmp/NeMo-Skills/dockerfiles/Dockerfile.sandbox /tmp/NeMo-Skills \
  && docker run --rm -p 6000:6000 nemo-skills-sandbox
```

CCC defaults to `http://127.0.0.1:6000/execute`; override with
`NEMO_SKILLS_SANDBOX_HOST` / `NEMO_SKILLS_SANDBOX_PORT` if the sandbox is
elsewhere. Cluster/SLURM users can co-launch the sandbox via Skills'
`nemo_gym_rollouts(with_sandbox=True)` — separate path, not covered here.

## Running

```bash
gym dataset collate --config benchmarks/ioi/config.yaml \
  --output-dir benchmarks/ioi/data \
  --mode benchmark_preparation

gym env start \
    --benchmark ioi \
    --model-type vllm_model

gym eval run --no-serve \
  --agent ioi_simple_agent \
  --input benchmarks/ioi/data/ioi26_benchmark.jsonl \
  --output results/ioi_rollouts.jsonl \
  --num-repeats 50 \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-output-tokens 131072 \
  +num_repeats_add_seed=true
```

## Notes on the 2026 tasks

- Four of the six are `Communication` tasks graded through a `manager` process;
  `monuments` is `Batch` with a checker, and `magiccity` is `BatchAndOutput`
  with 50 scored subtasks under partial scoring. The upstream compile script
  handles all three shapes.
- No subtask is output-only, so every one is scored by running the submitted
  program — nothing requires pre-computed answer files.
- `magiccity` alone contributes 50 of the 83 scored subtasks, so per-subtask
  averages across the benchmark are dominated by it; prefer `total_score` or the
  per-problem breakdown.
