# Reproducing the published evaluation results

This tutorial demonstrates how to reproduce the evaluation results for the NVIDIA
Nemotron 3.5 Lightning 30B A3B model using NeMo Gym.

Most benchmarks ship as a recipe: a script that wraps `gym eval run` with the settings
the published run used. To run one, copy the two templates, fill them in, and run from
the Gym repo root:

```bash
cp scripts/more/env.yaml.example env.yaml   # policy endpoint and judges
cp scripts/more/.env.example .env           # keys and paths
set -a; source .env; set +a
scripts/more/gpqa.sh
```

Run from the repo root, not from this directory: each recipe resolves its dataset and
prepare script relative to your working directory.

Every recipe accepts `LIMIT` for a quick smoke, `OUT` for the output directory,
`PARALLEL` for concurrency and `RESUME` to continue an interrupted run. Results land in
`./results/<benchmark>`.

Terminal-Bench, SWE-bench and the base-model suite are not Gym recipes — each has its
own section below.

## Prerequisites

- **Python 3.13.14 or newer** — Gym does not install on 3.12.
- **[uv](https://docs.astral.sh/uv/getting-started/installation/) on your `PATH`** — Gym
  builds every server's virtualenv with it, so nothing starts without it.
- **Gym installed**, from the repo root:

  ```bash
  uv venv --python 3.13.14 && source .venv/bin/activate
  uv sync
  ```

## Overview

| Benchmark | What it measures | Run |
|---|---|---|
| GPQA Diamond | Graduate-level science questions | `scripts/more/gpqa.sh` |
| HLE | Humanity's Last Exam — hard, broad knowledge | `scripts/more/hle.sh` |
| AA-LCR | Long-context reasoning | `scripts/more/aa-lcr.sh` |
| AA-Omniscience | Knowledge and hallucination | `scripts/more/omniscience.sh` |
| [SciCode](#scicode) | Scientific coding, graded against numeric ground truth | `scripts/more/scicode.sh` |
| BrowseComp | Web browsing agent | `scripts/more/browsecomp/browsecomp.sh` |
| Tau3 | Tool use against a simulated customer | `scripts/more/tau3.sh` |
| [CritPt](#critpt) | Research-level physics | `scripts/more/critpt.sh` |
| [GDPval](#gdpval) | Real-world work products, judged | `scripts/more/gdpval/gdpval.sh` |
| [PinchBench](#pinchbench) | The model as the brain of an OpenClaw agent | `scripts/more/pinchbench.sh` |
| [Terminal-Bench 2.1](#terminal-bench-and-swe-bench-nel-next) | Agentic terminal use | `nel eval run scripts/more/nel-next/terminal-bench-2.1.yaml` |
| [SWE-bench Verified](#terminal-bench-and-swe-bench-nel-next) | Agentic coding against a repo's tests | `nel eval run scripts/more/nel-next/swebench-verified.yaml` |
| [SWE-bench Multilingual](#terminal-bench-and-swe-bench-nel-next) | The same, across languages | `nel eval run scripts/more/nel-next/swebench-multilingual.yaml` |
| [Base suite](#base-pretraining-model) | 21 pretraining benchmarks plus RULER | see [`base/`](./base/) |

A linked name needs setup beyond the prerequisites above — follow the link. The rest run
as they are.

## Resuming an interrupted run

Every gym-native recipe takes `RESUME=1`, which passes `--resume` and continues from what
is on disk. Pass the same `OUT` you used before:

```bash
RESUME=1 OUT=./results/browsecomp scripts/more/browsecomp/browsecomp.sh
```

Completed rollouts are kept; rollouts still in flight when the run died are re-run.

The `nel-next/` benchmarks track runs themselves:

```bash
nel eval run --resume scripts/more/nel-next/terminal-bench-2.1.yaml
nel eval resume <run-id>
```

**GDPval** (comparison mode) records a stage as complete even if every rollout in it
failed, so a later resume skips it forever. If an attempt ends with zero rollouts, delete
`evaluator_rollouts_multistage_state.jsonl` before resuming — only at zero.

**CritPt** resumes within the current pass. If the Artificial Analysis quota runs out,
re-score the cached submissions instead of regenerating them:
`python -m resources_servers.critpt.replay --cache-dir "$CRITPT_CACHE_DIR/<run-subdir>"`

## Benchmarks needing extra setup

Most recipes need nothing beyond the prerequisites above. These do:

### SciCode

Scoring needs `test_data.h5` (~1 GB) — the numeric ground-truth the test cases are
checked against. It is not downloaded automatically. Get it from the official SciCode
[Google Drive folder](https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR),
save it anywhere, and point `TEST_DATA` at it:

```bash
sha256sum /path/to/test_data.h5
# 48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890

TEST_DATA=/path/to/test_data.h5 scripts/more/scicode.sh
```

It must be readable by the user running the recipe — otherwise every test case fails
and the run scores 0 with no error.

### GDPval

The agent writes real work products (spreadsheets, documents, code output) and runs
its code inside an Apptainer sandbox. Three things must exist before a run.

**1. The sandbox.** The definition ships with Gym; build the image once:

```bash
scripts/more/gdpval/build-gdpval-sif.sh /abs/path/gdpval.sif
export GDPVAL_CONTAINER_PATH=/abs/path/gdpval.sif
```

You do **not** need to install the document-generation stack yourself. The agent writes
deliverables inside the sandbox, and the image already carries LibreOffice plus
`python-docx`, `python-pptx`, `openpyxl`, `fpdf2`, `reportlab`, `weasyprint` and the
rest.

**2. Root, for the PDF conversion step.** Office deliverables are converted to PDF on
the host — not in the sandbox — and the resources server prepares LibreOffice on
startup by running:

```bash
apt-get install -y libreoffice fonts-liberation default-jre-headless libreoffice-java-common
```

That call needs root and returns early if it fails, so pre-installing the packages
doesn't substitute. Without root the conversion silently becomes a no-op — the run
exits 0, but deliverables reach the judge unconverted and score lower with no error.
Run as root, or in a container where you are.

**3. A judge endpoint serving all three panel models.** Deliverables are graded by a
panel — GPT-5.5, Gemini 3.1 Pro and Claude Opus 4.8, one sampled per call — all routed
through the `gdpval_judge_model` block in `env.yaml`, so one endpoint has to serve all
three.

For a full run give `TAVILY_API_KEY` several keys — `'[tvly-1,tvly-2]'` — since a single
key rate-limits and the agent then cannot search.

#### Rubric mode (default)

A judge scores each deliverable against its task rubric.

```bash
scripts/more/gdpval/gdpval.sh
```

#### Comparison mode

Scores deliverables pairwise against reference models **you generate yourself**. Point
`GDPVAL_REFS` at a directory holding one subdirectory per reference, named after the
model:

```
refs/
  gptoss_120b/        task_<id>/repeat_0/...
  gemma4_26b/         task_<id>/repeat_0/...
```

Produce each one with an execute-only run of that model — point the policy in
`env.yaml` at it, then:

```bash
EXECUTE_ONLY=true OUT=./tmp scripts/more/gdpval/gdpval.sh && mv ./tmp/deliverables ./refs/gptoss_120b
```

Then score your model against whatever you collected:

```bash
GDPVAL_REWARD_MODE=comparison GDPVAL_REFS=./refs scripts/more/gdpval/gdpval.sh
```

The recipe recognises these names and anchors each at the
[Artificial Analysis GDPval-AA v2](https://artificialanalysis.ai/evaluations/gdpval-aa)
rating the published figures were fitted against. The live board has moved since, so
these deliberately no longer match it:

| Reference | ELO | | Reference | ELO |
|---|---|---|---|---|
| `deepseek_v4_pro` | 1307 | | `qwen35_397b` | 962 |
| `glm51_fp8` | 1257 | | `gptoss_120b` | 799 |
| `kimi_k26` | 1191 | | `gemma4_26b` | 761 |
| `nemotron3_ultra` | 1164 | | `qwen3_30b_thinking` | 308 |
| `qwen36_35b` | 1049 | | | |

Supply **two or more** and the run switches to the two-stage fit used for the published
numbers: a 45-task pass over all of them to place the model, then the full task set
against the four nearest. Supply all nine and the method matches the reference run.

`JUDGE_ONLY=true` re-scores deliverables you already have, and needs neither the
sandbox nor a search key.

#### Reproducing our numbers

We ran comparison mode: deliverables judged against all nine reference models above,
fitted in two stages, graded by the three-judge panel. The scale is Artificial Analysis'
GDPval-AA v2 ELO, anchored to a human baseline of 1000 — so roughly 1000 is human-level
on these tasks.

The closest you can get is to generate those nine reference sets yourself and run the
same way: same method, same anchors, same scale. It will land near our number rather
than on it — a gap of tens of points is noise here, not a regression.

That is also the expensive path: nine reference sets means nine full 220-task agentic
runs before you score your own. Rubric mode is one run and no references. It gives a
self-contained score rather than an ELO, which is enough if you are comparing your own
runs rather than positioning against published models.

### PinchBench

Each task runs in its own sandbox, built from the image definition that ships with Gym.
Build it once — this needs **Docker**, because the Apptainer path converts a Docker
image rather than building directly:

```bash
export PINCHBENCH_SIF=/abs/path/pinchbench.sif
bash responses_api_agents/pinchbench/setup_scripts/build_image.sh --apptainer
```

The build script writes to `PINCHBENCH_SIF` when it is set, so exporting it first points
both the build and the recipe at the same file.

On a host without Docker, build the image elsewhere and copy the `.sif` across.

The policy endpoint is given to the recipe directly rather than through `env.yaml`,
because OpenClaw calls it itself: set `PINCHBENCH_MODEL_BASE_URL`,
`PINCHBENCH_MODEL_API_KEY` and `PINCHBENCH_MODEL_NAME` alongside the judge and search
keys — all listed in `.env.example`.

### CritPt

The Artificial Analysis API scores in batches of exactly 70 distinct problems, so a full
run — 70 problems repeated five times — costs five scoring calls.

That shapes three things. `LIMIT` counts problems rather than rollouts, and below 70 no
batch can fill, so those rollouts are never scored. Each key carries a daily scoring quota
and rotates only once a key is rate-limited, so pass several to get through all five calls:
`ARTIFICIAL_ANALYSIS_API_KEY='[key-1,key-2]'`. And concurrency should stay at its default
of 350, since a rollout waiting to be scored holds its slot — lower values starve the
batches and the run wedges until it times out.

If 350 is more than your endpoint can take, five single-repeat runs at 70 give the same
result. Average their `mean/reward`:

```bash
for i in 1 2 3 4 5; do
  CRITPT_REPEATS=1 PARALLEL=70 OUT=./results/critpt/run_$i scripts/more/critpt.sh
done
```

Every submission and grader response is cached under `<OUT>/critpt_cache`. If a run dies
on quota, re-score it once the quota resets without repeating any inference:

```bash
python -m resources_servers.critpt.replay --cache-dir <OUT>/critpt_cache
```

## Terminal-Bench and SWE-bench (`nel-next/`)

Terminal-Bench 2.1, SWE-bench Verified and SWE-bench Multilingual are still being migrated
to Gym and for now still belong to
[NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator). Until that migration lands
they are published here as YAML configs in `nel-next/` and are run with `nel eval run`
rather than as Gym recipes.

Two things have to be in place first.

**1. NeMo Evaluator, installed from `main`.**

```bash
pip install "nemo-evaluator[harbor] @ git+https://github.com/NVIDIA-NeMo/Evaluator.git@main"
```

The SWE-bench Multilingual config sets `sandbox.scrub_git_history`, which the 0.3.0
release on PyPI does not recognise — it refuses to load the file. Do not delete that line to
silence the error: it strips each task repository's later commits, and without it the
agent can read the official fix straight out of the git history and score far too high.

**2. An AWS sandbox.** Every task runs in its own ECS Fargate container, and that
infrastructure is not created for you. Apply the reference Terraform stack from the
[NeMo Evaluator repository](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/terraform)
in your own AWS account, in the region you plan to run in, then point
`sandbox.ecr_repository` in the config at that account's harbor ECR. Export
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` next to `POLICY_API_KEY`.

Then run whichever config you want:

```bash
nel eval run nel-next/terminal-bench-2.1.yaml
```

## Base (pretraining) model

The recipes in this directory cover the **instruct** model. Recipes for the **base**
model live in [`base/`](./base/) — 21 short-context benchmarks plus RULER for
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16`.

They work differently: they are `nemo-evaluator-launcher` configs run with
`nel run --config ...`, not Gym CLI scripts, because the base benchmarks are
`lm-evaluation-harness` and `nemo-skills` tasks rather than Gym environments. See
[`base/README.md`](./base/README.md) for their prerequisites, which differ from the ones
above.

## License

Apache 2.0 — see the repository `LICENSE`.
