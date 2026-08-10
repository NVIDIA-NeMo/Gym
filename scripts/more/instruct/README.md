# Instruct model recipes

Recipes for `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B`. Most benchmarks ship as a Gym
recipe in [`gym/`](./gym/): a script that wraps `gym eval run` with the settings the
published run used. Terminal-Bench and the two SWE-bench suites are NeMo Evaluator configs
in [`nemo-evaluator/`](./nemo-evaluator/) — see [that section](#nemo-evaluator-recipes)
below.

To run a Gym recipe, copy the two templates, fill them in, and run from the Gym repo root:

```bash
cp scripts/more/instruct/gym/env.yaml.example env.yaml   # policy endpoint and judges
cp scripts/more/instruct/gym/.env.example .env           # keys and paths
set -a; source .env; set +a
scripts/more/instruct/gym/gpqa.sh
```

Run from the repo root, not from this directory: each recipe resolves its dataset and
prepare script relative to your working directory.

Every recipe accepts `LIMIT` for a quick smoke, `OUT` for the output directory,
`PARALLEL` for concurrency and `RESUME` to continue an interrupted run. Results land in
`./results/<benchmark>`.

## Prerequisites

- **Python 3.13.14 or newer** — Gym does not install on 3.12.
- **[uv](https://docs.astral.sh/uv/getting-started/installation/) on your `PATH`** — Gym
  builds every server's virtualenv with it, so nothing starts without it.
- **Gym installed**, from the repo root:

  ```bash
  uv venv --python 3.13.14 && source .venv/bin/activate
  uv sync
  ```

## Serving the model

- **Hosted:** `https://integrate.api.nvidia.com/v1/chat/completions`, model
  `nvidia/nemotron-3.5-lightning-30b-a3b`, key `NVIDIA_API_KEY`.
- **Self-hosted vLLM:** serve
  [`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B)
  — check its model card for the authoritative serving guidance; the command below is what
  these recipes were validated with, on `vllm/vllm-openai:v0.26.0`.

```bash
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B \
  --served-model-name nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B --port 8000 \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 --data-parallel-size 2 \
  --data-parallel-backend ray --data-parallel-size-local 2 --api-server-count 1 \
  --trust-remote-code --gpu-memory-utilization 0.85 \
  --enable-prefix-caching --enable-chunked-prefill \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --kv-cache-dtype fp8 --mamba-cache-mode align \
  --max-num-batched-tokens 131072 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'
```

The NeMo Evaluator recipes were served with `--gpu-memory-utilization 0.95`,
`--max-num-seqs 32` and an explicit `--max-model-len 262144` instead of
`--max-num-batched-tokens` — throughput knobs rather than output ones, so one endpoint
serves every recipe here.

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

TEST_DATA=/path/to/test_data.h5 scripts/more/instruct/gym/scicode.sh
```

It must be readable by the user running the recipe — otherwise every test case fails
and the run scores 0 with no error.

### GDPval

The agent writes real work products (spreadsheets, documents, code output) and runs
its code inside an Apptainer sandbox. Three things must exist before a run.

**1. The sandbox.** The definition ships with Gym; build the image once:

```bash
scripts/more/instruct/gym/gdpval/build-gdpval-sif.sh /abs/path/gdpval.sif
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
scripts/more/instruct/gym/gdpval/gdpval.sh
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
EXECUTE_ONLY=true OUT=./tmp scripts/more/instruct/gym/gdpval/gdpval.sh && mv ./tmp/deliverables ./refs/gptoss_120b
```

Then score your model against whatever you collected:

```bash
GDPVAL_REWARD_MODE=comparison GDPVAL_REFS=./refs scripts/more/instruct/gym/gdpval/gdpval.sh
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

One caveat on `RESUME`: a stage is journalled complete once it finishes, whether or not
any rollout in it succeeded, and a resume then skips it for good. If a stage produces
zero rollouts — an endpoint that was down the whole time, a bad key — delete
`evaluator_rollouts_multistage_state.jsonl` from `OUT` before resuming, or that stage
contributes nothing to the fit and the ELO comes out wrong with no error. Only at zero;
a stage that partly succeeded resumes correctly.

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

That shapes three things. `LIMIT` counts rollouts and repeats are grouped, so a limited
run never gathers 70 different problems and is never scored. Each key carries a daily scoring quota
and rotates only once a key is rate-limited, so pass several to get through all five calls:
`ARTIFICIAL_ANALYSIS_API_KEY='[key-1,key-2]'`. And concurrency should stay at its default
of 350, since a rollout waiting to be scored holds its slot — lower values starve the
batches and the run wedges until it times out.

If 350 is more than your endpoint can take, five single-repeat runs at 70 give the same
result. Average their `mean/reward`:

```bash
for i in 1 2 3 4 5; do
  CRITPT_REPEATS=1 PARALLEL=70 OUT=./results/critpt/run_$i scripts/more/instruct/gym/critpt.sh
done
```

Every submission and grader response is cached under `<OUT>/critpt_cache`. If a run dies
on quota, re-score it once the quota resets without repeating any inference:

```bash
python -m resources_servers.critpt.replay --cache-dir <OUT>/critpt_cache
```

## NeMo Evaluator recipes

Terminal-Bench 2.1, SWE-bench Verified and SWE-bench Multilingual are still being migrated
to Gym and for now still belong to
[NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator). Until that migration lands
they are published here as YAML configs in [`nemo-evaluator/`](./nemo-evaluator/) and are
run with `nel eval run` rather than as Gym recipes.

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

Then run whichever config you want, adding `--resume` to continue an interrupted run:

```bash
nel eval run scripts/more/instruct/nemo-evaluator/terminal-bench-2.1.yaml
```
