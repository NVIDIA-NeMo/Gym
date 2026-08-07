# Reproducing the published evaluation results

## What these are

## Prerequisites

- **Python 3.13.14 or newer.** Gym does not install on 3.12.
- **`uv` on your `PATH`.** Some benchmarks shell out to scripts that call it and
  fail with a bare `exit 127` without it.

## Running a recipe

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

TEST_DATA=/path/to/test_data.h5 ./scicode.sh
```

It must be readable by the user running the recipe — otherwise every test case fails
and the run scores 0 with no error.

### GDPval

The agent writes real work products (spreadsheets, documents, code output) and runs
its code inside an Apptainer sandbox. Three things must exist before a run.

**1. The sandbox.** The definition ships with Gym; build the image once:

```bash
./build-gdpval-sif.sh /abs/path/gdpval.sif
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
through the `gdpval_judge_model` block in `env.yaml`. `JUDGE_API_KEY` must be an `sk-`
key: `nvapi-` keys return 401 on the multimodal payloads this benchmark sends.

#### Rubric mode (default)

A judge scores each deliverable against its task rubric.

```bash
./gdpval.sh
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
EXECUTE_ONLY=true OUT=./tmp ./gdpval.sh && mv ./tmp/deliverables ./refs/gptoss_120b
```

Then score your model against whatever you collected:

```bash
GDPVAL_REWARD_MODE=comparison GDPVAL_REFS=./refs ./gdpval.sh
```

The recipe recognises these names and anchors each at its rating from the
[Artificial Analysis GDPval-AA v2 board](https://artificialanalysis.ai/evaluations/gdpval-aa),
snapshot 2026-07-04:

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

Those values are pinned to that snapshot rather than tracking the live board, because
the anchors define the scale a score sits on. The board moves as models are added and
re-run, and Artificial Analysis publishes more than one ELO scale, so a rating only
means something alongside the scale and date it came from.

A directory with no recognised names is treated as a single unrated baseline, anchored
at `GDPVAL_REFERENCE_ELO` (default 1290).

`JUDGE_ONLY=true` re-scores deliverables you already have, and needs neither the
sandbox nor a search key.

#### What the numbers mean

With several rated references your score is fitted on the published scale, so it is
comparable **in kind** to published GDPval figures — though not identical to ours,
since your reference deliverables are your own rather than the ones our run used.

With a single reference, or an unrated baseline, the anchor only shifts the scale and
the informative output is `comparison/win_rate` rather than the ELO. Rubric mode is a
self-contained score.

For a full run, give `TAVILY_API_KEY` several keys — `'[tvly-1,tvly-2]'` — since a
single key rate-limits and the agent then cannot search.

### Terminal-Bench and SWE-bench (`nel-next/`)

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

## Results

## Comparing your numbers

## Base (pretraining) model

The recipes in this directory cover the **instruct** model. Recipes for the **base**
model live in [`base/`](./base/) — 21 short-context benchmarks plus RULER for
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16`.

They work differently: they are `nemo-evaluator-launcher` configs run with
`nel run --config ...`, not Gym CLI scripts, because the base benchmarks are
`lm-evaluation-harness` and `nemo-skills` tasks rather than Gym environments. See
[`base/README.md`](./base/README.md) for their prerequisites, which differ from the ones
above.
