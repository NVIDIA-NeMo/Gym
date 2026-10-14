# Reproducing the published evaluation results

This tutorial demonstrates how to reproduce the evaluation results for the NVIDIA
Nemotron 3.5 Super VL model (`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`) using
NeMo Gym.

Every benchmark is one Gym config here, `<benchmark>.yaml`, holding the settings the
published run used; one shared script, `run.sh`, runs any of them.

## Quick start

To run a recipe, copy the two templates, fill them in, and run from the Gym repo root:

```bash
cp nemotron_recipes/super-3.5/env.yaml.example env.yaml   # model endpoint and judges
cp nemotron_recipes/super-3.5/.env.example .env           # keys and paths
set -a; source .env; set +a
nemotron_recipes/super-3.5/run.sh gpqa
```

Run from the repo root, not from this directory: each recipe resolves its dataset and
prepare script relative to your working directory. `run.sh` without arguments lists the
recipes.

Each recipe's header lists what it needs from `.env` (`Set in .env:`); `.env.example`
explains each of them.

Every recipe accepts `LIMIT` for a quick smoke (except CritPt, see its header), `OUT` for
the output directory, `PARALLEL` for concurrency (16 by default for SWE-bench and
Terminal-Bench, 4 VMs for OSWorld, unlimited for the rest) and `RESUME` to continue an
interrupted run. `LIMIT` keeps the first N rows of the prepared data, where a question's
repeats come one after another, so `LIMIT=3` is usually 3 samples of the first question
(APEX: 3 problems x 16):

```bash
LIMIT=3 nemotron_recipes/super-3.5/run.sh gpqa     # quick smoke; not a score
RESUME=1 nemotron_recipes/super-3.5/run.sh gpqa    # continue an interrupted run
```

Results land in `./results/<recipe>`; the scores are in `evaluator_rollouts_aggregate_metrics.json`.

`run.sh` first switches the repo's files to the Gym commit the recipe's published number was
produced with, leaving the recipes alone, and syncs your venv to that commit's lockfile. It
overwrites and deletes files and changes your venv, so run the recipes from a separate copy of
Gym (a fresh clone, or `git worktree add`), not the checkout you work in, and one recipe at a
time per copy. Set `PIN_GYM=0` to run against your current checkout instead.

## What you can reproduce

| Benchmark | What it measures | Recipe |
|---|---|---|
| GPQA Diamond | Graduate-level science questions | [`gpqa.yaml`](./gpqa.yaml) |
| HLE (text-only) | Humanity's Last Exam — hard, broad knowledge | [`hle.yaml`](./hle.yaml) |
| HLE (full) | The same, all questions including those with images | [`hle-vision.yaml`](./hle-vision.yaml) |
| AA-LCR | Long-context reasoning | [`aa-lcr.yaml`](./aa-lcr.yaml) |
| AA-Omniscience | Knowledge and hallucination | [`omniscience.yaml`](./omniscience.yaml) |
| SciCode | Scientific coding, graded against numeric ground truth | [`scicode.yaml`](./scicode.yaml) † |
| CritPt | Research-level physics | [`critpt.yaml`](./critpt.yaml) † |
| Tau3-Banking | Tool use as a bank agent, against a simulated customer | [`tau3.yaml`](./tau3.yaml) |
| Tau2 | Tool use as a retail, airline or telecom agent, against a simulated customer | [`tau2.yaml`](./tau2.yaml) |
| LiveCodeBench v6 | Competitive programming, graded by running the code against tests | [`livecodebench.yaml`](./livecodebench.yaml) |
| IFBench | Following verifiable output constraints | [`ifbench.yaml`](./ifbench.yaml) |
| GraphWalks | Long-context graph traversal | [`graphwalks.yaml`](./graphwalks.yaml) |
| APEX Shortlist | Hard competition math | [`apex-shortlist.yaml`](./apex-shortlist.yaml) |
| MMLU-ProX | MMLU-Pro's multiple-choice questions in 10 languages | [`mmlu-prox.yaml`](./mmlu-prox.yaml) |
| WMT24++ | Translation from English into 9 languages, scored by xCOMET-XXL | [`wmt24pp.yaml`](./wmt24pp.yaml) † |
| SWE-bench Verified | Agentic coding against a repo's tests | [`swebench-verified.yaml`](./swebench-verified.yaml) † |
| SWE-bench Multilingual | The same, across nine languages | [`swebench-multilingual.yaml`](./swebench-multilingual.yaml) † |
| SWE-bench Pro | The same, on larger and harder tasks | [`swebench-pro.yaml`](./swebench-pro.yaml) † |
| Terminal-Bench 2.1 | Agentic terminal use | [`terminal-bench-2.1.yaml`](./terminal-bench-2.1.yaml) † |
| OSWorld-Verified | Computer use on a real Ubuntu desktop | [`osworld.yaml`](./osworld.yaml) † |
| BrowseComp | Answering hard questions by searching and reading the web | [`browsecomp.yaml`](./browsecomp.yaml) |
| GDPval-AA v2 | Real-world work products (reports, spreadsheets, slide decks), judged | [`gdpval.yaml`](./gdpval.yaml) † |

† needs setup beyond installing Gym — see
[Benchmarks needing extra setup](#benchmarks-needing-extra-setup). The rest run as they are
once `env.yaml` and `.env` are filled in (some need a judge or simulator endpoint there, see
[Judges and graders](#judges-and-graders); BrowseComp also needs an [Exa](https://exa.ai)
search key).

## How a recipe is built

Each recipe is one YAML file, `<benchmark>.yaml`, naming the Gym commit
(`gym_commit`) and Gym benchmark config (`benchmark_config`) it builds on, plus whatever
that benchmark needs beyond the shared settings in [`common.yaml`](./common.yaml): thinking
on, temperature 1.0, top-p 0.95 and no output cap (CritPt sets its run's cap). `run.sh`
loads the benchmark config, the model config, `common.yaml` and the recipe, in that order,
each overriding the ones before.

`env.yaml` is applied over all of them, so keep it to endpoints and keys: a sampling or
benchmark setting there would silently replace the published one.

## Prerequisites

- **Python 3.13.14 or newer**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/) on your `PATH`** — Gym
  builds every server's virtualenv with it, so nothing starts without it.
- **Gym installed**, from the repo root:

  ```bash
  uv venv --python 3.13.14 && source .venv/bin/activate
  uv sync
  ```

## Serving the model

Reproducing these numbers depends on the model being served the right way. Check the
[model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16)
for the authoritative serving guidance.

The published runs served the model with vLLM 0.30.0, on one node of 4 GPUs per server:

```bash
# TODO: confirm public vLLM version and flags before release.
vllm serve nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 \
  --served-model-name nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 --port 8000 \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --trust-remote-code --gpu-memory-utilization 0.8 \
  --enable-prefix-caching --enable-chunked-prefill --async-scheduling --api-server-count 1 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --kv-cache-dtype fp8 --mamba-cache-dtype auto --mamba-ssm-cache-dtype float32 \
  --no-enable-mamba-cache-stochastic-rounding \
  --max-model-len 262144 --max-num-seqs 1024 --max-num-batched-tokens 135680 \
  --limit-mm-per-prompt '{"image":128,"video":1}' --video-pruning-rate 0.5 \
  --media-io-kwargs '{"video":{"fps":2,"num_frames":256}}' \
  --skip-mm-profiling --mm-encoder-attn-backend TORCH_SDPA \
  --hf-overrides '{"architectures":["NemotronH_Super_Omni_Reasoning_V3"],"use_thumbnail":false,"vision_config":{"args":{"model":"vit_huge_patch16_224","min_num_patches":1024,"max_num_patches":13312,"num_cls_tokens":3,"num_registers":7,"summary_idxs":[0,1]}},"norm_mean":[0.48145466,0.4578275,0.40821073],"norm_std":[0.26862954,0.26130258,0.27577711]}'
```

Every recipe except SWE-bench and Terminal-Bench 2.1 (below) was run against this. MMLU-ProX and
WMT24++ were served with `--max-model-len 32768`, which also capped each answer; their recipes
set `max_output_tokens: 32768` instead, so they need no change. Two recipes changed it:

| Recipe | Change | Why it matters |
|---|---|---|
| GraphWalks | `--max-model-len 1048576 --max-num-batched-tokens 8192` | Needed: prompts are up to ~361,000 tokens |
| OSWorld | `--max-num-batched-tokens 32768` | Optional: with the default above, the published setup repeatedly ran out of GPU memory on OSWorld; lower it if yours does. Does not change answers |

The vision flags (`--limit-mm-per-prompt` and the rest) matter for HLE (full) and OSWorld,
which send images; text-only recipes ignore them. SWE-bench and Terminal-Bench 2.1 were served
with the same parsers, context and cache dtypes but without the vision flags, split across
separate prefill and decode servers (2 to 8 nodes each) for throughput.

CritPt asks for 245,760 output tokens per turn and needs `--max-model-len 262144`. With a
shorter context its turns come back empty and score 0 without an error.

## Judges and graders

Some recipes are graded by another model or service, or need one to play the user: HLE (text-only
and full), AA-LCR, AA-Omniscience, BrowseComp and GDPval-AA v2 (judge models), Tau2 and Tau3
(user simulator) and CritPt (Artificial Analysis API). Each recipe pins the model it uses; you
provide its endpoint in `env.yaml` and its key in `.env`, separately for each recipe, so the
judges can come from different providers. Both templates have a block per recipe, and the
recipe's header says which API your endpoint must serve.

## Benchmarks needing extra setup

Most recipes need nothing beyond the prerequisites, `env.yaml` and `.env`. These do:

### SciCode

Scoring needs `test_data.h5` (~1 GB), the numeric ground truth the test cases are
checked against. It is not downloaded automatically. Get it from the official SciCode
[Google Drive folder](https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR),
save it anywhere, and point `TEST_DATA` at it:

```bash
sha256sum /path/to/test_data.h5
# 48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890

TEST_DATA=/path/to/test_data.h5 nemotron_recipes/super-3.5/run.sh scicode
```

It must be readable by the user running the recipe, otherwise every test case fails
and the run scores 0 with no error.

### CritPt

The Artificial Analysis API grades CritPt in batches of all 70 problems, so a full run (70
problems x 5 repeats) makes 5 grading calls.

- Leave `PARALLEL` unset, or at 350 or more. A rollout holds its slot until its batch is
  graded, so with fewer slots no batch fills and the run fails once the 6-hour verify
  timeout expires.
- `LIMIT` does not work: a limited run never collects 70 different problems, so it is never
  graded. The recipe's header shows how to smoke it instead.

### WMT24++

Translations are scored by xCOMET-XXL, which runs on the GPUs of the machine that runs Gym,
one copy per GPU:

- **GPUs.** Each copy needs 43 GB for the model's weights, plus room for the translations it
  scores at once (`comet_batch_size` in `wmt24pp.yaml`, 16). By default `run.sh` uses every
  GPU with 75 GB free. To include GPUs with less, list the GPUs to use in
  `CUDA_VISIBLE_DEVICES`; on Slurm it is set to all your job's GPUs, so set it to the ones
  vLLM does not use. If scoring runs out of memory, lower `comet_batch_size`: it changes how
  fast the scoring runs, not the scores.
- **The model.** [Unbabel/XCOMET-XXL](https://huggingface.co/Unbabel/XCOMET-XXL) is gated:
  accept its terms on the Hugging Face Hub and set `HF_TOKEN`. `run.sh` downloads it (43 GB)
  on the first run, into `HF_HOME` or `./cache/huggingface`.

### SWE-bench Verified, Multilingual and Pro

Each task runs in a container managed by
[OpenSandbox](https://github.com/opensandbox-group/OpenSandbox), the sandbox the published
runs used, and the model works in it through the OpenCode agent. Run the OpenSandbox server on
the Linux machine that runs Gym, with Docker installed:

1. Download OpenCode, as the published run did (many task images have no `curl` to fetch it
   themselves):

   ```bash
   mkdir -p ~/opencode/glibc ~/opencode/musl
   curl -L https://github.com/sst/opencode/releases/download/v1.17.11/opencode-linux-x64.tar.gz | tar xz -C ~/opencode/glibc
   curl -L https://github.com/sst/opencode/releases/download/v1.17.11/opencode-linux-x64-musl.tar.gz | tar xz -C ~/opencode/musl
   ```

   The musl build is for SWE-bench Pro's Alpine-based task images.

2. Generate a config:

   ```bash
   uvx opensandbox-server==1.1.1 init-config ./sandbox.toml --example docker
   ```

   In `sandbox.toml`, set three lines:
   - `api_key` under `[server]` (uncomment it): a long random string (e.g. `openssl rand -hex 24`).
   - `publish_host = "127.0.0.1"` under `[docker]` (uncomment it). Each sandbox runs a command
     service that takes no key; this keeps it off your network, though other users of this
     machine can still reach it.
   - `allowed_host_paths` under `[storage]`: your OpenCode folder and this recipes folder,
     as absolute paths, e.g.
     `["/path/to/opencode", "/path/to/Gym/nemotron_recipes/super-3.5"]`. The recipes mount
     both read-only into each sandbox; OpenSandbox refuses any other folder (and every folder
     while the list is empty, whatever its comment says).

3. Start the server and keep it running:

   ```bash
   uvx opensandbox-server==1.1.1 --config ./sandbox.toml
   ```

4. Point Gym at it: in `.env`, set `OPENSANDBOX_DOMAIN=127.0.0.1:8080`,
   `OPENSANDBOX_API_KEY` to the `api_key` above and `OPENCODE_DIR` to your OpenCode folder
   (absolute path).

OpenCode runs inside the sandboxes and calls Gym's model server back, so these recipes bind
Gym's servers to Docker's bridge gateway (`default_host: 172.17.0.1`), reachable from the
sandboxes but not from your network. If `ufw` is active (`sudo ufw status`), it drops those
calls too, and each task hangs until its 3-hour timeout. Allow them on Gym's server ports
(10001-20000 by default):

```bash
sudo ufw allow in on docker0 to 172.17.0.1 port 10001:20000 proto tcp
```

Each task runs in its own container on this machine, 16 at a time by default (`PARALLEL`
changes it). A container can use up to 2 CPUs and 16 GB of memory (Verified and
Multilingual) or 4 CPUs and 30 GB (Pro), so keep `PARALLEL` within what your machine has.
Each task also has its own Docker image of a few GB, pulled from Docker Hub on first use: run
`docker login` first to avoid Docker Hub's pull limits, and leave room for the images on disk.

### Terminal-Bench 2.1

Each task runs in a container managed by OpenSandbox, as for SWE-bench: follow steps 2-4 of
[SWE-bench](#swe-bench-verified-multilingual-and-pro), without OpenCode (step 1,
`allowed_host_paths` and `OPENCODE_DIR` are for SWE-bench only). One server serves both.

Each task runs in its own container on this machine, 16 at a time by default (`PARALLEL`
changes it), with up to 4 CPUs and 16 GB of memory each, so keep `PARALLEL` within what your
machine has. Task images come from Docker Hub: run `docker login` first to avoid its pull
limits.

The tasks come from
[harbor-framework/terminal-bench-2-1](https://github.com/harbor-framework/terminal-bench-2-1);
`terminal-bench-2.1.setup.sh` pins them to the published run's commit, `36d417f5`.

### OSWorld-Verified

Each task runs in its own Ubuntu VM, in Docker on the machine that runs Gym. You need:

- Linux x86_64 with Docker, and read and write access to `/dev/kvm` (`ls -l /dev/kvm`;
  usually by being in the `kvm` group).
- About 16 GB of RAM per VM running at once: 4 by default, `PARALLEL` changes it.
- Room for the VM disk: `run.sh` downloads it on the first run with Gym's own script (about
  11 GB download, 24.5 GB on disk, checksum verified) into `OSWORLD_VM_DIR`, by default
  `docker_vm_data` in the Gym repo root. Later runs only re-check the checksum.

### GDPval-AA v2

The agent writes real work products (spreadsheets, documents, code output) and runs its code
inside an Apptainer sandbox; judges then grade what it produces. Two things must exist
before a run.

**1. The sandbox.** The definition ships with Gym; build the image once. Install
[Apptainer](https://apptainer.org/docs/admin/main/installation.html) and run `run.sh gdpval`
once first: it pins Gym, then stops and asks for the image. Then, from the repo root (several
GB, tens of minutes):

```bash
cd responses_api_agents/stirrup_agent/containers
apptainer build --fakeroot "$PWD/gdpval.sif" gdpval.def
export GDPVAL_CONTAINER_PATH=$PWD/gdpval.sif
cd -
```

You do **not** need to install the document-generation stack yourself. The agent writes
deliverables inside the sandbox, and the image already carries LibreOffice plus `python-docx`,
`openpyxl` and the rest. Its Python packages are pinned; its Debian packages take the current
Debian 13 point release, so a later build differs from the published run's image by security
updates only. On aarch64, five pinned packages that ship only x86_64 wheels are left out.

**2. Root.** The sandbox is started with `--home /root`, which an ordinary user cannot enter.
Office deliverables are also converted to PDF on the host — not in the sandbox — and the
resources server prepares LibreOffice on startup by running:

```bash
apt-get install -y libreoffice fonts-liberation default-jre-headless libreoffice-java-common
```

That call needs root, and pre-installing the packages doesn't substitute: if it fails,
comparison mode refuses to start, and rubric mode goes on with the judges seeing Office files
by name only, which scores them lower. Run as root, or in a container where you are.

#### Rubric mode (default)

A judge scores each deliverable against its task rubric. This needs no references, but it is
not the published metric.

```bash
nemotron_recipes/super-3.5/run.sh gdpval
```

#### Comparison mode

Scores deliverables pairwise against reference models **you generate yourself**. Point
`GDPVAL_REFS` at a directory holding one subdirectory per reference, named after the model:

```
refs/
  gptoss_120b/        task_<id>/repeat_0/...
  gemma4_26b/         task_<id>/repeat_0/...
```

Produce each one with an execute-only run of that model — point the policy in `env.yaml` at it,
then:

```bash
EXECUTE_ONLY=true OUT=./tmp nemotron_recipes/super-3.5/run.sh gdpval && mv ./tmp/deliverables ./refs/gptoss_120b
```

Then score your model against whatever you collected:

```bash
GDPVAL_REFS=./refs nemotron_recipes/super-3.5/run.sh gdpval
```

The recipe recognises these names and anchors each at the rating the published number was
fitted against. They are pinned in [`gdpval.setup.sh`](./gdpval.setup.sh): other ratings would
move your score off that scale.

| Reference | ELO | | Reference | ELO |
|---|---|---|---|---|
| `deepseek_v4_flash_0731` | 1468 | | `qwen36_35b` | 992 |
| `deepseek_v4_pro` | 1223 | | `qwen35_122b` | 925 |
| `inkling_small` | 1191 | | `qwen35_397b` | 905 |
| `glm51_fp8` | 1181 | | `gptoss_120b` | 745 |
| `kimi_k26` | 1115 | | `gemma4_26b` | 713 |
| `nemotron3_ultra` | 1091 | | `qwen3_30b_thinking` | 274 |

The run fits in two stages, as the published one did: 45 tasks against every reference you
supplied, then all 220 tasks against the four nearest. A single reference only fixes an
arbitrary offset; supply all twelve and the method matches the published run. The published
number is `comparison/stage_1/eval_elo`, the last stage's.

`JUDGE_ONLY=true` re-scores deliverables you already have, and needs neither the sandbox nor a
search key.

#### Reproducing our numbers

We ran comparison mode: deliverables judged against all twelve reference models above, fitted
in two stages, graded by the three-judge panel. The closest you can get is to generate those
twelve reference sets yourself and run the same way: same method, same anchors, same scale.

That is also the expensive path: twelve reference sets means twelve full 220-task agentic runs
before you score your own. Rubric mode is one run and no references. It gives a self-contained
score rather than an ELO, which is enough if you are comparing your own runs rather than
positioning against published models.

`PARALLEL` sets both Gym's and the agent's concurrency (32 by default; the published run used
220). A full comparison run takes hours and makes about a thousand judge calls. `LIMIT` works
in rubric mode only: comparison mode samples each stage's tasks from the whole benchmark.

## What to expect from your numbers

**Where the published numbers are.** Table 10 of the Nemotron 3.5 Super tech report. Each
recipe's header names the metric to compare (`Score:`): the key in
`evaluator_rollouts_aggregate_metrics.json` behind each of its rows in that table. A published
number is the mean of two runs, except CritPt, BrowseComp and AA-Omniscience (one run each).

**Run the whole benchmark.** Every recipe except CritPt can be cut short for a smoke test
with `LIMIT`. A limited run tells you the setup works. It is not a score, and should not be
reported as one.

**Expect run-to-run variation.** These benchmarks sample at `temperature > 0` and average
over repeats, so two runs of the same recipe against the same endpoint will not agree to
the last digit. Judge a difference against that spread, not against the exact published
value.

**Serve the model yourself to compare.** To compare against the published numbers, serve the
model as [Serving the model](#serving-the-model) describes and point the recipes at it.
Running against an endpoint someone else operates works fine, but then a gap is ambiguous:
you cannot tell whether it came from their serving stack or from the model. That is a sound
way to measure a deployment you already have — it is not a basis for confirming published
numbers.

**Check for failed answers.** An answer that fails, e.g. on a judge call, is left out of the
score and written to `evaluator_rollouts_failures.jsonl` in the output directory. Compare
with the published number only if that file is empty; otherwise rerun with `RESUME=1`.

## License

Apache 2.0 — see the repository `LICENSE`.
