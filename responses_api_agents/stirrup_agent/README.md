# Stirrup Agent: GDPVal Evaluation Environment

A NeMo Gym responses API agent that uses the [Stirrup](https://github.com/ArtificialAnalysis/Stirrup)
agent-loop framework to evaluate language models on [GDPVal](https://huggingface.co/datasets/openai/gdpval) —
a benchmark of real-world professional knowledge-work tasks across sectors like finance, law,
healthcare, and engineering.

## Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Full GDPVal Evaluation](#full-gdpval-evaluation)
- [Configuration](#configuration)
- [Advanced Features](#advanced-features)
  - [Apptainer Sandboxing](#apptainer-sandboxing)
  - [Pairwise ELO Judging](#pairwise-elo-judging)
  - [Tavily Web Search](#tavily-web-search)
- [Extending to New Tasks](#extending-to-new-tasks)
- [Licensing](#licensing)

## Overview

Stirrup Agent is a pluggable agent wrapper built on the Stirrup framework. Task-specific
logic (prompt construction, scoring, file handling) lives in a `TaskStrategy` — this repo
ships the GDPVal strategy out of the box, and new benchmarks can be added in a single file.

For each GDPVal task, the agent:
1. Receives a professional prompt (e.g. *"Prepare a patent filing brief..."*), an optional
   set of reference files, and a scoring rubric.
2. Runs a tool-using loop (shell, code execution, file I/O, optional web search) until it
   produces one or more deliverable files.
3. A judge LLM scores each deliverable against the rubric, producing a reward in `[0, 1]`.

## How It Works

```
┌─────────────┐   prompt    ┌─────────────┐   tool calls   ┌──────────────┐
│ Input JSONL │ ──────────► │  Stirrup    │ ─────────────► │  sandbox     │
│  (task)     │             │  Agent      │                │ (optionally  │
└─────────────┘             │  (policy    │ ◄───────────── │  Apptainer)  │
                            │   model)    │   tool results └──────────────┘
                            └──────┬──────┘
                                   │ deliverables
                                   ▼
                            ┌─────────────┐   rubric score
                            │  Judge LLM  │ ─────────────►  reward ∈ [0, 1]
                            └─────────────┘
```

## Dataset

- **Source**: [`openai/gdpval`](https://huggingface.co/datasets/openai/gdpval) — 220 tasks
  across 9 occupational sectors. Each task contains a prompt, optional reference files, and
  a scoring rubric.
- **Download**:
  ```bash
  bash responses_api_agents/stirrup_agent/setup_scripts/gdpval.sh
  ```
  This writes `responses_api_agents/stirrup_agent/data/gdpval.jsonl` (220 tasks).
- **Smoke-test example**: `responses_api_agents/stirrup_agent/data/example.jsonl` ships with
  one synthetic task for fast iteration (no network required).

## Prerequisites

1. **Install NeMo Gym** (see the [top-level README](../../README.md)):
   ```bash
   uv venv --python 3.12 && source .venv/bin/activate
   uv sync --extra dev --group docs
   ```
2. **Install document-generation dependencies** (needed for the GDPVal deliverable formats —
   `.docx`, `.xlsx`, `.pptx`, `.pdf`):
   ```bash
   uv pip install python-docx fpdf2 reportlab weasyprint PyPDF2 \
                  beautifulsoup4 seaborn python-pptx markdown2 \
                  pdfminer.six openpyxl lxml Pillow
   # System-level (Ubuntu/Debian):
   sudo apt install libreoffice libpango1.0-dev libcairo2-dev libgdk-pixbuf2.0-dev
   ```
3. **(Optional) Install Apptainer** if you want sandboxed code execution
   (see [Apptainer Sandboxing](#apptainer-sandboxing)).

## Quick Start

The canonical entry point for GDPVal is the benchmark at
[`benchmarks/gdpval/`](../../benchmarks/gdpval/README.md), which composes this
agent with the GDPVal resources server and supports
`gym eval prepare` + `gym eval run`:

```bash
# 1. Prepare the GDPVal benchmark JSONL.
gym eval prepare --benchmark gdpval

# 2. Collect rollouts end-to-end (servers spin up automatically).
JUDGE_API_KEY=... HF_TOKEN=... \
gym eval run \
  --model-type openai_model \
  --benchmark gdpval \
  --split benchmark \
  --output results/gdpval_rubric.jsonl \
  --model-url https://api.openai.com/v1 \
  --model-api-key $OPENAI_API_KEY \
  --model gpt-4.1-2025-04-14
```

Each output line contains `responses_create_params`, the full `response`, a
`reward` in `[0, 1]`, and a `judge_response` with per-criterion breakdown.
Aggregate metrics (`mean/reward` for rubric mode, ELO for comparison mode)
land in `results/gdpval_rubric_metrics.json`.

## Configuration

The agent reads its Hydra config at `configs/stirrup_gdpval.yaml`. Notable keys:

| Key | Default | Meaning |
|-----|---------|---------|
| `task` | `gdpval` | Which `TaskStrategy` to use. |
| `agent_max_turns` | `100` | Turn cap for the agent loop. |
| `concurrency` | `32` | Stirrup's internal parallelism per worker. |
| `temperature` | `1.0` | Policy sampling temperature. |
| `system_prompt_template` | `???` | Path to the system prompt Jinja2 template. |
| `user_prompt_template` | `???` | Path to the user prompt Jinja2 template. |
| `resources_server` | required | Reference to the GDPVal resources server (which scores the deliverable via `/verify`). |
| `gdpval_container_path` | `null` | Path to an Apptainer `.sif` (see below). |
| `persist_deliverables_dir` | `null` | If set, each task's artifacts land in `<dir>/task_<task_id>/`. The resources server reads this dir to score the deliverable. |
| `judge_only` | `false` | If true, skip task execution and score the deliverables already cached under `persist_deliverables_dir` (see [Judge-Only Mode](#judge-only-mode)). |
| `model_id` | `null` | HF model id or local path used to load a tokenizer for dynamic output sizing. |
| `completion_token_buffer` | `1000` | Safety margin (in tokens) reserved when sizing `max_completion_tokens` per call. |

Env vars honored: `TAVILY_API_KEY`, `HF_TOKEN`, `OPENAI_API_KEY`.

### Dynamic `max_completion_tokens` sizing

Stirrup's `ChatCompletionsClient` sends a static
`max_completion_tokens = self._max_tokens` on every call.  For long-context
models (Ultra V3, Qwen3-Coder-30B's 131K, etc.), this can exceed
`max_model_len − prompt_tokens` once the prompt grows, and the server
returns an HTTP 400 (or `finish_reason=length` with zero output) that the
agent cannot recover from.

The wrapper ships a `DynamicMaxTokensChatCompletionsClient`
(`nemo_client.py`) that, on every request:

1. Tokenises the message history + tool schemas with a HuggingFace
   `AutoTokenizer` loaded from `model_id`.
2. Computes `max_completion_tokens = context_window − input_tokens − completion_token_buffer`.
3. Replicates upstream's response parsing but does **not** raise
   `ContextOverflowError` on `finish_reason=length`; the agent loop
   terminates normally via the `finish` tool or `max_turns`.

Set `model_id` to the same checkpoint (or HF id) you are serving via
vLLM and the tokeniser match is exact.  Leave it unset and the client
falls back to a conservative character-count estimate — slower to
allocate completion budget but always safe.  `completion_token_buffer`
absorbs the residual gap between our estimate and the exact prompt the
server renders (chat-template wrappers, tool-schema injection).  The
default `1000` works in practice; raise it (e.g. 2000–5000) if you see
sporadic HTTP 400 responses at the vLLM proxy.

## Advanced Features

### Judge-Only Mode

`judge_only` re-scores a set of deliverables that an earlier run already
produced, **without** re-executing the (expensive) agent task. It is the
counterpart to a plain execute-then-judge run: the agent loop is skipped
entirely and only the resources server `/verify` step runs.

How it works:

- For each task, the agent locates the cached deliverable directory at
  `persist_deliverables_dir/task_<task_id>/repeat_<rollout_index>/` — the same
  layout a normal run writes when `persist_deliverables_dir` is set.
- If the directory exists, a placeholder response is built and `/verify` is
  called with `deliverables_dir` pointing at the cached files, so the judge
  scores the on-disk deliverables (not fresh model output).
- If no cached directory exists for a task, that task is reported as
  **skipped** (terminal — re-dispatching it would not create the files) and
  `/verify` is never called.

Requirements / notes:

- `persist_deliverables_dir` must be set (and absolute); it is the source of
  the deliverables to score. The agent raises at startup otherwise.
- Run judge-only over the same benchmark / `num_repeats` that produced the
  cache so the `task_<id>/repeat_<n>` directories line up.

Enable it via the config key or the `JUDGE_ONLY` env var honored by
`benchmarks/gdpval/config.yaml`:

```bash
JUDGE_ONLY=true PERSIST_DELIVERABLES_DIR=/abs/path/to/cached/deliverables \
  ng_e2e_collect_rollouts ...
```

or as a Hydra override:

```bash
++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.judge_only=true
```

### Apptainer Sandboxing

Some GDPVal tasks ask the model to install packages or run untrusted code. By default the
agent uses a local sandbox; setting `gdpval_container_path` to an Apptainer `.sif` routes
all `code_exec` calls through a persistent container.

Build the supplied container definition:

```bash
apptainer build gdpval.sif responses_api_agents/stirrup_agent/containers/gdpval.def
```

Then:

```yaml
# env.yaml or gym env start override
stirrup_agent:
  responses_api_agents:
    stirrup_agent:
      gdpval_container_path: /abs/path/to/gdpval.sif
```

### Pairwise ELO Judging

Pairwise comparison vs. a reference model is built into the GDPVal resources
server (`resources_servers/gdpval`). Drive it from the benchmark config:

```bash
gym eval run \
  --model-type vllm_model \
  --benchmark gdpval \
  --split benchmark \
  --output results/gdpval_compare.jsonl \
  ++gdpval_resources_server.resources_servers.gdpval.reward_mode=comparison \
  ++gdpval_resources_server.resources_servers.gdpval.reference_deliverables_dir=output/gdpval/reference-model
```

LibreOffice preconversion of Office docs runs inside `verify()` automatically;
ELO is computed in `aggregate_metrics()`. See `benchmarks/gdpval/README.md`
for the full recipe.

### Tavily Web Search

To give the agent web access (some GDPVal tasks benefit from fresh facts), set
`TAVILY_API_KEY` in the environment. The agent automatically exposes `web_search` and
`web_fetch` tools backed by the [Tavily Search API](https://tavily.com).

### Task-Only Execution Mode

Sometimes you want to *run* the tasks and keep their deliverables without judging
them — e.g. to build a reference set for later pairwise comparison, to defer
scoring to a separate pass, or to inspect raw model outputs. Set `execute_only:
true` (or export `EXECUTE_ONLY=true` with the benchmark config) to do exactly
that:

- Each task runs through the Stirrup agent and its deliverables are cached to
  `persist_deliverables_dir/task_<task_id>/repeat_<n>/` (the same layout used by
  comparison mode's `reference_deliverables_dir`).
- The judge `/verify` call is **skipped entirely** — no judgement is made or
  sent, and no LLM-judge tokens are spent.
- Each rollout row carries the `response`, the `deliverables_dir`, and
  `execute_only: true`, but **no `reward`** and **no `judge_response`**.
- `aggregate_metrics` returns baseline (reward-free) stats instead of proxying
  to the judge server.

`execute_only: true` **requires** `persist_deliverables_dir` to be set to an
absolute path — without it nothing is saved and the mode is rejected at startup.

```bash
EXECUTE_ONLY=true \
PERSIST_DELIVERABLES_DIR=/abs/path/to/output/gdpval/my-model \
HF_TOKEN=... \
ng_e2e_collect_rollouts \
  "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml,benchmarks/gdpval/config.yaml]" \
  ++split=benchmark \
  ++output_jsonl_fpath=results/gdpval_execute_only.jsonl
```

The cached deliverables can later be scored with a separate rubric or
comparison run by pointing the resources server at the same directory.

## Extending to New Tasks

To add a benchmark `my_bench`:

1. Implement `responses_api_agents/stirrup_agent/tasks/my_bench.py` as a `TaskStrategy`
   subclass (`extract_task_info`, `build_system_prompt`, `build_user_prompt`,
   `score_deliverable`).
2. Register it in `app.py:_load_task_registry()`.
3. Add `configs/stirrup_my_bench.yaml` setting `task: my_bench`.

That's it — the agent loop, sandboxing, caching, and rollout collection are shared.

## Licensing

- **Code**: Apache License 2.0 (see repository `LICENSE`).
- **Dependencies**: `stirrup` (Apache 2.0), `jinja2` (BSD 3-Clause), `datasets` (Apache 2.0),
  `python-docx`, `openpyxl`, `PyPDF2`, etc. See `requirements.txt` and the top-level
  `pyproject.toml` for full attribution.
- **Dataset**: GDPVal is released by OpenAI at
  [huggingface.co/datasets/openai/gdpval](https://huggingface.co/datasets/openai/gdpval).
  Refer to that page for dataset licensing terms.
