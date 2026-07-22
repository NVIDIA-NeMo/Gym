# GDPVal benchmark

[GDPVal](https://huggingface.co/datasets/openai/gdpval) — 220 professional
knowledge-work tasks scored by an LLM judge against per-task rubrics. This
benchmark wires the Stirrup-based agent (`responses_api_agents/stirrup_agent`)
to the GDPVal resources server (`resources_servers/gdpval`).

## Prepare data

Downloads `openai/gdpval` from HuggingFace and writes
`data/gdpval_benchmark.jsonl`:

```bash
gym eval prepare --benchmark gdpval
```

## Run rubric mode (default)

Each deliverable is scored 0–1 against the task rubric.

```bash
gym eval run \
    --model-type vllm_model \
    --benchmark gdpval \
    --output results/gdpval_rubric.jsonl \
    --split benchmark \
    --model-url <vllm_base_url> \
    --model-api-key <vllm_api_key> \
    --model <served_model_name>
```

Required environment variables for the judge:

- `JUDGE_API_KEY` — sk- key for the judge inference API (nvapi- keys 401 on
  multimodal payloads)
- `JUDGE_BASE_URL` — defaults to NVIDIA's internal inference API
- `JUDGE_MODEL_NAME` — the single-judge fallback model (used only when the
  [multi-judge panel](#multi-judge-panel) is disabled); defaults to
  `gcp/google/gemini-3.1-pro-preview`
- `HF_TOKEN` — for downloading reference files (avoids HF anonymous rate limits)

By default deliverables are graded by a **panel** of judges (GPT-5.5, Gemini 3.1
Pro Preview, Claude Opus 4.8), one sampled per call. See
[Multi-judge panel](#multi-judge-panel) for how it works and how to configure or
disable it.

## Run comparison mode (pairwise ELO vs. a reference model)

Each deliverable is judged against a reference model's deliverable for the
same `task_id`; aggregate metrics include ELO relative to a configurable
anchor (default 1000).

```bash
gym eval run \
    --model-type vllm_model \
    --benchmark gdpval \
    --output results/gdpval_compare.jsonl \
    --split benchmark \
    ++gdpval_resources_server.resources_servers.gdpval.reward_mode=comparison \
    ++gdpval_resources_server.resources_servers.gdpval.reference_deliverables_dir=/path/to/reference/output
```

The reference directory must be laid out as
`<reference_deliverables_dir>/task_<task_id>/` with `finish_params.json` and
the deliverable files (the same layout the Stirrup agent persists).

## Run multi-stage adaptive ELO (Best Practice - AA v2 Benchmark Method)

Multi-stage ELO estimates the eval model's rating in a sequence of *stages*
instead of judging every task against every reference. Each stage judges a
sampled subset of tasks against an adaptively-chosen subset of references, fits
an anchored Bradley-Terry MLE ELO, and uses that estimate to pick the references
for the next stage (typically: fewer references but more tasks as the estimate
sharpens). It runs through the **same** `gym eval run` pipeline and emits the
**same** artifacts as a normal run, so MLflow/nemo-evaluator picks it up
unchanged.

### Prerequisite

Comparison mode with two or more **`reference_models`**, each with an `elo`
anchor (the ratings the MLE is fit against). For example, in a config overlay:

```yaml
gdpval_resources_server:
  resources_servers:
    gdpval:
      reward_mode: comparison
      reference_models:
        claude_opus_4_8: {deliverables_dir: /gdpval/refs/claude_opus_4_8, elo: 1599}
        glm5_2:          {deliverables_dir: /gdpval/refs/glm5_2,          elo: 1513}
        minimax_m3:      {deliverables_dir: /gdpval/refs/minimax_m3,      elo: 1392}
        deepseek_v4_pro: {deliverables_dir: /gdpval/refs/deepseek_v4_pro, elo: 1304}
        qwen3_7_max:     {deliverables_dir: /gdpval/refs/qwen3_7_max,     elo: 1280}
        kimi_k2_6:       {deliverables_dir: /gdpval/refs/kimi_k2_6,       elo: 1193}
        nemotron_3_ultra: {deliverables_dir: /gdpval/refs/nemotron_3_ultra, elo: 1168}
        human_gold:      {deliverables_dir: /gdpval/refs/human_gold,      elo: 1000}
        qwen3_5_397b:    {deliverables_dir: /gdpval/refs/qwen3_5_397b,    elo: 956}
        gemma4_31b:      {deliverables_dir: /gdpval/refs/gemma4_31b,      elo: 781}
        gpt_oss_120b:    {deliverables_dir: /gdpval/refs/gpt_oss_120b,    elo: 775}
        gpt_oss_20b:     {deliverables_dir: /gdpval/refs/gpt_oss_20b,     elo: 519}
```

(or the equivalent `++gdpval_resources_server.resources_servers.gdpval.reference_models.<id>.{deliverables_dir,elo}=...`
CLI overrides — see `config.yaml`).

### Enable it

Add two overrides to your comparison-mode run:

```bash
gym eval run \
    --model-type vllm_model \
    --benchmark gdpval \
    --output results/gdpval_multistage.jsonl \
    --split benchmark \
    ++gdpval_resources_server.resources_servers.gdpval.reward_mode=comparison \
    ++multistage.enabled=true \
    ++multistage.stages='[{num_tasks: 5}, {num_tasks: 88, num_models: 4}]'
```

The example above runs two stages:

- **Stage 1** — `num_tasks: 5`, no `num_models` ⇒ judge 5 tasks against **all 12**
  references for a rough ELO.
- **Stage 2** — `num_tasks: 88`, `num_models: 4` ⇒ judge 88 tasks against the
  **4 references closest** to the stage-1 ELO for a tight final estimate.

For example, if Stage 1 places the eval model near **1168** (≈ Nemotron 3 Ultra),
Stage 2 zooms in on the four nearest anchors — `kimi_k2_6` (1193),
`qwen3_7_max` (1280), `deepseek_v4_pro` (1304), and `human_gold` (1000) — spending
the saved judge budget on more tasks instead of distant references like
`claude_opus_4_8` (1599) or `gpt_oss_20b` (519).

### Fresh vs. cached deliverables

- **Fresh** (generate deliverables): nothing extra. The agent persists each
  deliverable to `persist_deliverables_dir` (default `output/gdpval/deliverables`,
  overridable via the `PERSIST_DELIVERABLES_DIR` env var), and a task that recurs
  in a later stage is judged from its cached deliverable instead of re-running the
  policy.
- **Cached / judge-only** (score existing deliverables, no policy GPUs): set the
  `JUDGE_ONLY` and `PERSIST_DELIVERABLES_DIR` env vars so the agent skips the
  policy and scores the cached deliverables:

```bash
JUDGE_ONLY=true \
PERSIST_DELIVERABLES_DIR=/path/to/deliverables_cache \
gym eval run \
    --model-type vllm_model --benchmark gdpval --split benchmark \
    --output results/gdpval_multistage.jsonl \
    ++gdpval_resources_server.resources_servers.gdpval.reward_mode=comparison \
    ++multistage.enabled=true \
    ++multistage.stages='[{num_tasks: 5}, {num_tasks: 88, num_models: 4}]'
```

  The cache must contain a `task_<id>/repeat_<n>/` dir for every repeat the run
  requests (the benchmark defaults to `num_repeats: 1`, i.e. `repeat_0`; raise it
  with `++...datasets.0.num_repeats=N` and the cache needs `repeat_0`…`repeat_{N-1}`).

### Full run as a single stage

The default (no `multistage.*`) is unchanged: all tasks vs. all references. To
express the full run explicitly as a one-stage multi-stage run:

```bash
    ++multistage.enabled=true ++multistage.stages='[{num_tasks: 220}]'
```

### `multistage.*` options

| Key | Default | Meaning |
|-----|---------|---------|
| `stages` | *(required)* | List of `{num_tasks, num_models?, seed?}` (or `"N:M:seed"` strings). `num_models` omitted ⇒ all references. |
| `column` | `[occupation]` | Dataset column(s) the task sample is drawn proportionally over. |
| `distribution_path` | *(auto)* | Reuse/write the task-distribution JSON here; built from the dataset when absent. |
| `dataset_path` | *(prepared dataset)* | Dataset the distribution is built from. |
| `nested_tasks` | `false` | `true` makes each stage a superset of the previous; default samples stages independently (more information per stage). |
| `seed` | *(none)* | Seed for reproducible task sampling and reference selection. |
| `reuse_cached_deliverables` | `true` | Judge a task's cached deliverable in later stages instead of re-running the policy. |

### Resuming an interrupted multi-stage run

Set `RERUN_INCOMPLETE=true` (with the same `PERSIST_DELIVERABLES_DIR` as the
original run) to resume a staged run that was cut short. A task whose deliverable
already **finished** on disk (marked by `finish_params.json`) skips the policy
rollout and is judged from cache; a task that never finished is re-rolled. On top
of that, `rerun_incomplete` reuses **cached judgements**: the verify cache is keyed
by each stage's reference subset, so a resumed stage that reselects the same
references returns its cached judgement instead of re-judging. Use the same
`multistage.seed` so the stage task sampling — and therefore the reference subsets —
are reproducible across the resumed run. See
[Task Re-run Mode](../../responses_api_agents/stirrup_agent/README.md#task-re-run-mode)
for the full semantics.

## Multi-judge panel

By default every GDPVal deliverable is graded by a **panel** of frontier LLM
judges rather than a single model. For each scoring call one panel member is
sampled, so the reward pools verdicts across leading labs instead of trusting one
judge. The panel applies to **every** judge mode — rubric (text / visual /
structured) *and* pairwise comparison, including multi-stage ELO.

The default panel (see `benchmarks/gdpval/config.yaml`) is:

| Member | Model (default) | Reasoning |
|--------|-----------------|-----------|
| `gpt-5.5` | `openai/openai/gpt-5.5` | medium |
| `gemini-3.1-pro` | `gcp/google/gemini-3.1-pro-preview` | high (reads audio + video) |
| `claude-opus-4.8` | `aws/anthropic/bedrock-claude-opus-4-8` | thinking enabled |

All three route through the single `gdpval_judge_model` proxy server and differ
only by model id + reasoning knobs, so one judge endpoint is enough. Override the
model ids with the `JUDGE_GPT_MODEL`, `JUDGE_GEMINI_MODEL`, and
`JUDGE_CLAUDE_MODEL` env vars.

### How sampling works

- **Rubric (text/visual):** one member is sampled per task and grades the
  deliverable. Its label is recorded on the judge response as `judge_name`.
- **Structured rubric:** a member is sampled *per trial*, so the averaged score
  pools the panel across `rubric_structured_num_trials` trials
  (`metadata.trial_judges` records which graded each trial).
- **Comparison / multi-stage ELO:** a member is sampled *per pairwise trial*
  (`num_comparison_trials`), alternating position swaps as before. The response
  carries `judge_panel` (the panel that graded the rollout), `per_judge` (pooled
  eval-perspective win/loss/tie/trial counts per member), and each matchup's
  `trial_judges`.

### Reproducibility

Judge selection is seeded from a stable identity so a rerun of the same task
draws the same judges: `(task_id, "rubric")` for rubric mode and
`(task_id, ref_id, ref_repeat)` for comparison. Set `JUDGE_SAMPLING_SEED` (or
`++gdpval_resources_server.resources_servers.gdpval.judge_sampling_seed=<int>`)
to additionally shift the whole stream. This makes multi-stage ELO reruns
replayable per stage — combined with `RERUN_INCOMPLETE` the reselected reference
subset draws the same panel members it did originally.

### Audio / video routing

Audio and video capability is tracked **per modality** — a judge may read one but
not the other (e.g. MiniMax-M3 reads video but has no audio tower). Tasks whose
deliverables or references contain media (detected by extension, including inside
`.zip` archives) are routed accordingly:

- **Video**: routed to the member(s) flagged `handles_video: true` — Gemini 3.1
  Pro Preview by default, which reads video natively. If no member reads video,
  `on_missing_av_judge` decides: `warn` (default) grades with the full,
  video-blind panel and logs that the scores are unreliable; `error` fails the
  task hard.
- **Audio**: routed to the member(s) flagged `handles_audio: true`. Audio is
  always best-effort — if no routed judge reads audio (e.g. any task graded solely
  by MiniMax-M3), the audio files are **dropped with a warning** and the rest of
  the deliverable (video / images / text) is still graded. Never fatal.

Only Gemini among the frontier judges reads audio/video; GPT and Claude read
neither.

### Configuring the panel

Each member accepts:

| Field | Default | Meaning |
|-------|---------|---------|
| `name` | `model` | Label used in logs and the per-judge metrics breakdown. |
| `model` | *(legacy default)* | Upstream model id the judge endpoint expects. |
| `model_server` | `judge_model_server` | Point a member at a distinct endpoint instead of the shared proxy. |
| `create_params_overrides` | `{}` | Generation/reasoning knobs merged into `chat.completions.create` (e.g. `{reasoning_effort: high}`, `{extra_body: {...}}`). A `null` value drops a default. |
| `weight` | `1.0` | Relative sampling weight. |
| `handles_audio` | `false` | Member reads audio natively (eligible to grade audio tasks — see above). |
| `handles_video` | `false` | Member reads video natively (eligible to grade video tasks — see above). |

To grade with a **single judge** instead of the panel, set `judge_panel` to
`null` — the lone judge is then taken from `judge_model_server` +
`judge_responses_create_params_overrides`:

```bash
    ++gdpval_resources_server.resources_servers.gdpval.judge_panel=null
```

## Local multimodal judge (MiniMax-M3)

By default the judges are frontier models hosted on a third-party inference API
that decode PDFs natively. You can instead judge with a **local, open,
multimodal** model — MiniMax-M3, which reads images **and video** (it has no audio
tower — see below), so it scores most GDPVal deliverable modalities with no
third-party inference API. You host MiniMax-M3 yourself; the only twist is that
**gym does not spawn it** — gym connects to your vLLM server over its
OpenAI-compatible `/v1` endpoint.

Two pieces make this work:

1. **`judge_media_mode: images_and_text`** — a VLM can't decode a raw
   `application/pdf` data URL, so each PDF/Office-doc **page is rasterized to a
   PNG** (via PyMuPDF) and the **extracted text is attached** alongside. This
   applies to every judge mode (rubric text/visual/structured and pairwise
   comparison). See `resources_servers/gdpval/media_conversion.py`. Knobs:
   `judge_pdf_render_dpi` (default 144), `judge_pdf_max_pages` (cap per file),
   `judge_pdf_include_text` (attach the text copy). Because MiniMax-M3 reads
   video, `judge_handles_video: true` keeps video deliverables as native media
   instead of filename-only stubs. It has **no audio tower**, so
   `judge_handles_audio` stays false and audio deliverables are dropped with a
   warning (everything else is still graded).
2. **A self-hosted MiniMax-M3 endpoint** — you serve the model from the vendor's
   `vllm/vllm-openai:minimax-m3` container and gym connects to its
   OpenAI-compatible `/v1` endpoint. See
   `resources_servers/gdpval/configs/gdpval_minimax_selfhosted_judge.yaml`.

> **Why self-host instead of letting gym spawn it?** The bundled
> `vllm/models/minimax_m3/nvidia/` plugin hardcodes FlashInfer CuTe-DSL kernels
> (`gemma_rmsnorm`, fused MoE, MLA) with no fallback. On Blackwell/GB200 those
> hit the `nvidia-cutlass-dsl` 4.5.2 JIT bug (`Expected an MLIR object (got
> OpResultList)`, [vllm#45392](https://github.com/vllm-project/vllm/issues/45392))
> and abort engine startup during the profiling pass. The plugin is designed to
> run inside the vendor container, where the FlashInfer/cutlass-dsl/CUDA combo is
> matched. The gym-spawned attempt is preserved for reference in
> `gdpval_minimax_local_judge.yaml` +
> `responses_api_models/local_vllm_model/configs/MiniMaxAI/MiniMax-M3.yaml`, but
> it does **not** currently start on GB200 — use the self-hosted overlay instead.

### Run it

First serve MiniMax-M3 yourself on your GPU node(s). NVIDIA users should use
the internal serving workflow, which is maintained separately from Gym. Use a
**single-node TP=4** instance, or independent single-node TP=4 replicas for
data-parallel throughput. Do NOT form one TP=8 group across two nodes: the
cross-node MXFP8 all-reduce path on this container produces *garbage logits
that still start cleanly*.

```bash
# Verify the self-hosted endpoint from the gym host:
curl -s http://<judge-host>:5000/v1/models
```

Then point gym at that endpoint (the port and served-model name must match your
deployment):

```bash
export MINIMAX_BASE_URL=http://<judge-host>:5000/v1
export MINIMAX_MODEL=minimax-m3
export MINIMAX_API_KEY=unused

gym eval run \
    --model-type vllm_model \
    --benchmark gdpval \
    --config resources_servers/gdpval/configs/gdpval_minimax_selfhosted_judge.yaml \
    --split benchmark \
    --output results/gdpval_minimax_judge.jsonl \
    ++gdpval_resources_server.resources_servers.gdpval.reward_mode=comparison \
    ++gdpval_resources_server.resources_servers.gdpval.reference_models...=...
```

The overlay repoints the benchmark's existing `gdpval_judge_model` proxy at your
endpoint (a thin HTTP proxy — no GPU used by gym for the judge), sets
`judge_panel: null`, selects `judge_media_mode: images_and_text`, and sets
`judge_handles_video: true` / `judge_handles_audio: false`.

Notes:

- **Endpoint env vars**: `MINIMAX_BASE_URL` (must include `/v1`), `MINIMAX_MODEL`
  (defaults to `MiniMaxAI/MiniMax-M3`; match the container's
  `--served-model-name`), and `MINIMAX_API_KEY` (any non-empty string; vLLM
  ignores it).
- **Context budget**: high-DPI page images consume context faster than a
  frontier judge; the overlay caps `judge_pdf_max_pages: 30`. Lower the DPI or
  page cap if you hit `finish_reason: length`.
- **Video**: MiniMax-M3 reads video natively, so video deliverables are passed as
  native `video_url` media blocks (not filename stubs) via
  `judge_handles_video: true`.
- **Audio**: MiniMax-M3 has **no audio tower** (its `config.json` is an
  image+video VLM with no audio config), so `judge_handles_audio` stays false —
  audio deliverables are dropped with a warning and the rest of the deliverable
  (video/images/text) is still graded. Route audio tasks to an audio-capable
  judge (e.g. Gemini) if you need them scored.

## Aggregate metrics

After `gym eval run` returns, the resources server's
`/aggregate_metrics` endpoint emits headline scores in
`results/<output>_metrics.json`:

- Rubric mode: `mean/reward` (pass@1 equivalent)
- Comparison mode: `comparison/wins`, `comparison/losses`, `comparison/ties`,
  `comparison/win_rate`, `comparison/eval_elo`, `comparison/normalized_elo`
- Multi-stage mode: the headline `comparison/eval_elo` is the **last** stage's
  fit; each stage is also reported as `comparison/stage_<k>/eval_elo` (plus
  `.../num_tasks` and `.../num_references`), alongside `comparison/num_stages`.
