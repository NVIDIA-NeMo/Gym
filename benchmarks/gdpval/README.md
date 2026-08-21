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
instead of judging every task against every reference. Each stage samples `T`
tasks (`T` is configurable per stage and **defaults to the full task set** — all
220 GDPVal tasks) and assigns **each task a single reference model** drawn
uniformly at random (every included reference weighted equally) from the stage's
adaptively-chosen set of references. It then fits an anchored Bradley-Terry MLE
ELO — pooling each reference's win/loss/tie counts over the tasks assigned to it
— and uses that estimate to pick the references for the next stage (typically
narrowing to fewer references closest to the estimate, and growing `T`, so each
reference gets a larger share of tasks as the estimate sharpens). Within each
judged comparison a panel judge is still sampled per trial with equal
probability. It runs through the **same** `gym eval run` pipeline and emits the
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
        deepseek_v4_pro:    {deliverables_dir: /gdpval/refs/deepseek_v4_pro,    elo: 1299}
        glm51_fp8:          {deliverables_dir: /gdpval/refs/glm51_fp8,          elo: 1250}
        kimi_k26:           {deliverables_dir: /gdpval/refs/kimi_k26,           elo: 1191}
        nemotron3_ultra:    {deliverables_dir: /gdpval/refs/nemotron3_ultra,    elo: 1160}
        qwen36_35b:         {deliverables_dir: /gdpval/refs/qwen36_35b,         elo: 1045}
        qwen35_397b:        {deliverables_dir: /gdpval/refs/qwen35_397b,        elo: 960}
        gptoss_120b:        {deliverables_dir: /gdpval/refs/gptoss_120b,        elo: 775}
        gemma4_26b:         {deliverables_dir: /gdpval/refs/gemma4_26b,         elo: 752}
        qwen3_30b_thinking: {deliverables_dir: /gdpval/refs/qwen3_30b_thinking, elo: 267}
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
    ++multistage.stages='[{num_tasks: 45}, {num_tasks: 220, num_models: 4}]'
```

The example above runs two stages:

- **Stage 1** — `num_tasks: 45`, no `num_models` ⇒ sample **45** tasks and
  include **all 9** references; each task is judged against **one** randomly-assigned reference for a rough ELO.
- **Stage 2** — `num_tasks: 220` and `num_models: 4` ⇒ the **full 220-task set** against only the **4 references closest** to the stage-1 ELO; each task is judged against one of those four, concentrating the larger task budget on the nearest anchors for a tight final estimate.

For example, if Stage 1 places the eval model near **1170**,
Stage 2 zooms in on the four nearest anchors — `nemotron3_ultra` (1160), `kimi_k26` (1191),
`glm51_fp8` (1250), and `qwen36_35b` (1045) — spending
the full task budget on those instead of distant references like
`gptoss_120b` (775) or `qwen3_30b_thinking` (267).

`num_tasks` is optional per stage; omit it to judge the full task distribution
(the default). Every task is still compared against a single sampled reference.

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
    ++multistage.stages='[{num_tasks: 45}, {num_models: 4}]'
```

  The cache must contain a `task_<id>/repeat_<n>/` dir for every repeat the run
  requests (the benchmark defaults to `num_repeats: 1`, i.e. `repeat_0`; raise it
  with `++...datasets.0.num_repeats=N` and the cache needs `repeat_0`…`repeat_{N-1}`).

### Full run as a single stage

The default (no `multistage.*`) is unchanged: all tasks vs. all references (each
deliverable judged against every configured reference). A single **multi-stage**
stage differs — it samples `T` tasks (defaulting to the full set) but assigns
each task just one reference — so it is *not* equivalent to the non-multistage
full run:

```bash
    ++multistage.enabled=true ++multistage.stages='[{num_tasks: 220}]'
```

### `multistage.*` options

| Key | Default | Meaning |
|-----|---------|---------|
| `stages` | *(required)* | List of `{num_tasks?, num_models?, seed?}` (or `"[num_tasks]:[num_models]:seed"` strings). `num_tasks` omitted ⇒ full task set; `num_models` omitted ⇒ all references. |
| `column` | `[occupation]` | Dataset column(s) the task sample is drawn proportionally over. |
| `distribution_path` | *(auto)* | Reuse/write the task-distribution JSON here; built from the dataset when absent. |
| `dataset_path` | *(prepared dataset)* | Dataset the distribution is built from. |
| `nested_tasks` | `false` | `true` makes each stage's task sample a superset of the previous; default samples each stage independently. |
| `seed` | *(none)* | Seed for reproducible task sampling, per-task reference assignment, and reference selection. |
| `reuse_cached_deliverables` | `true` | Judge a task's cached deliverable in later stages instead of re-running the policy. |

### Resuming an interrupted multi-stage run

Set `RERUN_INCOMPLETE=true` (with the same `PERSIST_DELIVERABLES_DIR` as the
original run) to resume a staged run that was cut short. A task whose deliverable
already **finished** on disk (marked by `finish_params.json`) skips the policy
rollout and is judged from cache; a task that never finished is re-rolled. On top
of that, `rerun_incomplete` reuses **cached judgements**: the verify cache is keyed
by each task's assigned reference, so a resumed stage that reassigns the same
reference to a task returns its cached judgement instead of re-judging. Each
stage's sampled tasks and per-task reference assignment are recorded in the stage
journal (and re-derived deterministically from the stage seed when a recorded
plan predates them), so they replay identically on resume; use the same
`multistage.seed` so a fresh re-plan of any not-yet-started stage draws the same
tasks and assignment. See
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
| `gemini-3.1-pro` | `gcp/google/gemini-3.1-pro-preview` | high (handles audio/video) |
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

Tasks whose deliverables or references contain audio or video files (detected by
extension, including inside `.zip` archives) are routed to the panel member(s)
flagged `handles_audio_video: true` — Gemini 3.1 Pro Preview by default, which
reads those modalities natively. The whole rollout for that task is graded by the
AV-capable subset, and comparison responses set `av_routed: true`. If no member
is flagged AV-capable, the full panel is used unchanged (best-effort).

### Configuring the panel

Each member accepts:

| Field | Default | Meaning |
|-------|---------|---------|
| `name` | `model` | Label used in logs and the per-judge metrics breakdown. |
| `model` | *(legacy default)* | Upstream model id the judge endpoint expects. |
| `model_server` | `judge_model_server` | Point a member at a distinct endpoint instead of the shared proxy. |
| `create_params_overrides` | `{}` | Generation/reasoning knobs merged into `chat.completions.create` (e.g. `{reasoning_effort: high}`, `{extra_body: {...}}`). A `null` value drops a default. |
| `weight` | `1.0` | Relative sampling weight. |
| `handles_audio_video` | `false` | Eligible to grade audio/video tasks (see above). |

To grade with a **single judge** instead of the panel, set `judge_panel` to
`null` — the lone judge is then taken from `judge_model_server` +
`judge_responses_create_params_overrides`:

```bash
    ++gdpval_resources_server.resources_servers.gdpval.judge_panel=null
```

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
