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

- `JUDGE_API_KEY` — API key for the judge inference endpoint
- `JUDGE_BASE_URL` — OpenAI-compatible judge endpoint
- `JUDGE_MODEL_NAME` — the single-judge fallback model (used only when the
  [multi-judge panel](#multi-judge-panel) is disabled); defaults to
  `gcp/google/gemini-3.1-pro-preview`
- `HF_TOKEN` — for downloading reference files (avoids HF anonymous rate limits)

By default deliverables are graded by a **panel** of judges (GPT-5.5, Gemini 3.1
Pro Preview, Claude Opus 4.8), one sampled per call. See
[Multi-judge panel](#multi-judge-panel) for how it works, the per-member
environment variables, and how to configure or disable it.

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
220 GDPVal tasks) and assigns **each task a single reference model** from the
stage's adaptively-chosen set. The default is an independent uniform draw;
partial-completion stages use a seeded balanced assignment so every selected
reference gets nearly the same number of planned tasks. It then fits an anchored
Bradley-Terry MLE ELO — pooling each reference's win/loss/tie counts over the
tasks assigned to it — and uses that estimate to pick the references for the
next stage (typically narrowing to fewer references closest to the estimate, and growing `T`, so each
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

### Stage completion

Without a `partial_completion` policy (next section), a non-final stage
completes only when every planned row has a usable judged battle (at least one
win, loss or tie against its assigned reference) and the stage's ELO fit is
finite. Until then the next stage is not planned:

- If a row is still retryable or was never dispatched, the run writes its
  rollouts and aggregate metrics (degraded, with no `comparison/eval_elo`), logs
  that it stopped, and exits 0. Continue with `--resume`, or set
  `++multistage.retry_inprocess=true` to retry `timeout_exceeded` and
  `transient` rows in the same process, up to `NEMO_GYM_MAX_ROLLOUT_ATTEMPTS`
  (default 3) attempts per row.
- If no row is left to retry (the rest are terminal or out of attempts), the run
  writes its rollouts and then fails, without aggregate metrics. That needs a
  `partial_completion` policy or a data fix, not a resume.

Single-stage runs and the final stage are not held open this way: the run
reports whatever final-stage rows succeeded and withholds the headline
`comparison/eval_elo` if any are missing (see [Aggregate metrics](#aggregate-metrics)).

### Accepting partial calibration after timeouts

Calibration stages retry incomplete work by default. To advance after a bounded
number of persisted task timeouts, opt in on that non-final stage and set both
overall and per-reference evidence floors:

```bash
++multistage.stages='[{num_tasks: 45, partial_completion: {min_success_fraction: 0.9, min_per_reference_success_fraction: 0.5, min_successful_rows_per_reference: 1}}, {num_tasks: 220, num_models: 4}]'
```

This example requires at least 90% of all planned calibration rows, at least 50%
of each selected reference's assigned rows, and at least one judged row for every
selected reference. The ELO fit must be finite and include every selected
reference. A row that could still be retried is waived only when its latest
persisted failure class is in `waivable_failure_classes`, which defaults to
`[timeout_exceeded]`; `transient` is the only other allowed class.
`tolerate_unresolved: true` waives any persisted failure class, including
agent-server outages (`agent_request_failed`), so set the coverage floors tight.
It also accepts a missing reference deliverable (`reference_missing`) straight
away. Otherwise GDPVal retries it across `--resume` launches for up to
`NEMO_GYM_MAX_ROLLOUT_ATTEMPTS` (default 3) attempts before it becomes an
omission; `transport_assignment_repair` with `reference_availability_only: true`
avoids it when a stage is planned. Undispatched or drained
rows always keep the stage open. Terminal and max-attempt omissions still count
against every coverage floor.

For a freshly planned policy-enabled stage, task-to-reference assignments are
balanced before dispatch. When the policy is added while resuming an existing
strict run, the recorded assignment remains authoritative so the successful
rows are reused exactly as collected.

Partial completion is deliberately unavailable on the final stage. Its accepted
row set and omitted keys are frozen in the stage journal, so resume cannot absorb
late rows and silently change downstream reference selection. Enabling this
policy while resuming an incomplete stage reuses its successful rollout evidence
instead of invalidating the rollout cache.

### Fresh vs. cached deliverables

- **Fresh** (generate deliverables): set `PERSIST_DELIVERABLES_DIR` to an
  absolute path. The agent persists each deliverable to `persist_deliverables_dir`
  and rejects a relative path at startup, including the config's fallback
  `output/gdpval/deliverables`. A task that recurs in a later stage is judged
  from its cached deliverable instead of re-running the policy.
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
| `stages` | *(required)* | List of `{num_tasks?, num_models?, seed?, partial_completion?}` (or `"[num_tasks]:[num_models]:seed"` strings). `num_tasks` omitted ⇒ full task set; `num_models` omitted ⇒ all references. `partial_completion` is an opt-in non-final calibration policy (see [above](#accepting-partial-calibration-after-timeouts)) with keys `min_success_fraction` (default `1.0`), `min_per_reference_success_fraction` (`1.0`), `min_successful_rows_per_reference` (`1`), `waivable_failure_classes` (`[timeout_exceeded]`) and `tolerate_unresolved` (`false`). |
| `column` | `[occupation]` | Dataset column(s) the task sample is drawn proportionally over. |
| `distribution_path` | *(auto)* | Reuse/write the task-distribution JSON here; built from the dataset when absent. |
| `dataset_path` | *(prepared dataset)* | Dataset the distribution is built from. |
| `nested_tasks` | `false` | `true` makes each stage's task sample a superset of the previous; default samples each stage independently. |
| `seed` | *(none)* | Seed for reproducible task sampling, per-task reference assignment, and reference selection. |
| `reuse_cached_deliverables` | `true` | Judge a task's cached deliverable in later stages instead of re-running the policy. |
| `retry_inprocess` | `false` | Retry `timeout_exceeded` and `transient` rows in the same process, up to `NEMO_GYM_MAX_ROLLOUT_ATTEMPTS`; other retryable rows wait for `--resume`. |
| `transport_assignment_repair` | unset | Reassign tasks between a stage's references before dispatch; see [Transport-aware reference assignment](#transport-aware-reference-assignment). |

### Transport-aware reference assignment

`multistage.transport_assignment_repair` moves tasks between a newly planned
stage's selected references so that fewer candidate/reference pairs exceed the
comparison judge's attachment limits. It keeps each reference's task count and
changes as few tasks as possible, and never moves a task to a reference that has
no `finish_params.json` for it. Turn it on with `enabled: true`:

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | `false` | Run the repair when each stage is planned. |
| `max_file_bytes`, `max_raw_bytes`, `max_wire_bytes`, `max_section_raw_bytes` | the comparison judge limits (`GDPVAL_MAX_FILE_BYTES_FOR_JUDGE`, `GDPVAL_MAX_TOTAL_RAW_ATTACHMENT_BYTES_FOR_JUDGE`, `GDPVAL_MAX_TOTAL_SERIALIZED_REQUEST_BYTES_FOR_JUDGE`, `GDPVAL_MAX_SECTION_RAW_ATTACHMENT_BYTES_FOR_JUDGE`) | A pair is incompatible if a PDF, image, audio or video file (ZIP members included) exceeds `max_file_bytes`, or if either side's total size of such files exceeds `max_section_raw_bytes`. A pair with audio or video must also keep its combined size within `max_raw_bytes`, and that size base64-encoded plus `framing_reserve_bytes` under `max_wire_bytes`. |
| `framing_reserve_bytes` | `4194304` | Request overhead added to the base64 size of an audio/video pair. |
| `reference_availability_only` | `false` | Skip the size checks and only move tasks away from references with no `finish_params.json` for the task. |

- Without `reference_availability_only`, only a task whose candidate
  deliverable already exists under the agent's `persist_deliverables_dir` (which
  must be absolute) when its stage is planned can move. With an empty
  deliverables directory that is only a task an earlier stage already produced,
  so the repair is meant for judge-only runs and reruns over an existing
  deliverables cache.
- `reference_availability_only: true` needs no candidate deliverable, so it also
  works on a fresh run.
- A task it cannot place keeps its reference and is listed in the stage's plan
  record in the journal, under `transport_assignment_repair`:
  `unrepairable_tasks` (no candidate deliverable) or `unroutable_tasks` (no
  compatible reference). If no count-preserving assignment exists, the whole
  stage keeps its draw and `infeasible` gives the reason. An incompatible pair
  left in place fails at dispatch as it would without the repair.
- A plan already in the journal is replayed unchanged. The repair settings are
  part of the resume fingerprint, so changing them starts the run fresh.

### Resuming an interrupted multi-stage run

Use the same output path and original run arguments and pass `--resume` to reuse
the rollout file, failure sidecar, and multi-stage journal. Add the
`partial_completion` policy shown below when recovering a strict calibration
stage that ended with bounded timeouts:

```bash
gym eval run \
    --resume \
    --model-type vllm_model --benchmark gdpval --split benchmark \
    --output results/gdpval_multistage.jsonl \
    ++gdpval_resources_server.resources_servers.gdpval.reward_mode=comparison \
    ++multistage.enabled=true \
    ++multistage.stages='[{num_tasks: 45, partial_completion: {min_success_fraction: 0.9, min_per_reference_success_fraction: 0.5, min_successful_rows_per_reference: 1}}, {num_tasks: 220, num_models: 4}]'
```

Add `RERUN_INCOMPLETE=true` with the same `PERSIST_DELIVERABLES_DIR` when an
unfinished task must resume or reuse its policy deliverable. A task whose
deliverable already **finished** on disk (marked by `finish_params.json`) skips
the policy rollout and is judged from cache; a task that never finished is
re-rolled. `rerun_incomplete` also reuses **cached judgements** keyed by the
task's assigned reference. Each stage's sampled tasks and per-task reference
assignment are recorded in the stage journal (and re-derived deterministically
from the stage seed for older plans), so they replay identically on resume. Use
the same `multistage.seed` so an unplanned stage draws the same tasks.

The journal (`<output stem>_multistage_state.jsonl`, next to the output file)
is tied to a fingerprint of the dataset rows and task distribution; the stage
list, seeds, `nested_tasks`, `column`, `reuse_cached_deliverables` and
`transport_assignment_repair`; the reference ids and ELOs (not the contents of
their deliverable directories); `num_repeats` and the other result-affecting
rollout settings; and every server setting except endpoints, API keys, headers,
`rerun_incomplete` and `concurrency`. A changed fingerprint, or a run without
`--resume`, starts fresh and renames the old rollouts, failure sidecar, journal
and aggregate metrics to `<file>.stale.<ns>`. A stage restart moves the rows it
drops to `<file>.pruned.<ns>`. Neither is deleted; remove them by hand.
`partial_completion`, `retry_inprocess` and `NEMO_GYM_MAX_ROLLOUT_ATTEMPTS` are
not fingerprinted, so they can be added or changed on a resume (except a
policy already accepted as partial; see below). Cached judgements
(`RERUN_INCOMPLETE`) are keyed by the same fingerprint.

After a stage has been accepted as partial, its policy, included evidence, and
omitted keys are frozen. Changing or removing that policy on `--resume` fails
closed; start a fresh output path if you intentionally want a different
calibration decision. See
[Task Re-run Mode](../../responses_api_agents/stirrup_agent/README.md#task-re-run-mode)
for the full semantics.

## Agent settings

The benchmark config sets these Stirrup agent keys on
`gdpval_stirrup_agent.responses_api_agents.stirrup_agent` and leaves
`prompt_estimator_truncate_history_thinking` unset. Override them with
`++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.<key>=...`.

| Key | GDPVal | Stirrup default | Meaning |
|-----|--------|-----------------|---------|
| `context_window_tokens` | `262144` | `262144` | Model context window used to size each call's `max_completion_tokens` and to decide when Stirrup compacts the context. It replaces the window Stirrup derived from the request's `max_output_tokens`, which now only lowers the per-call cap `max_completion_tokens_cap` (default `64000`). |
| `min_completion_tokens` | `8192` | `1024` | Floor on each call's `max_completion_tokens`. GDPVal tasks routinely need multi-thousand-token scripts: a 1,024-token completion cannot hold a useful tool call and can start a sticky `code_exec({})` loop after long reasoning turns. `max_completion_tokens_cap` always applies, and when the tokenizer renders the full prompt the remaining context is also a strict bound. |
| `prompt_estimator_truncate_history_thinking` | unset | unset | Prompt estimator only; never sent to the model. Set it to `true` for checkpoints whose chat template drops reasoning from assistant turns before the last user turn, so the estimate matches. |
| `min_compaction_summary_words` | `50` | `1` | Minimum words in a context-compaction summary, which rejects near-empty summaries that would erase progress on long sessions. After 3 rejected attempts the agent raises an error instead of replacing its history. |
| `truncation_recovery` | `true` | `false` | After a call spends its whole completion budget without a usable tool call, run the next call with thinking disabled and a one-time instruction to act now. The budget is unchanged and the instruction is not recorded in the trajectory. Disabling thinking needs a model server that forwards request `chat_template_kwargs` (`forward_request_chat_template_kwargs: true` on `vllm_model`); otherwise only the instruction is sent. |

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
| `claude-opus-4.8` | `aws/anthropic/bedrock-claude-opus-4-8` | adaptive thinking, high effort (no temperature) |

Each member has its own fixed-model proxy (`gdpval_gpt55_judge_model`,
`gdpval_gemini31_judge_model`, `gdpval_claude48_judge_model`). All three point
at the same OpenAI-compatible endpoint (`JUDGE_BASE_URL`, `JUDGE_API_KEY`), so
one judge endpoint is still enough. `gdpval_judge_model` remains the
single-judge endpoint used when the panel is disabled.

| Variable | Default | Meaning |
|----------|---------|---------|
| `JUDGE_GPT_MODEL`, `JUDGE_GEMINI_MODEL`, `JUDGE_CLAUDE_MODEL` | see the table above | Upstream model id for each member. |
| `JUDGE_GEMINI_API_KEY` | `JUDGE_API_KEY` | Separate key for the Gemini proxy. |
| `GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS` | `2` | Concurrent requests through the Gemini proxy. |
| `JUDGE_SAMPLING_SEED` | unset | Shifts judge sampling (see [Reproducibility](#reproducibility)). |

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

For final/reportable ELO runs, enable `strict_comparison_trials`. The resources
server then rejects a comparison row unless every planned reference matchup
returns exactly `num_comparison_trials` valid votes, with no skipped matchup or
invalid verdict. The option defaults to `false` for backward compatibility; set
it with
`++gdpval_resources_server.resources_servers.gdpval.strict_comparison_trials=true`.
Rejected rows remain retryable with the normal `--resume` workflow instead of
silently contributing fewer votes to the ELO fit.

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
| `media_mode` | `judge_media_mode` | Comparison only: `native_pdf`, `images_and_text`, or `native_pdf_overflow_images` (see below). |
| `max_native_pdf_pages`, `max_native_pdf_documents`, `max_native_pdf_bytes` | unset (no limit) | Comparison only: provider limits on the native PDFs in one request. |
| `max_native_pdf_bytes_per_document` | unset | Comparison only: largest single native PDF. Required, with `max_native_pdf_pages`, for `native_pdf_overflow_images`. |
| `max_image_base64_bytes`, `max_total_image_base64_bytes` | unset | Comparison only: per-image and per-request encoded image limits. |
| `max_video_files` | unset | Comparison only: videos per request. |
| `raster_dpi_tiers` | `[judge_pdf_render_dpi]` | Comparison only: DPIs tried in order for `images_and_text`; the first that fits the limits is used. |
| `max_serialized_request_bytes` | unset | Comparison only: size cap on the serialized request. |

The removed `handles_audio_video` flag is still accepted: it sets both
`handles_audio` and `handles_video` and logs a deprecation warning. Setting it
together with either new flag is an error. Member names (`name`, or `model` when
`name` is unset) must be unique.

### Judge transport per member (comparison mode)

In comparison mode each member receives PDFs and rendered Office files in its own
representation:

- `native_pdf`: PDFs are sent as `application/pdf` data URLs (Claude by default).
- `images_and_text`: each page is rasterized to PNG and the extracted text is
  attached (GPT-5.5 by default), trying `raster_dpi_tiers` in order.
- `native_pdf_overflow_images`: PDFs are sent natively, and documents above the
  member's native caps are rasterized instead (Gemini by default).

Before trials are sampled, a deterministic preflight removes any member that
cannot carry the matchup within its limits; `images_and_text` members are also
removed when the matchup's pages and images exceed `judge_max_images_per_request`.
If every member is removed for every reference matchup, the row fails with the
terminal class `transport_ineligible` (retried on `--resume`).

Rubric mode samples from the same panel but sends every member the server-level
`judge_media_mode` (`native_pdf` by default). It ignores the per-member
`media_mode` and limits and runs no preflight. A member that rejects the payload
yields an invalid judge response (or a terminal failure for size and
context-length errors), and because rubric sampling is seeded per task, a retry
draws the same member.

To grade with a **single judge** instead of the panel, set `judge_panel` to
`null` — the lone judge is then taken from `judge_model_server` +
`judge_responses_create_params_overrides`:

```bash
    ++gdpval_resources_server.resources_servers.gdpval.judge_panel=null
```

## Judge input options

Fields on `gdpval_resources_server.resources_servers.gdpval`:

| Field | Default | Meaning |
|-------|---------|---------|
| `judge_media_mode` | `native_pdf` | Server-level representation, `native_pdf` or `images_and_text`. Used by every rubric judge and by panel members without their own `media_mode`. |
| `judge_pdf_render_dpi` | `144` | Raster DPI for `images_and_text`. |
| `judge_pdf_max_pages` | `50` (the benchmark config sets `1000`) | Pages rasterized per file. In comparison mode `images_and_text` judges use `max(judge_pdf_max_pages, judge_max_images_per_request)`, and `native_pdf_overflow_images` members rasterize overflow documents up to `judge_pdf_max_pages`. |
| `judge_pdf_include_text` | `true` | Attach the extracted text copy next to the page images. |
| `judge_max_images_per_request` | `450` | Comparison only: request-wide image budget. An `images_and_text` member is not used for a matchup that needs more images, and overflow members rasterize within it. |
| `judge_reference_files_recursive` | `false` | Comparison only: include files in subdirectories of the task's `reference_files/`. Submission directories stay shallow. |
| `judge_reference_files_from_eval` | `false` | Comparison only: read `reference_files/` from the eval task directory instead of each reference model's directory. Enable only after validating those prepared inputs. |
| `strict_comparison_trials` | `false` | See [How sampling works](#how-sampling-works). |
| `count_eval_missing_as_loss`, `missing_eval_task_ids` | `false`, `[]` | Comparison, multi-stage `stage_index` 1 only (indices start at 0): a listed task whose eval deliverable has no `finish_params.json` is scored as a loss against every reference instead of failing. Set the same two fields on `gdpval_stirrup_agent` so judge-only runs forward those tasks to the resources server. |
| `judge_handles_audio`, `judge_handles_video` | `false` | Audio/video capability of the single judge when `judge_panel` is `null`. |
| `on_missing_av_judge` | `warn` | See [Audio / video routing](#audio--video-routing). |

Comparison requests are also bounded by these environment variables (defaults in
MiB): `GDPVAL_MAX_FILE_BYTES_FOR_JUDGE` (250),
`GDPVAL_MAX_SECTION_RAW_ATTACHMENT_BYTES_FOR_JUDGE` (96),
`GDPVAL_MAX_SECTION_ENCODED_ATTACHMENT_CHARS_FOR_JUDGE` (128),
`GDPVAL_MAX_TOTAL_RAW_ATTACHMENT_BYTES_FOR_JUDGE` (300),
`GDPVAL_MAX_TOTAL_ENCODED_ATTACHMENT_CHARS_FOR_JUDGE` (400) and
`GDPVAL_MAX_TOTAL_SERIALIZED_REQUEST_BYTES_FOR_JUDGE` (420). An attachment
left out because of a byte limit is replaced by an omission marker the judge can
see; ZIP members skipped by the archive limits are only logged.

## Judge timeouts and retries

| Layer | Setting | Behavior |
|-------|---------|----------|
| Judge proxy, per upstream request | `max_http_attempts` on each `openai_model` judge proxy | `1800` in the benchmark config (Gym's default is 3). Retries 404, 408, 429, 500, 502, 503, 504 and 520 responses 0.5 s apart, about 15 minutes in total. A spent quota or invalid key (by error code) stops at once. |
| GDPVal judge call | `GDPVAL_JUDGE_REQUEST_TIMEOUT_SECONDS` | 300 s in comparison mode, 1800 s in rubric mode. A comparison call that times out is not retried and its reference matchup is dropped. Time queued behind a proxy's `max_concurrent_requests` counts toward it. |
| GDPVal retry | built in | Errors that name throttling or a 502/503/504 are retried: 5 attempts with 5–60 s backoff in comparison mode, and 6 attempts with doubling backoff from 2 s for the text and visual rubric. The structured rubric makes `rubric_structured_formatting_retries` (default 3) attempts per trial, shared with unparseable replies, with backoff doubling from 5 s, and also retries an error whose message contains "timeout". Request-size and context-length errors are terminal. |
| Claude gateway | `timeout` and `request_timeout` in the Claude proxy's `extra_body` | 900 s. Gateway-side keys for the default `JUDGE_BASE_URL`; they are sent in every Claude request body, so remove them if your endpoint rejects unknown fields. |

The proxy window is longer than the comparison timeout, so a comparison call the
resources server has given up on can keep retrying at the proxy until the window
closes, and a persistent 404 (for example an unknown model route) takes the whole
window to surface. Lower the window per proxy if you want faster failures:

```bash
    ++gdpval_gpt55_judge_model.responses_api_models.openai_model.max_http_attempts=60
```

## Local multimodal judge (MiniMax-M3)

By default the judges are frontier models hosted on a third-party inference
API. You can instead judge with a **local, open,
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
   `judge_pdf_include_text` (attach the text copy); see
   [Judge input options](#judge-input-options) for how comparison mode caps
   pages. The text copy is extracted with pdfminer.six, which needs
   `cryptography`; without it the text copy is empty and a warning is logged.
   Because MiniMax-M3 reads
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
> matched. Gym does not currently ship a MiniMax-M3 `local_vllm_model` config for
> this path; use the self-hosted overlay instead.

### Run it

Serve MiniMax-M3 from the public vLLM image `vllm/vllm-openai:minimax-m3` and
expose its OpenAI-compatible `/v1` endpoint. In our testing, a **single-node
TP=4** instance worked, as did independent single-node TP=4 replicas for
data-parallel throughput. Do NOT form one TP=8 group across two nodes unless
you have validated that stack: the cross-node MXFP8 all-reduce path can produce
*garbage logits that still start cleanly*.

```bash
# Verify the self-hosted endpoint from the gym host:
curl -s http://<judge-host>:5000/v1/models
```

Then point gym at that endpoint in repo-root `env.yaml` (the port and
served-model name must match your deployment):

```yaml
MINIMAX_BASE_URL: http://<judge-host>:5000/v1
MINIMAX_MODEL: minimax-m3
MINIMAX_API_KEY: unused
```

```bash
gym eval run \
    --config benchmarks/gdpval/config.yaml \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml \
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

- **Endpoint config**: `MINIMAX_BASE_URL` (must include `/v1`), `MINIMAX_MODEL`
  (defaults to `MiniMaxAI/MiniMax-M3`; match the container's
  `--served-model-name`), and `MINIMAX_API_KEY` (any non-empty string; vLLM
  ignores it).
- **Context budget**: high-DPI page images consume context faster than a
  frontier judge; the overlay caps `judge_pdf_max_pages: 30` per file in rubric
  mode. In comparison mode lower `judge_max_images_per_request` instead (see
  [Judge input options](#judge-input-options)). Lower the DPI or these caps if
  you hit `finish_reason: length`.
- **Idle proxies**: the benchmark's three panel proxies still start (no GPU).
  Drop them with `~gdpval_gpt55_judge_model ~gdpval_gemini31_judge_model
  ~gdpval_claude48_judge_model` for a fully clean run.
- **Video**: MiniMax-M3 reads video natively, so video deliverables are passed as
  native `video_url` media blocks (not filename stubs) via
  `judge_handles_video: true`.
- **Audio**: MiniMax-M3 has **no audio tower** (its `config.json` is an
  image+video VLM with no audio config), so `judge_handles_audio` stays false —
  audio deliverables are dropped with a warning and the rest of the deliverable
  (video/images/text) is still graded. Route audio tasks to an audio-capable
  judge (e.g. Gemini) if you need them scored.

### Gym-spawned Kimi-K2.6 judge

`resources_servers/gdpval/configs/gdpval_kimi_local_judge.yaml` has gym spawn a
Kimi-K2.6 vLLM engine with its vision tower as the single judge
(`gdpval_judge_model_local`), with `judge_panel: null` and
`judge_media_mode: images_and_text`. Its serving settings mirror
`responses_api_models/local_vllm_model/configs/moonshotai/Kimi-K2.6.yaml`
(TP=8, which spans two 4-GPU nodes). Set `KIMI_CHECKPOINT_PATH` to use a local
copy of the weights instead of `moonshotai/Kimi-K2.6`.

```bash
gym eval run --benchmark gdpval --model-type vllm_model \
    --config resources_servers/gdpval/configs/gdpval_kimi_local_judge.yaml \
    --split benchmark --output results/gdpval_kimi_judge.jsonl \
    ++gdpval_resources_server.resources_servers.gdpval.reward_mode=comparison \
    ++gdpval_resources_server.resources_servers.gdpval.reference_models...=...
```

The same page-cap notes apply as for MiniMax-M3. The benchmark's four OpenAI
judge proxies start idle; drop them with `~<name>` if you want a clean run.

## Aggregate metrics

After `gym eval run` returns, the resources server's
`/aggregate_metrics` endpoint emits headline scores in
`results/<output>_metrics.json`:

- Rubric mode: `mean/reward` (pass@1 equivalent), computed over rows with a
  usable judge response. Rows flagged `invalid_judge_response` are left out and
  counted in `rubric/aggregate_rows_total`, `rubric/aggregate_rows_included`,
  `rubric/legacy_invalid_rows_excluded` and
  `rubric/aggregate_rows_included_fraction`.
- Comparison mode: `comparison/wins`, `comparison/losses`, `comparison/ties`,
  `comparison/judged`, `comparison/win_rate` (omitted when no vote was judged),
  `comparison/eval_elo`, `comparison/normalized_elo`
- Multi-stage mode: each stage is reported as `comparison/stage_<k>/eval_elo`
  (plus `.../normalized_elo`, `.../num_tasks`, `.../num_references`,
  `.../judged_tasks`, `.../judged_votes`, `.../imputed_loss_tasks` and
  `.../imputed_loss_votes`), alongside `comparison/num_stages`. When rows record
  the run's final stage (`expected_final_stage_index`), the headline
  `comparison/eval_elo` is that stage's fit and `comparison/headline_stage_index`
  names it. If that stage is missing, incomplete or has no finite fit, the
  headline is withheld and `comparison/final_stage_degraded` is `1`. Rows without
  the field keep the **last** observed stage as the headline.
