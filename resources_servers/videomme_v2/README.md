# VideoMME-v2: VideoAnyRes single-turn and agentic video evaluation

This resources server provides the shared task/scoring primitives and a
stateless Gym verifier. Its single-turn config uses the stock `simple_agent`
with `max_steps: 1` and disables tools in each prepared model request.
Single-turn means the **VideoAnyRes** adaptive video input, not a fixed
frame count or a replay of agentic `watch_video` tool media. It performs one
policy inference on the prepared video, followed by deterministic verification.

The agentic config instead selects `opencode_video_agent`, a narrow subclass of
stock `opencode_sandboxed_agent`, with the optional `vllm_video_model` converter.
It stages the raw video and a pinned self-contained visual-tool runtime inside
the sandbox; stock OpenCode chooses tools/windows and keeps its native prompts
and compaction. No benchmark-specific system prompt or forced tool call is added.
This path has its own cohort (`setting: agentic`), result directory and metric
namespace, `Video_MME_v2_agentic`. Single-turn uses `Video_MME_v2`.
Both report Raw Acc and official Nonlinear Score, overall and by L1/L2/L3.

This is not an EF-certified benchmark: the portable VideoAnyRes preparation
dependency, a registered dataset artifact and real-model acceptance are separate
gates. No checkpoint, credential, benchmark question or media is included.

## Task boundary

`task.py` defines a frozen `VideoQuestion` with a question ID, source URL and
fixed dataset revision, video ID/source/SHA256, question, two to eight options,
and optional level/group. It has no gold-answer field.

`prepare_single_turn()` emits one user turn containing an already-prepared
`input_video` and question text. It does not choose sampling parameters.
`prepare_agentic()` emits question text plus the bound sandbox video path; it
does not force a tool call or add benchmark-specific system instructions.

Both builders return the standard `responses_create_params` /
`verifier_metadata` separation. Gold and provenance stay in verifier metadata.
Changing the gold label while keeping the question/media fixed produces exactly
the same model-facing request. Callers must forward **only**
`responses_create_params` to the model, never the whole prepared row.

## Single-turn media boundary

The preprocessing layer chooses full-video temporal coverage and spatial patch
targets jointly under the model context budget. It uses VideoAnyRes sampling,
two-frame grouping, the source aspect ratio and the native timestamp closure.
The count includes visual placeholders, timestamps, wrappers and the
non-video prompt, reserving space for the configured output. A fixed 64-frame
or 576px policy is not an interchangeable implementation of this setting.

Freeze the preparation policy and source hashes with each run. The resulting
row is passed to stock Gym execution; neither this verifier nor `simple_agent`
resamples the video. A run must also supply the JSON `cohort_manifest` referenced
by `configs/videomme_v2_single_turn.yaml`, with a fixed dataset revision, expected
task IDs, `setting: single_turn`, answer policy and repeat count.

The prepared `input_video` is a NeMo-RL native video manifest, so the media it
names (a frame cache or the source video) must be visible to serving, and
preparation uses NeMo-RL's preprocessing modules; neither is bundled into Gym.
The fixed-media NPZ tool path and OpenCode are outside the single-turn path.
Agentic uses those visual tools, not VideoAnyRes resampling on each generation
request.

## Scoring boundary

`grade_answer()` consumes final assistant text, private `VerifierMetadata`,
and the run setting. Do not pass an entire transcript, tool result or reasoning
message as final text. Explicit thinking blocks are removed before parsing;
unclosed reasoning does not count as a final answer.

- `boxed_only_v1` (default): one unambiguous available uppercase letter in
  `\boxed{...}`, including `\text{...}` / `\mathrm{...}` wrappers. Repeated
  identical boxed letters are unambiguous; conflicting letters are rejected.
- `boxed_or_letter_v1`: additionally accepts final visible text consisting of
  exactly one uppercase option letter. It never guesses a letter from prose.

The outcomes distinguish correct, incorrect, no answer, format error, ambiguous
answer, completed execution failure and infrastructure failure. The caller owns
failure classification. Infrastructure failures return `reward=None` and
`evaluation_completed=False`; completed execution failures return zero, even
when partial output contains a correct letter.

Gym requires a numeric reward. `app.py` represents an infrastructure failure as
`reward: 0`, `mask_sample: true`, `evaluation_completed: false`; it does not
reclassify that failure as a measured wrong answer. Fixed-cohort metrics keep
the missing measurement in their declared denominator.

## Metrics

The following names use `Video_MME_v2` for single-turn and
`Video_MME_v2_agentic` for agentic. Scores are percentages (0–100).

| Metric suffix | Definition |
| --- | --- |
| `/raw_acc` | Correct questions / frozen expected questions × 100 |
| `/nonlinear_score` | Mean official score across frozen four-question groups |
| `/L1/raw_acc`, `/L2/raw_acc`, `/L3/raw_acc` | Per-level question accuracy |
| `/L1/nonlinear_score`, `/L2/nonlinear_score`, `/L3/nonlinear_score` | Per-level mean group score |

Gym's standard `mean/Video_MME_v2` and `mean/Video_MME_v2_agentic` keys are
raw accuracy on a 0–1 scale, not nonlinear scores.

Relevance groups score 0, 6.25, 25, 56.25 or 100 for zero through four correct
answers. Logic groups use the source `group_structure` and ordered question
IDs to apply dependency-aware scoring. The implementation matches the
[official scoring functions](https://github.com/MME-Benchmarks/Video-MME-v2/blob/28fc3bcfbd4d162594c8e4bad866b694d9b035aa/evaluation/test_video_mme_v2.py).
Answers from different repeats are never combined into a single group.

Use cohort `schema_version: 2` to require complete group metadata before a run.
Each `groups` entry contains `group_id`, `level` (1–3), `group_type` (`relevance`
or `logic`), and exactly four `task_ids` in source question order. Logic groups
also require `group_structure`: `"[1, 2, 3, 4]"`, `"[1, [2, 3], 4]"` or
`"[[1, 2], 3, 4]"`. Groups must partition the frozen cohort without overlap.
Group IDs are scoped by level. Group metadata stays outside model input.

Schema 1 (the default) also accepts raw-only cohorts. An optional `task_levels`
mapping must cover all frozen task IDs and enables raw per-level breakdowns.
Without group metadata, `/nonlinear_available` is false and no numeric
`/nonlinear_score` is emitted; without level metadata, no per-level raw score
is emitted. A few questions from different groups cannot produce a group score.

Duplicate/unexpected IDs and mixed settings, parser policies or revisions are
rejected. Missing/infrastructure-failed rows remain in the question and group
denominators. `/coverage`, `/is_complete`, `/group_coverage` and
`/nonlinear_complete` distinguish provisional results from complete results.
No-answer, format-error and completed execution-failure attempts score zero.

Results record their answer policy and outcome labels; they need not match
other answer parsers on every edge case.

## CPU tests

From the Gym repository root using Python 3.13.14 or later:

```bash
python -m pytest resources_servers/videomme_v2/tests -q
```

Fixtures are invented geometric-shape questions with `example.invalid` sources.
Tests use Gym's standard test dependencies and exercise the actual response
schemas, verifier, aggregate metrics and session contract in process. No
benchmark downloads, video decoder, external model endpoint, GPU or training
framework is needed. These tests do not replace multimodal transport, actual
model execution or EF onboarding acceptance.

Code: Apache-2.0. No source dataset is redistributed by this component.
