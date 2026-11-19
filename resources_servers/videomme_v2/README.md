# VideoMME-v2: single-turn video evaluation

This resources server provides the shared task/scoring primitives and a
stateless Gym verifier. Its single-turn config runs the stock `simple_agent`
with `max_steps: 1` through a `single_agent_turn_legacy` Environment Server and
disables tools in each prepared model request. It performs one policy inference
on the prepared video, followed by deterministic verification, and reports Raw
Acc and the official Nonlinear Score, overall and by L1/L2/L3, under
`Video_MME_v2`.

An agentic setting (a sandboxed agent with video tools) is planned as a
separate change. The verifier already accepts `setting: agentic` cohorts and
reports them under their own namespace, `Video_MME_v2_agentic`.

This is not an EF-certified benchmark: a registered dataset artifact and a full
real-model baseline are separate steps. No checkpoint, credential, benchmark
question or benchmark video is included; the examples are invented questions
about synthetic clips (see [Example data](#example-data)).

## Task boundary

`task.py` defines a frozen `VideoQuestion` with a question ID, source URL and
fixed dataset revision, video ID/source/SHA256, question, two to eight options,
and optional level/group. It has no gold-answer field.

`prepare_single_turn()` emits one user turn containing an already-prepared
`input_video` and question text. It does not choose sampling parameters.
`prepare_agentic()` emits question text plus the bound sandbox video path for
the planned agentic setting; it does not force a tool call or add
benchmark-specific system instructions.

Both builders return the standard `responses_create_params` /
`verifier_metadata` separation. Gold and provenance stay in verifier metadata.
Changing the gold label while keeping the question/media fixed produces exactly
the same model-facing request. Callers must forward **only**
`responses_create_params` to the model, never the whole prepared row.

## Single-turn inputs

Each row's user turn holds one `input_video` part and the question text. The
stock `vllm_model` forwards the part to vLLM as a `video_url`, so a row can
carry either URL form that vLLM accepts:

- a `data:video/mp4;base64,...` URL, which needs no server setup (the committed
  examples use it);
- a `file://` URL, which vLLM loads only when it starts with
  `--allowed-local-media-path` and the file is under that directory on the
  vLLM host's own disk.

vLLM then chooses the frames. It samples 32 frames unless the server sets
another count with `--media-io-kwargs`, for example
`'{"video": {"fps": 2, "num_frames": 256}}'`.

Gym has no preparation command for Video-MME v2 (`video_mcqa/prepare.py`
covers VideoMME-v1 and LVBench). Build each row from the authors' annotation
with `prepare_single_turn()`; `data/example.jsonl` shows the resulting shape:

```python
from resources_servers.video_mcqa.task import VideoQuestion, prepare_single_turn

question = VideoQuestion(
    task_id="videomme_v2:<question id>",
    dataset_source="<annotation source URL>",
    dataset_revision="<40-character annotation revision>",
    video_id="<video id>",
    video_source="<original video URL>",
    video_sha256="<SHA-256 of the original video>",
    question="<question text>",
    options=("<option A>", "<option B>", "<option C>", "<option D>"),
    level=None,  # or the question's level, 1-3
    group_id=None,  # or its four-question group
)
row = prepare_single_turn(question, expected_answer="B", video_url="data:video/mp4;base64,...")
```

This adapter was built for VideoAnyRes preparation, which chooses full-video
temporal coverage and spatial patch targets jointly under the model context
budget (VideoAnyRes sampling, two-frame grouping, the source aspect ratio and
the native timestamp closure) and emits a NeMo-RL native video manifest as the
`input_video`. That preparation is in neither Gym nor NeMo-RL main. Runs with
plain `data:` or `file://` URLs and vLLM's own frame sampling are not the
VideoAnyRes protocol, and their scores are not comparable with VideoAnyRes
runs.

Freeze the preparation policy and source hashes with each run. Neither this
verifier nor `simple_agent` resamples the video. A run must also supply the JSON
`cohort_manifest` referenced by `configs/videomme_v2_single_turn.yaml`, with a
fixed dataset revision, expected task IDs, `setting: single_turn`, answer policy
and repeat count.

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
As in the official scorer, `/L<n>/raw_acc` counts questions by their own level
and `/L<n>/nonlinear_score` counts groups by the group's level, which must be
the level of its last question; one group's questions may have different
levels (the official annotation gives the first three questions of a logic
group their own levels). Without `task_levels`, each question takes its group's
level, so a cohort of the official data lists every question's level there.
Without group metadata, `/nonlinear_available` is false and no numeric
`/nonlinear_score` is emitted; without level metadata, no per-level raw score
is emitted. A few questions from different groups cannot produce a group score.

Duplicate/unexpected IDs and mixed settings, parser policies or revisions are
rejected. Missing/infrastructure-failed rows remain in the question and group
denominators. `/coverage`, `/is_complete`, `/group_coverage` and
`/nonlinear_complete` distinguish provisional results from complete results.
No-answer, format-error and completed execution-failure attempts score zero.

Results record their answer policy and outcome labels. The prompt asks for a
boxed letter, and `boxed_only_v1` scores an unboxed or conflicting answer
as 0. The official script asks for a bare letter and takes the first A-H
letter it finds, so these scores are not comparable to official Video-MME-v2
results.

Rows that Gym adds to count a task as zero (`count_failure_classes_as_zero`,
`count_missing_rollouts_as_zero`) carry no verifier fields. Aggregation skips
them, and the frozen cohort counts their tasks as missing.

## Example data

`data/example.jsonl` holds five invented questions about five synthetic
4-second clips: a moving square, a blue background, one dot, a bar moving down
and a counter that ends on 5. The clips are stored once, in
`resources_servers/video_mcqa/data/example_videos/`. Each row inlines its clip
as an MP4 `data:` URL; `video_source` names the committed file and
`video_sha256` is its SHA-256. `data/example_cohort.json` freezes the five
example task IDs, and `data/example_rollouts.jsonl` holds one real-model
rollout per row. The model and serving setup are in the
[shared README](../video_mcqa/README.md#example-data).

## CPU tests

From the Gym repository root using Python 3.13.14 or later:

```bash
python -m pytest resources_servers/videomme_v2/tests resources_servers/video_mcqa/tests -q
```

The shared parser, task and accuracy tests are in `video_mcqa/tests`.
Fixtures are invented geometric-shape questions with `example.invalid` sources.
Tests use Gym's standard test dependencies and exercise the actual response
schemas, verifier, aggregate metrics and session contract in process. No
benchmark downloads, video decoder, external model endpoint, GPU or training
framework is needed. These tests do not replace multimodal transport, actual
model execution or EF onboarding acceptance.

The [official dataset terms](https://github.com/MME-Benchmarks/Video-MME-v2#-dataset)
restrict use to academic research, prohibit commercial use and restrict redistribution.
Code: Apache-2.0. No source dataset is redistributed by this component. The
synthetic example clips are original to this repository (Apache-2.0).
