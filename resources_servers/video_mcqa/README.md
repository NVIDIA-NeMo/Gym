# Shared video MCQA preparation

This package contains the common task contract, boxed-answer parser, local
annotation adapters and accuracy verifier used by VideoMME-v1 and LVBench.
VideoMME-v2 re-exports the common task/parser APIs and has its own grouped
nonlinear scoring. These adapters do not configure agents, media processing,
sampling, system prompts or compaction.

## Prepare local evaluation data

Use the benchmark authors' annotations and an explicit media inventory. This
package never downloads questions/videos, reconstructs filenames, uses temporal
answer hints, or silently drops unavailable media.

```bash
python -m resources_servers.video_mcqa.prepare \
  --benchmark videomme_v1 \
  --annotations /data/annotations.parquet \
  --dataset-revision <40-character-dataset-commit> \
  --media-manifest /data/media.json \
  --setting single_turn \
  --output-dir /data/prepared/videomme-v1-single
```

Choose `lvbench` and `agentic` for the other benchmark and setting. JSON, JSONL
and Parquet are supported; only Parquet input needs the optional `pyarrow`
package. The revision must identify the annotation snapshot, not a model or
evaluation-code commit. Preparation produces `tasks.jsonl`, `cohort.json` and a
final `receipt.json` with input/output SHA256s. Existing output directories are
not overwritten. Keep these evaluation artifacts outside source control.
Repeated option text in source annotations is preserved with
`annotation_warnings: ["duplicate_option_text"]` outside model inputs and counted
in the receipt. It is not repaired, relabelled or excluded from the benchmark.
The shared question contract otherwise rejects duplicate options
unless the annotation loader explicitly opts in for that source row.

The media manifest maps **original video IDs** to bindings. Invented example:

```json
{
  "example-video": {
    "source_url": "https://example.invalid/original-video",
    "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "source": "/data/videos/example.mp4",
    "sandbox_path": "video/example.mp4",
    "video_url": "<descriptor-from-the-VideoAnyRes-preprocessor>"
  }
}
```

Single-turn requires `video_url` from the VideoAnyRes media-preparation layer,
and makes one model call with no tools. No fixed frame count is chosen here.
Agentic requires `source` and a safe relative `sandbox_path`; its row adds the
`video_binding` contract so OpenCode receives the original video
inside its sandbox. Preparation checks source bytes whenever `source` is
provided, hashing each resolved file once. Single-turn inventories without a
source path must already have been verified by the media-preparation layer.
The agent independently enforces its configured allowed source roots and hash.

Gold labels, source provenance, task categories, durations-as-subsets and
LVBench `time_reference` are never injected into the user prompt. The prompt
asks to watch the video, answer the question and box the selected letter.

## Verification and metrics

Each run declares a complete cohort before inference: benchmark, official source
URL, pinned dataset revision, setting, task IDs, reporting categories and repeat
count. Verification rejects cross-benchmark, cross-setting and cross-revision
results. Duplicate tasks or unexpected repeat indices cannot be aggregated.

Default answer policy is `boxed_only_v1`. `boxed_or_letter_v1` is an explicit
alternative for saved predictions that contain only a bare uppercase letter.
No LLM judge, answer repair, rationale scoring or substring guessing is used.

`raw_acc` is a percentage; `mean/<benchmark>` is the matching 0-1 Gym alias.
Missing answers, ambiguous answers and execution failures receive zero credit.
Infrastructure errors are masked and inconclusive. Missing/inconclusive rows
remain in the frozen denominator and set `is_complete=false`; these partial
numbers are not final benchmark results. On a complete cohort, accuracy agrees
with the upstream exact-letter scorers. The upstream LVBench scorer skips
absent predictions; this implementation deliberately does not call a smaller,
partially answered subset a completed benchmark.

Overall accuracy is question-weighted, not the macro average of category scores.
LVBench multi-label questions count once overall and once in each assigned
capability. Category labels are percent-encoded as metric path components.
Both settings use separate namespaces. Neither v1 nor LVBench reports v2's
nonlinear score.

These are evaluation-only adapters. Dataset terms remain those of their
authors; this Apache-2.0 code does not relicense or redistribute benchmark data.
Runtime capsule packaging, dataset registration and full EF acceptance are
separate from these loaders/verifiers. The configs are `verified: false`.
