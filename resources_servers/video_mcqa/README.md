# Shared video MCQA preparation

This package contains the common task contract, boxed-answer parser, local
annotation adapters and accuracy verifier used by VideoMME-v1 and LVBench.
VideoMME-v2 re-exports the common task/parser APIs and has its own grouped
nonlinear scoring. These adapters do not configure agents, media processing,
sampling, system prompts or compaction. The direct tests of the shared parser,
task contract and accuracy code are in `tests/`.

## Single-turn inputs

A single-turn row's user turn holds one `input_video` part and the question
text. Gym's stock `vllm_model` forwards the part to vLLM as a `video_url`, so a
row can carry either URL form that vLLM accepts:

- a `data:video/mp4;base64,...` URL, which needs no server setup (the committed
  examples use it);
- a `file://` URL, which vLLM loads only when it starts with
  `--allowed-local-media-path` and the file is under that directory on the
  vLLM host's own disk.

vLLM then chooses the frames. It samples 32 frames unless the server sets
another count with `--media-io-kwargs`, for example
`'{"video": {"fps": 2, "num_frames": 256}}'`. These adapters were built for
VideoAnyRes preparation, which emits a NeMo-RL native video manifest as the
`video_url` and is in neither Gym nor NeMo-RL main. Runs with plain `data:` or
`file://` URLs are not the VideoAnyRes protocol, and their scores are not
comparable with VideoAnyRes runs.

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

Single-turn requires a `video_url` for each video, one of the forms in
[single-turn inputs](#single-turn-inputs), and makes one model call with no
tools. No fixed frame count is chosen here. Agentic requires `source` and a safe
relative `sandbox_path`; its row adds the `video_binding` contract for a
sandboxed video agent, which is planned as a separate change. Preparation checks source bytes whenever `source` is
provided, hashing each resolved file once. Single-turn inventories without a
source path must already have been verified by the media-preparation layer.
The agent independently enforces its configured allowed source roots and hash.

Gold labels, source provenance, task categories, durations-as-subsets and
LVBench `time_reference` are never injected into the user prompt. The prompt
asks to watch the video, answer the question and box the selected letter.

## Running the shared verifier directly

`configs/video_mcqa_single_turn.yaml` serves this verifier without a benchmark
subclass: the `benchmark` in its `video_mcqa_cohort_manifest` selects VideoMME-v1
or LVBench scoring and metric names. Like the benchmark configs, it runs the
stock `simple_agent` for one step through a `single_agent_turn_legacy`
Environment Server.

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

Rows that Gym adds to count a task as zero (`count_failure_classes_as_zero`,
`count_missing_rollouts_as_zero`) carry no verifier fields. Aggregation skips
them, and the frozen cohort counts their tasks as missing.

Overall accuracy is question-weighted, not the macro average of category scores.
LVBench multi-label questions count once overall and once in each assigned
capability. Category labels are percent-encoded as metric path components.
Both settings use separate namespaces. Neither v1 nor LVBench reports v2's
nonlinear score.

## Example data

`data/example_videos/` holds five synthetic 4-second, 320x240 H.264 clips drawn
for the examples: a square moving right, a blue background, one dot, a bar
moving down, and a seven-segment counter that ends on 5. They contain no
benchmark or other source media. The VideoMME-v2, VideoMME-v1, LVBench and
`video_mcqa` example rows ask one invented question about each clip; each row
inlines its clip as an MP4 `data:` URL, and `video_source` and `video_sha256`
name the committed file. This package's own examples are the VideoMME-v1 rows.
Each server's `data/example_cohort.json` freezes its five example task IDs.

`data/example_rollouts.jsonl` in each server holds one rollout per row from
`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` (BF16, one GPU), served by
the vLLM nightly image `vllm-openai:nightly-c718e2d7402c` with the model card's
video settings and the model's default sampling:

```bash
vllm serve nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 --max-model-len 131072 \
  --trust-remote-code --video-pruning-rate 0.5 --reasoning-parser nemotron_v3 \
  --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}'
```

They were collected through Gym's stock `vllm_model`, for example:

```bash
gym env start "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/videomme_v1/configs/videomme_v1_single_turn.yaml]" \
  +policy_base_url=http://127.0.0.1:8000/v1 +policy_api_key=EMPTY \
  +policy_model_name=nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
  +videomme_v1_cohort_manifest=$PWD/resources_servers/videomme_v1/data/example_cohort.json
gym eval run --no-serve --agent videomme_v1_single_turn \
  --input resources_servers/videomme_v1/data/example.jsonl \
  --output resources_servers/videomme_v1/data/example_rollouts.jsonl
```

These are evaluation-only adapters. Dataset terms remain those of their
authors; this Apache-2.0 code does not relicense or redistribute benchmark data.
Dataset registration and full EF acceptance are separate from these
loaders/verifiers. The configs are `verified: false`.
