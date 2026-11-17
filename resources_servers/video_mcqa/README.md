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
relative `sandbox_path`; its row adds the `video_binding` that the
[agentic setting](#agentic-setting) puts into each episode's sandbox, and its
prompt names the video at `--sandbox-root/sandbox_path` (default `/testbed`).
Preparation checks source bytes whenever `source` is provided, hashing each
resolved file once. Single-turn inventories without a source path must already
have been verified by the media-preparation layer. At each seed, the agentic
server checks again that the source is under its `video_roots` and that the
copy in the sandbox has the row's SHA-256.

Gold labels, source provenance, task categories, durations-as-subsets and
LVBench `time_reference` are never injected into the user prompt. The prompt
asks to watch the video, answer the question and box the selected letter.

## Running the shared verifier directly

`configs/video_mcqa_single_turn.yaml` serves this verifier without a benchmark
subclass: the `benchmark` in its `video_mcqa_cohort_manifest` selects VideoMME-v1
or LVBench scoring and metric names. Like the benchmark configs, it runs the
stock `simple_agent` for one step through a `single_agent_turn_legacy`
Environment Server.

## Agentic setting

The agentic configs of the three benchmarks (`videomme_v2_agentic.yaml`,
`videomme_v1_agentic.yaml`, `lvbench_agentic.yaml`) run the stock
`opencode_agent` (OpenCode 1.17.11) through a `single_agent_turn_legacy`
Environment Server, as `visual_coding_opencode` does: each episode's seed lends
OpenCode a sandbox that holds the episode's video, and the episode's close stops
it. `sandbox.py` implements the lending for the `videomme_v2` and `video_mcqa`
servers. Like the single-turn configs, these are for evaluation only: the dataset
terms below still apply.

### Sandbox image

`sandbox_image/Dockerfile` builds the image: OpenCode 1.17.11 and its plugin
dependencies (the pins of `visual_coding_opencode`), the static FFmpeg 7.0.2
build of mwader/static-ffmpeg, Python 3.13 with NumPy 2.4.6, Pillow 12.3.0 and
Pydantic 2.13.5, and the video tools in `/opt/video-mcqa`. For the model's own
commands it also has ripgrep (OpenCode's grep and glob tools), fonts for FFmpeg's
`drawtext` filter, and ImageMagick with the fonts its labels default to. Each pin
is checked during the build, and so are `drawtext`, ripgrep and `montage`.

```bash
docker build --platform linux/amd64 -t <registry>/video-mcqa-sandbox:<tag> \
    resources_servers/video_mcqa/sandbox_image
export VIDEO_MCQA_SANDBOX_IMAGE=<registry>/video-mcqa-sandbox:<tag>
```

The sandbox provider must be able to pull the image. Each seed checks that the
image's video tools have this repository's SHA-256 (`TOOL_FILES` in
`sandbox.py`) and refuses the episode otherwise, so rebuild the image after a
change to `sandbox_image/video_tools*`.

### Video tools

`sandbox_image/video_tools.mjs` is an OpenCode plugin with two tools. Each call
runs `python3 -I -m video_tools.cli` in the sandbox with the `tools.json` that
the seed writes from the server's `video_tools` settings.

- `watch_video(video_path, start, end, num_frames, resize_long_edge)` cuts the
  source window `[start, end)` into `num_frames` equal bins and samples each at
  its center. Each sample is the latest source frame whose presentation time is
  not after the sampling time, so a frame the video holds can repeat. Frames
  keep their aspect ratio: scaled to `resize_long_edge`, then each side is
  rounded down to the model's grid cell, `patch_size / downsample_ratio` pixels
  (`video_tools.profile`; the default is 14 / 0.5 = 28). FFmpeg
  resizes by pixel area, and each frame is a JPEG of quality 95 without chroma
  subsampling. The text gives the window, the frame count, size and visual
  tokens, then `This is a video:` and one `Frame i sampled at t seconds:` line
  per frame, in the order of the images. These are the frame rules of Nemotron
  4's agentic video training.
- `view_image(path)` returns an image file unchanged.

Both tools read regular files under the media root and under
`video_tools.scratch_roots`, by default `/tmp/opencode`: OpenCode's bash tool
keeps temporary files there, such as frames or contact sheets the model made.
A relative path starts at the media root, and a symlink that leads out of these
directories is refused. The tool descriptions name only the workspace.

A request outside the limits is a tool error; frames and sizes are never
reduced silently. One frame is at most 2048×2048 pixels, the frames of one call
at most `profile.video_max_pixels` (67,108,864) pixels, an image file at most
32 MiB, and one call at most `timeout_seconds` (1,260 s). Every request sends the
episode's earlier tool images again, so the plugin caps them at
`max_tool_media_bytes` (400 MiB of base64): from the tool result whose images
would pass it on, each result shows a notice in place of its images, the same in
every request.

OpenCode keeps a tool result's text in the tool message and, for an
OpenAI-compatible provider, moves its images into the user message after it
(`Attached media from tool result:`). The frame labels and the images are
therefore not interleaved; the model matches them by order.

### Seeding

At each seed the server:

1. refuses (422) a row whose `video_binding.source` is not a regular file under
   `video_roots` (relative paths start at the repository root);
2. starts the sandbox with `tools.json` and two checksum lists;
3. uploads the video with `nemo_gym.sandbox.upload_path` to
   `sandbox_media_root/sandbox_path` (`/testbed/...`);
4. in one sandbox command, checks the image's video tools (422 if they differ)
   and the video's SHA-256 (422 if it is not the row's), makes the video
   read-only and runs the tools' own check (422 if FFmpeg or the configuration
   is missing).

Any other failure, including a command whose result has an `error_type` or the
`-1` sentinel, stops the sandbox and raises, and the Environment Server runs the
episode again. OpenCode works in `sandbox_workdir` (`/testbed/work`), and with
`sandbox_network_access: model_only` the sandbox reaches only the policy model
server. Grading never reads the sandbox.

The upload sends each video in one request. Gym's OpenSandbox provider reads
the whole file and sends it in one write, which failed for files of about
1.6–2.0 GB in our runs (for example, 6 LVBench videos, 90 of its 1,549
questions, are larger than 1.5 GiB). Such videos need OpenSandbox uploads
streamed from disk
([NVIDIA-NeMo/Gym#5904](https://github.com/NVIDIA-NeMo/Gym/pull/5904)); until then
their seeds fail, and the cohort counts those questions as missing.

### Model settings

OpenCode sessions send no `chat_template_kwargs`. For a model whose chat
template takes `enable_thinking`, set it on the policy model
server, which applies it to every call:
`++policy_model.responses_api_models.vllm_model.chat_template_kwargs.enable_thinking=true`.
Sessions run with a fixed title and without compaction, so every call is a call
of OpenCode's build agent.

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
In the agentic setting, a run that the Environment Server sends to verify as
`failed` (a model 400, 413 or 422, an out-of-memory kill, or a run without an
outcome) or `incomplete` (a time, output or context limit) is an execution
failure, read from the response's own status, so a replayed row scores the
same; tool calls are allowed. In the single-turn setting, a failed model call is
an infrastructure error, masked and inconclusive, and a tool call is an
execution failure. Missing/inconclusive rows
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
The VideoMME-v2, VideoMME-v1 and LVBench servers also have agentic examples:
`data/example_agentic.jsonl` asks the same questions, with each clip bound by
`video_binding` and named at `/testbed/videos/<clip>`, and
`data/example_agentic_cohort.json` freezes them.

`data/example_rollouts.jsonl` in each server holds one rollout per row from
`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` (BF16, one GPU), served by
vLLM `0.30.1rc1.dev143+g29468dde8` with the model card's video settings and the
model's default sampling (temperature 0.6, top-p 0.95, at most 16,384 output
tokens):

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

In each server the model answers the first four questions correctly. On the
counter it reads the final 5 as 9, or runs out of output tokens, so the rows also
show the `incorrect` and `no_answer` outcomes.

`data/example_agentic_rollouts.jsonl` in the VideoMME-v2, VideoMME-v1 and
LVBench servers holds one agentic rollout per row from the same model on one
GPU, served by vLLM `0.32.0rc1.dev190+g3f1bcfd82` with the model card's flags
plus OpenCode's tool calls and room for the frames of a `watch_video` call. The
model's default sampling applies (temperature 0.6, top-p 0.95; OpenCode sends
none for this model), with at most 16,384 output tokens per call, and the
sandboxes are OpenSandbox sandboxes from the image of `sandbox_image/Dockerfile`:

```bash
vllm serve nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 --max-model-len 131072 \
  --trust-remote-code --video-pruning-rate 0.5 --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --limit-mm-per-prompt '{"image": 256, "video": 0}'
```

```bash
export VIDEO_MCQA_SANDBOX_IMAGE=<the sandbox image, pinned by digest>  # and OPENSANDBOX_DOMAIN, OPENSANDBOX_API_KEY
gym env start "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/videomme_v2/configs/videomme_v2_agentic.yaml]" \
  +policy_base_url=http://127.0.0.1:8000/v1 +policy_api_key=EMPTY \
  +policy_model_name=nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
  +videomme_v2_agentic_cohort_manifest=$PWD/resources_servers/videomme_v2/data/example_agentic_cohort.json \
  "+videomme_v2_agentic_video_roots=[$PWD/resources_servers/video_mcqa/data/example_videos]" \
  ++videomme_v2_agentic_opencode.responses_api_agents.opencode_agent.context_window=131072 \
  ++videomme_v2_agentic_opencode.responses_api_agents.opencode_agent.max_output_tokens=16384
gym eval run --no-serve --agent videomme_v2_agentic_opencode \
  --input resources_servers/videomme_v2/data/example_agentic.jsonl \
  --output resources_servers/videomme_v2/data/example_agentic_rollouts.jsonl
```

All 15 episodes complete with a `\boxed{}` answer. The model answers the five
VideoMME-v2 questions correctly, each after two or three `watch_video` calls,
and two of the five VideoMME-v1 and four of the five LVBench questions. In four
of those ten episodes it answers without calling a tool.

These are evaluation-only adapters. Dataset terms remain those of their
authors; this Apache-2.0 code does not relicense or redistribute benchmark data.
Dataset registration and full EF acceptance are separate from these
loaders/verifiers. The configs are `verified: false`.
