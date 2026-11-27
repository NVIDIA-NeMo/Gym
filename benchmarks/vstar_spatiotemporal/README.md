# V-STaR (spatio-temporal reasoning)

Video spatio-temporal reasoning on [V-STaR](https://huggingface.co/datasets/V-STaR-Bench/V-STaR). For each
annotated event the model is asked *what* happens, *when* it happens and *where* the object is, and the three
answers are combined into one score. This is the video benchmark V-STaR, not the image
visual-search benchmark V\*.

- Integration profile: `custom-gym-verifier`
- Scorer: [`resources_servers/vstar_spatiotemporal`](../../resources_servers/vstar_spatiotemporal/README.md)
- Harness: `simple_agent`, single turn, no tools, through the `legacy_agent` relay (see [Limitations](#limitations))
- Judge: an LLM rates the "what" answers (see [Run conditions](#run-conditions))

## Provenance

| | |
| --- | --- |
| Source | V-STaR (`V-STaR-Bench/V-STaR`), `V_STaR_test.json` |
| Tasks | 10,470: 2,094 items x 5 sub-tasks, over 732 videos in 9 domains |
| License | CC BY-NC-SA 4.0 (dataset); the benchmark code is Apache-2.0 |
| Videos | Shipped with the dataset; not redistributed here |

No items are filtered. `prepare.py` refuses to run if the annotation file or any annotated video is missing. The
dataset is not downloaded automatically.

Code and prompts adapted from other projects (the same record is in the header of each file):

| Here | Source | License |
| --- | --- | --- |
| `resources_servers/vstar_spatiotemporal/app.py`: the judge prompts (verbatim), `temporal_iou`, `box_iou`, `spatial_scores`, `parse_judge_score` | [`eval.py`](https://github.com/V-STaR-Bench/V-STaR/blob/1894682b15be0a0c0c5ab24be6e8252743cd6c2f/eval.py) of V-STaR at commit `1894682b` | the repository declares none |
| `app.py`: the number pattern in `parse_span`, `balance_brackets`; `prepare.py`: the question wording | [`inference_demo.py`](https://github.com/V-STaR-Bench/V-STaR/blob/1894682b15be0a0c0c5ab24be6e8252743cd6c2f/inference_demo.py) of V-STaR at commit `1894682b` | the repository declares none |
| `prepare.py`: `resized_size` and its constants | `smart_resize` in [`vision_process.py`](https://github.com/QwenLM/Qwen2.5-VL/blob/a30e36facd0a5131d9ed59e93210c7ac5de75adb/qwen-vl-utils/src/qwen_vl_utils/vision_process.py) of qwen-vl-utils at commit `a30e36fa` | Apache-2.0 |

## Task contract

Each item is asked as five independent single-turn tasks (`verifier_metadata.sub_task`):

| Sub-task | Question | Given | Expected answer |
| --- | --- | --- | --- |
| `vqa` | what | — | one or two sentences |
| `temporal` | when (chain 1) | — | `From <t>start</t>s to <t>end</t>s` |
| `spatial` | where (chain 1) | the annotated time span | `{"tracks":[{"id":1,"t":T,"box":[x1,y1,x2,y2]}]}` |
| `spatial_2` | where (chain 2) | how many seconds the event lasts | `{"from":S,"to":E,"tracks":[...]}` |
| `temporal_2` | when (chain 2) | the annotated boxes, without their times | `From <t>start</t>s to <t>end</t>s` |

The two chains are V-STaR's two question orders (what-when-where and what-where-when). The tasks do not see each
other's answers.

Each task is a system turn and one user turn, built by `prepare.py` and stored in the row
(`prompt_source: prepared`). The system turn asks for reasoning in `<think>` and the answer in `<answer>` tags, and
for locations on a 0-1000 grid. The user turn holds the video's frames, each preceded by `Frame i at <t>s: `, and
the question ending in a clause that fixes the answer format. Boxes are integers on a 0-1000 grid (per axis,
independent of pixel size).

The video itself is never sent. Frames are decoded with OpenCV one second apart from the first frame; a video longer
than 16 seconds gets 16 frames evenly spaced over its whole length instead. Each frame is resized so both sides are
multiples of 28 pixels (close to the original size) and stored as JPEG at quality 95. A task has at most 16 images.

Ground truth is in `verifier_metadata` only, apart from what a sub-task hands to the model by design (the table's
"Given" column).

Each row has a top-level `task_id`, `<vid>/<item index>/<sub-task>` (for example `abc123/12/spatial`). V-STaR items
have no ID of their own, so the item index is the item's position in `V_STaR_test.json`, the same number as
`verifier_metadata.item_index`. The ID is stable for as long as that file is unchanged, and it is not part of the
prompt.

### How frames reach the model

The five tasks of an item, and all items of a video, share the same frames. Each frame is referenced by a `file://`
URL (73 MB of JSONL; the frames take 1.1 GB). This needs a model server that reads local files, for example vLLM
started with `--allowed-local-media-path <frames dir>`. The `opencode_agent` and `pi_agent` harnesses take prompt
images only as data URLs, so they refuse these rows with a 422. With `VSTAR_EMBED_FRAMES=1` they accept the rows but
send the frames apart from their `Frame i at <t>s:` captions.

| Variable | Effect |
| --- | --- |
| `VSTAR_ROOT` | Required. Directory holding `V_STaR_test.json` and `videos/` |
| `VSTAR_FRAME_DIR` | Where extracted frames are written (default `benchmarks/vstar_spatiotemporal/data/frames`) |
| `VSTAR_FRAME_URL_ROOT` | Path at which the model server sees the frame directory, when it differs |
| `VSTAR_EMBED_FRAMES=1` | Embed frames as data URLs, for endpoints that cannot read local files |

Frames already extracted are reused, so `prepare.py` can be re-run to change only the URLs. Each video's frames
record the sampling rule they were extracted under (`sampling_version` in `frames.json`); frames from another rule,
or from before the rule was recorded, are extracted again. The JSONL is written to a temporary file and moved into
place only after every video has succeeded, so a failed preparation leaves no partial file and does not touch an
earlier complete one.

## Scoring

The reward of a task is 1 or 0 for `vqa` (judge rating of 2 or 3 out of 3), the temporal IoU for the "when" tasks,
and the mean box IoU over the annotated seconds for the "where" tasks. The benchmark metrics fold the five tasks of
each item back together, as the V-STaR evaluation does; see the
[scorer README](../../resources_servers/vstar_spatiotemporal/README.md).

| Metric | Meaning |
| --- | --- |
| `mAM` | Mean over the two chains of the arithmetic mean of VQA accuracy, temporal mIoU and spatial mIoU. Headline metric |
| `mLGM` | Mean over the two chains of the modified logarithmic geometric mean of the same three |
| `VQA_acc` | Fraction of items whose answer the judge rates 2 or 3 |
| `temporal_mIoU`, `temporal_2_mIoU` | Mean temporal IoU, chain 1 and chain 2 |
| `spatial_mIoU`, `spatial_2_mIoU` | Mean box IoU, chain 1 and chain 2 |
| `temporal*_R1@t`, `spatial*_mAP@t` | Fraction of spans, and mean fraction of boxes, at or above IoU `t` |
| `length/<Short\|Medium\|Long>/...`, `domain/<name>/...` | The main metrics by video length (under 1 min, 1-3 min, longer) and by domain |
| `parsed_rate_*` | Fraction of items whose reply held something to score, per sub-task and on average |

All values are fractions between 0 and 1, except `mLGM`, which is unbounded above. The mean reward over tasks is
not a V-STaR metric.

## Run conditions

The frames-as-images protocol this benchmark reproduces was run with temperature 0, top-p 1, repetition penalty 1.05,
reasoning enabled and a 32,768-token output budget; pass these through `responses_create_params` when comparing
against such results.

The V-STaR evaluation rates the "what" answers with Qwen2.5-72B-Instruct, sampling with the model's default settings
(temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.05): its `eval.py` calls `generate` with no sampling
arguments. This benchmark judges at temperature 0 instead, with the same prompts and a 512-token limit, so that a
rating does not change from one run to the next. The judge receives the text inside the reply's `<answer>` tags, or
the whole reply when it has none. The benchmark config declares a `judge_model` server for this; point it at an
OpenAI-compatible endpoint serving that model:

| Variable | Effect |
| --- | --- |
| `JUDGE_BASE_URL` | Judge endpoint (default `http://localhost:8001/v1`) |
| `JUDGE_MODEL` | Model name at that endpoint (default `Qwen/Qwen2.5-72B-Instruct`) |
| `JUDGE_API_KEY` | API key, if the endpoint needs one |

A different judge gives scores that are not comparable; record the judge with any reported score.

When a judge call fails (after Gym's retries), Gym masks that VQA rollout. Its item is then left out of every
V-STaR metric instead of counting as a wrong answer, and Gym reports the number of masked rollouts beside the
scores. A judge reply that is not a single integer from 0 to 3 is a completed call: it rates the answer as wrong.

## Prepare and run

```bash
export VSTAR_ROOT=/path/to/V-STaR   # holds V_STaR_test.json and videos/
gym eval prepare --benchmark vstar_spatiotemporal
gym env validate vstar_spatiotemporal
gym eval run --benchmark vstar_spatiotemporal --model-type vllm_model --split benchmark --limit 10 --num-repeats 1 \
    --output results/vstar_spatiotemporal_smoke.jsonl
```

`prepare.py` decodes the videos with OpenCV. `gym eval prepare` installs OpenCV into its own locked environment
(`pyproject.toml` and `uv.lock` in this directory pin `opencv-python-headless` 5.0.0.93); Gym's own environment
stays without it.

Decoding the 732 videos takes about 20 minutes.

## Limitations

- An LLM judge decides one third of the headline metric, so scores are comparable only between runs that use the
  same judge.
- A limited run (`--limit`) that does not cover all five tasks of an item scores the missing ones as 0. The same
  holds for a "when" or "where" rollout that failed. An item with no scored VQA rollout is left out instead.
- Decoded pixels can differ slightly between OpenCV builds, so the preparation environment pins one. Frames
  extracted with another build (for example one installed by hand) are not replaced until they are extracted again.
- The example rows of the resources server are synthetic (a moving rectangle) and contain no V-STaR footage.
- Scores are not comparable with published V-STaR results. V-STaR's reference inference (`inference_demo.py`)
  samples one frame per second. Here a video longer than 16 seconds gets 16 frames. From 60 seconds of video on,
  at most about one in four whole seconds of the video has a frame within half a second. The judge also runs at
  temperature 0 here and is sampled there (see [Run conditions](#run-conditions)).
- The benchmark runs `simple_agent` through the `legacy_agent` relay, which Gym keeps for Agent Servers that have
  not moved to the Environment Server protocols. On it, a session harness such as `opencode_agent` cannot run the
  benchmark. Moving to `single_agent_turn_legacy` is a configuration-only change (no tool grant, because the
  verifier has no tools), but the reported scores were measured on the relay, so the switch waits for a run that
  shows the rewards match. Those harnesses also need the frames embedded (see
  [How frames reach the model](#how-frames-reach-the-model)).
