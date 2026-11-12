# Molmo2-VideoTrackEval

Text-referred video point tracking on [allenai/Molmo2-VideoTrackEval](https://huggingface.co/datasets/allenai/Molmo2-VideoTrackEval).
The model is given a description such as "the player in the red shirt" and the frames of a clip, and reports a point
on every object matching the description in every frame where it is visible.

- Integration profile: `custom-gym-verifier`
- Scorer: [`resources_servers/molmo2_videotrack`](../../resources_servers/molmo2_videotrack/README.md)
- Harness: `simple_agent`, single turn, no tools

## Provenance

| | |
| --- | --- |
| Source | Molmo2-VideoTrackEval (Allen Institute for AI), `test` split, all five categories |
| Tasks | 3,147 expressions: animal 332, dance 905, misc 80, person 992, sports 838 |
| Annotation license | ODC-BY-1.0; the benchmark code is Apache-2.0 |
| Videos | Not shipped with the dataset. They come from APTv2, DanceTrack, SportsMOT, PersonPath22 and SA-V, each under its own terms, and are not redistributed here |

No rows are filtered. `prepare.py` refuses to run if an annotation file or any required frame is missing.

Frames must be extracted once with the dataset's own `scripts/preprocess_videos.py`, which writes
`videos/<category>/<clip>/<index>.jpg` with `index` aligned to the annotation arrays.

## Task contract

One task is one annotation row: one expression over one clip, which may match several objects. The single user turn
is built by `prepare.py` and stored in the row (`prompt_source: prepared`):

1. `The following N frames are taken 1 second apart, in temporal order, from a single video.` followed by the
   expression.
2. `Frame k:` followed by a frame, for `k = 1..N`.
3. An instruction to return `<answer>{"tracks":[{"id":1,"frame":1,"xy":[x,y]}]}</answer>` with integer coordinates
   on a 0-1000 grid, one entry per visible object per frame, and no entry where an object is not visible.

Frames are the clip's frames at local indices 0, fps, 2·fps, ... (fps rounded to an integer), so they are 1 second
apart starting at the first frame, whether or not the object is visible in them. A task has 3 to 30 frames. Frames
are sent unmodified at their original resolution. The 0-1000 grid is per axis and independent of pixel size.

Ground-truth points and segmentation masks are in `verifier_metadata` only.

### How frames reach the model

Many expressions share a clip, so embedding frames in every row would produce about 7 GB of JSONL. By default each
frame is referenced by a `file://` URL instead (44 MB of JSONL). This needs a model server that reads local files,
for example vLLM started with `--allowed-local-media-path <frames dir>`.

| Variable | Effect |
| --- | --- |
| `MOLMO2_VIDEOTRACK_ROOT` | Required. Dataset root holding `data/` and `videos/` |
| `MOLMO2_VIDEOTRACK_FRAME_URL_ROOT` | Path at which the model server sees `<root>/videos`, when it differs |
| `MOLMO2_VIDEOTRACK_EMBED_FRAMES=1` | Embed frames as data URLs, for endpoints that cannot read local files |

## Scoring

Per frame, predicted points are matched one-to-one to the visible objects and a matched point is correct when it
lies inside that object's mask; this gives a precision, recall and F1 per frame. Each task's score is the average
over its frames. See the [scorer README](../../resources_servers/molmo2_videotrack/README.md) for details.

| Metric | Meaning |
| --- | --- |
| `f1` | Mean task F1 over all expressions (headline; equals `mean/reward`) |
| `<category>/f1` | Mean task F1 within `animal`, `dance`, `misc`, `person` or `sports` |
| `precision`, `recall` | Mean task precision and recall |
| `parse_fail_rate` | Share of tasks whose answer yielded no point at all |

All are in [0, 1], higher is better. `f1` weights every expression equally, so it is the average of the category
scores weighted by their task counts, not their plain average.

This follows the protocol described in the Molmo2 paper (F1 of the points at 1 fps). It has not been checked
against Ai2's evaluation code, so compare with published Molmo2 numbers with care.

## Run conditions

Scoring is deterministic and uses no judge. The numbers this benchmark was set up to reproduce were obtained with
temperature 0.6, top-p 0.95, a 16,384-token output budget, one repeat, and reasoning enabled; pass these through
`responses_create_params` when comparing against them. Sampling is stochastic, so expect run-to-run variation.

## Prepare and run

```bash
export MOLMO2_VIDEOTRACK_ROOT=/path/to/Molmo2-VideoTrackEval
gym eval prepare --benchmark molmo2_videotrack
gym env validate molmo2_videotrack
gym eval run --benchmark molmo2_videotrack --model-type vllm_model --split benchmark --limit 5 --num-repeats 1 \
    --output results/molmo2_videotrack_smoke.jsonl
```

## Limitations

- With the default `file://` frames, the prepared JSONL is tied to the frame path it was prepared for and to a
  model server that can read that path.
- Prompts carry up to 30 images, some of them 3840x2160.
- Object ids in the answer are not scored: an answer that swaps two objects' ids between frames loses nothing.
