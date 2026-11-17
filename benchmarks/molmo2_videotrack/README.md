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

No rows are filtered. `prepare.py` refuses to run if an annotation file or any required frame is missing. It writes
the benchmark file only once every row has converted, so a failed run leaves no partial file and does not touch one
from an earlier complete run.

### Extracting the frames

The dataset has no frame-extraction script, so this folder has one. Each annotation row covers frames
`[start_frame, end_frame]` of one source video, and [`extract_frames.py`](extract_frames.py) writes
`videos/<category>/<clip>/<index>.jpg`, where `index` counts from the clip's first frame, so file `index` is the
frame that entry `index` of the annotation arrays describes. Download the source datasets under their own terms and
run, from the repository root:

```bash
python -m benchmarks.molmo2_videotrack.extract_frames --root /path/to/Molmo2-VideoTrackEval \
    --aptv2 /path/to/APTv2/videos --dancetrack /path/to/dancetrack/val --sportsmot /path/to/sportsmot/val \
    --sav /path/to/sav_test/JPEGImages_24fps --personpath22 /path/to/personpath22/videos
```

| Category | Source | Frame `j = start_frame + index` of the source video |
| --- | --- | --- |
| animal | APTv2 | `<video>/<j, 4 digits>.jpg` |
| dance | DanceTrack, validation split | `<video>/img1/<j + 1, 8 digits>.jpg` |
| sports | SportsMOT, validation split | `<video>/img1/<j + 1, 6 digits>.jpg` |
| misc | SA-V, test split | `<video>/<4 * j, 5 digits>.jpg`: the annotations are at 6 fps and SA-V's frames at 24 fps |
| person | PersonPath22, test split | frame `j`, counted from 0, of `<video>.mp4` as `ffmpeg` decodes it |

PersonPath22 needs `ffmpeg` on `PATH`, and the videos as PersonPath22's own download script writes them, including
the ones it fetches and re-encodes from MEVA, VIRAT and PathTrack. A category whose folder is not given is skipped,
and a complete clip folder is kept, so the script can be run once per source and repeated after an interruption.

The frames of the reported run were built with these mappings. `prepare.py` refuses a clip folder whose number of
frames differs from the annotation's `n_frames`: such a folder was not trimmed to the clip, or was extracted at
another frame rate, and its images would be scored against the annotations of other frames.

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

Each row has a top-level `task_id`, `<category>:<annotation id>` (for example `misc:sav_022374_2`), which does not
depend on row order. `gym eval plan` and task selection use it.

### How frames reach the model

Many expressions share a clip, so embedding frames in every row would produce about 7 GB of JSONL. By default each
frame is referenced by a `file://` URL instead (44 MB of JSONL). This needs a model server that reads local files,
for example vLLM started with `--allowed-local-media-path <frames dir>`.

| Variable | Effect |
| --- | --- |
| `MOLMO2_VIDEOTRACK_ROOT` | Required. Dataset root holding `data/` and `videos/` |
| `MOLMO2_VIDEOTRACK_FRAME_URL_ROOT` | Path at which the model server sees `<root>/videos`, when it differs |
| `MOLMO2_VIDEOTRACK_EMBED_FRAMES=1` | Embed frames as data URLs, for endpoints that cannot read local files |

Agent harnesses that accept images only as base64 data URLs, such as `opencode_agent` and `pi_agent`, reject the
default `file://` rows with HTTP 422; prepare the rows for them with `MOLMO2_VIDEOTRACK_EMBED_FRAMES=1`.

## Scoring

Per frame, predicted points are matched one-to-one to the visible objects and a matched point is correct when it
lies inside that object's mask; this gives a precision, recall and F1 per frame. Each task's score is the average
over its frames. See the [scorer README](../../resources_servers/molmo2_videotrack/README.md) for details.

| Metric | Meaning |
| --- | --- |
| `f1` | Mean task F1 over all expressions (headline; equals `mean/reward`) |
| `<category>/f1` | Mean task F1 within `animal`, `dance`, `misc`, `person` or `sports` |
| `precision`, `recall` | Mean task precision and recall |
| `parse_fail_rate` | Share of tasks whose answer yielded no usable point at all, in any frame |

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
