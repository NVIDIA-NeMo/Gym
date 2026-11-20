# multi_object_tracking resources server

Deterministic verifier for prompts that track several objects at once through a sequence of images. It is the
multi-object counterpart of the [got10k server](../got10k/README.md): every object is scored with that server's own
`score_sequence`, and the reward is the mean over objects.

It is a verifier, not a benchmark: there is no `benchmarks/multi_object_tracking`. It scores any dataset that can
be written in the row format below, for example box tracks derived from video segmentation masks.

## Request

Each row carries its ground truth in `verifier_metadata`. The schema is `MultiObjectTrackingVerifierMetadata` in
[`task_data.py`](task_data.py), which is both the verify request model and the `TaskData` that dataset tooling
validates rows against:

```json
{
  "sequence": "clip_000001",
  "image_width": 1280,
  "image_height": 720,
  "objects": [
    {"id": 1, "targets": [{"frame": 1, "bbox": [106.0, 244.0, 586.0, 341.0], "visible": 1.0}]},
    {"id": 2, "targets": [{"frame": 1, "bbox": [900.0, 80.0, 120.0, 200.0], "visible": 0.0}]}
  ]
}
```

`id` is the number the prompt gives the object. Each target is the got10k server's `Got10kTarget`: `frame` is the
image number used in the prompt (1..N), `bbox` is in pixels as `[x, y, width, height]`, and the frame is scored for
that object only when `visible` is above 0. `sequence` and `split` are bookkeeping and optional.

## Answer format

The expected answer is

```
<answer>{"tracks":[{"box":1,"frame":1,"bbox":[x1,y1,x2,y2],"visible":true},{"box":2,"frame":1,"bbox":[x1,y1,x2,y2],"visible":false}]}</answer>
```

with `xyxy` coordinates on a 0-1000 grid and one entry per (object, frame) pair. Only the assistant message is read,
not reasoning items.

- The entries are read from the last `<answer>` block, or from the full text when there is none. The list may sit
  under `tracks`, `boxes`, `bboxes`, `predictions` or `answers`, or be a bare list.
- The object id is read from `box`, `id`, `object`, `object_id`, `track_id` or `track`, the first that holds an
  integer. The box is read from `bbox`, `bbox_2d`, `bounding_box` or `pred_bbox`, or from `box` when `box` holds a
  list.
- An entry without an id is used only when the row has a single object. A single-object row answered in the got10k
  format (`{"boxes":[{"frame":1,"bbox":[...]}]}`) therefore gets the same reward here as in the got10k server.
- An entry with `"visible": false` is not used as a box.
- The first usable entry for an (object, frame) pair wins. Entries for other objects or frames are ignored.
- An answer cut off mid-entry keeps its complete entries.

## Scoring

Each object's boxes are scored exactly as in the got10k server: the grid box is converted to a clipped pixel box and
each frame where the object is visible scores the IoU with the ground truth, or 0 when there is no usable box.
Frames where the object is not visible are not scored, so a box given there, or a `"visible": false` claim, neither
earns nor costs anything. Claiming `"visible": false` on a frame where the object is visible scores 0 for it.

The reward is the mean, over the objects that are visible in at least one requested frame, of the object's mean IoU
over its visible frames, in [0, 1]. Objects that are never visible are left out. A row where no object is ever
visible gets reward 0 and contributes nothing to the aggregate metrics.

`compute_metrics` pools frames across all objects and rows and reports the got10k server's metrics:
`visible_gt_AO`, `visible_gt_SR@0.5`, `visible_gt_SR@0.75`, `visible_gt_samples` and `parse_pass_ratio`. The verify
response also returns per-object scores and a `parse_status` of `structured`, `salvaged`, `empty` or `parse_failed`.

## Data

`data/example.jsonl` holds five synthetic rows (one to three rectangles moving over a plain background, some hidden
in some frames), generated with `python -m resources_servers.multi_object_tracking.tests.synthetic`. It contains no
dataset imagery.

## Tests

```bash
gym env test --resources-server multi_object_tracking
```

## Licensing

Code: Apache-2.0.
