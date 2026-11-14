# got10k resources server

Deterministic verifier for the [GOT-10k tracking benchmark](../../benchmarks/got10k/README.md). It reads one box per
requested image from the model's answer and scores each by IoU against the ground truth.

## Request

Each row carries its ground truth in `verifier_metadata`:

```json
{
  "sequence": "GOT-10k_Val_000001",
  "image_width": 1280,
  "image_height": 720,
  "targets": [
    {"frame": 1, "source_frame": 11, "bbox": [106.0, 244.0, 586.0, 341.0], "visible": 7.0}
  ]
}
```

`frame` is the image number used in the prompt (1..N). `bbox` is in pixels as `[x, y, width, height]`. `visible` is
the GOT-10k `cover.label` value; a frame is scored only when it is above 0.

## Answer format

The expected answer is

```
<answer>{"boxes":[{"frame":1,"bbox":[x1,y1,x2,y2]},{"frame":2,"bbox":[x1,y1,x2,y2]}]}</answer>
```

with `xyxy` coordinates on a 0-1000 grid. Only the assistant message is read, not reasoning items. Parsing is
lenient so that a slightly malformed answer still gets credit for the frames it localized. In order, stopping at the
first step that yields a box:

1. Each `<answer>` block, then the full text, as JSON or a Python literal. Accepted shapes include a list under
   `boxes`, `bboxes`, `predictions` or `answers`; a bare list of entries or of boxes in frame order; an object keyed
   by frame number; entries using `box`, `bbox_2d` or `x1,y1,x2,y2` keys; and one object that repeats the `frame`
   and `bbox` keys for every frame.
2. `"frame": N ... "bbox": [...]` pairs matched by pattern, which recovers truncated JSON.
3. `frame N` or `image N` followed by a bracketed list.
4. A bare run of numbers: exactly four per frame, or five per frame when each group starts with its frame number.

Entries for frames that were not requested are ignored.

## Scoring

A recovered box is divided by 1000, clamped to [0, 1], converted to pixels, rounded to 3 decimals and clipped to the
image. A box with no positive area (for example swapped corners) is discarded. Each frame's score is the IoU with the
ground-truth box, or 0 when there is no usable box. Out-of-range coordinates are clamped, not rejected.

The reward is the mean IoU over the row's visible frames, in [0, 1]. A row with no visible frame gets reward 0 and
contributes nothing to the aggregate metrics.

`compute_metrics` pools frames across all rows: `visible_gt_AO` (mean IoU), `visible_gt_SR@0.5` and
`visible_gt_SR@0.75` (share of frames with IoU strictly above the threshold), `visible_gt_samples`, and
`parse_pass_ratio` (share of all requested frames with a usable box). The verify response also returns per-frame
scores and a `parse_status` of `structured`, `empty`, `ambiguous_numbers` or `parse_failed`.

## Data

`data/example.jsonl` holds five synthetic sequences (a rectangle moving over a plain background), generated with
`python -m benchmarks.got10k.tests.synthetic`. It contains no GOT-10k imagery.

## Tests

```bash
gym env test got10k
gym env test --resources-server got10k
```

## Licensing

Code: Apache-2.0. GOT-10k data: CC BY-NC-SA 4.0, not redistributed here.
