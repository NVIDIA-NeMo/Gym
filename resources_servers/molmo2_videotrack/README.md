# molmo2_videotrack resources server

Deterministic verifier for the [Molmo2-VideoTrackEval benchmark](../../benchmarks/molmo2_videotrack/README.md). It
reads the points the model reported for each frame and checks them against per-object segmentation masks.

## Request

Each row carries its ground truth in `verifier_metadata`:

```json
{
  "category": "animal",
  "image_width": 1920,
  "image_height": 1080,
  "frames": [
    {"frame": 1, "source_frame": 0, "objects": [{"xy": [542.0, 667.0], "mask": {"size": [1080, 1920], "counts": "..."}}]},
    {"frame": 2, "source_frame": 6, "objects": []}
  ]
}
```

`frame` is the frame number used in the prompt (1..N). `objects` lists the matching objects visible in that frame:
`xy` is a point on the object in pixels and `mask` is its COCO RLE segmentation mask (compressed string or a list of
run lengths). A frame in which no matching object is visible has an empty list.

## Answer format

The expected answer is

```
<answer>{"tracks":[{"id":1,"frame":1,"xy":[x,y]},{"id":2,"frame":1,"xy":[x,y]}]}</answer>
```

with coordinates on a 0-1000 grid, and no entry for an object in a frame where it is not visible. Only the assistant
message is read, not reasoning items; anything before a `</think>` tag is dropped. Each `<answer>` block is tried in
turn, then the full text, and the first that yields a point is used. A block is read as JSON, or as the first JSON
object inside it, and may be `{"tracks": [...]}`, `{"points": [...]}` or a bare list. `point` is accepted in place
of `xy`. Truncated JSON is not repaired.

If no JSON yields a point, Molmo2's own markup is read: `<tracks coords="1:1 1 502 347;...">`,
`<tracks coords="1: [502,347];...">` and `<points coords="1 1 502 347,...">`.

## Scoring

For each frame, predicted points are scaled to pixels and matched one-to-one to the visible objects' points so that
the total distance is smallest. A matched point is correct when the pixel nearest to it is inside that object's
mask. Then

- precision = correct / predicted points, recall = correct / visible objects, F1 = their harmonic mean;
- a frame with predictions but no visible object, or visible objects but no prediction, scores 0;
- a frame with neither is skipped.

The reward is the F1 averaged over the row's scored frames, in [0, 1]. A row with no scored frame gets 1.0: there
was nothing to find and the model reported nothing. Ids are ignored, and predictions for frame numbers that were not
shown are ignored.

`compute_metrics` reports the mean `f1`, `precision` and `recall` over rows, the same per category as
`<category>/f1` and so on with `<category>/samples`, and `parse_fail_rate`. The verify response also returns
per-frame counts and scores.

## Data

`data/example.jsonl` holds five synthetic clips (rectangles moving over a plain background) with embedded frames,
generated with `python -m benchmarks.molmo2_videotrack.tests.synthetic`. It contains no imagery from the source
video datasets.

## Tests

```bash
gym env test molmo2_videotrack
gym env test --resources-server molmo2_videotrack
```

## Licensing

Code: Apache-2.0. Molmo2-VideoTrackEval annotations: ODC-BY-1.0. Source videos: under their own datasets' terms, not
redistributed here.
