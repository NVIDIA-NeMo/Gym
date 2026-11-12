# vstar_spatiotemporal resources server

Verifier for the [V-STaR spatio-temporal reasoning benchmark](../../benchmarks/vstar_spatiotemporal/README.md). It
scores one of five sub-tasks per request and folds the five tasks of each item into the V-STaR metrics. The "what"
sub-task needs an LLM judge; the others are deterministic.

## Request

Each row carries its ground truth in `verifier_metadata`:

```json
{
  "item_index": 12,
  "sub_task": "spatial",
  "domain": "Animals",
  "fps": 29.97,
  "frame_count": 1805,
  "width": 640,
  "height": 360,
  "question": "what walks in front of a man in white?",
  "answer": "a dog",
  "timestamps": [20.2, 23.4],
  "bboxes": [{"timestamp": 21, "xmin": 270, "ymin": 137, "xmax": 361, "ymax": 283}]
}
```

`sub_task` is one of `vqa`, `temporal`, `spatial`, `spatial_2`, `temporal_2`. `item_index` is shared by the five
tasks of one item. `timestamps` is the annotated time span in seconds and `bboxes` the annotated box, in video
pixels, at each whole second inside it.

## Answer formats

Only the assistant message is read, not reasoning items. The content of the first `<answer>...</answer>` is used;
without one, the whole message.

| Sub-task | Reading |
| --- | --- |
| `vqa` | The answer text, sent to the judge as is. An empty `<answer></answer>` leaves the item unanswered |
| `temporal`, `temporal_2` | `m:ss` times become seconds; the answer must then hold exactly two numbers, taken as start and end. A number directly followed by a letter (`2s`) is not read, so `From <t>2</t>s to <t>6</t>s` parses and `From 2s to 6s` does not |
| `spatial`, `spatial_2` | JSON with a `tracks` list, or a bare list, of `{"t": T, "box": [x1,y1,x2,y2]}` entries or `{"boxes": {T: [x1,y1,x2,y2]}}` entries |

For box answers, coordinates are on a 0-1000 grid and are scaled to video pixels; a box whose coordinates are all at
most 1 is read as fractions of the frame. Each time is rounded to the nearest whole second, and when several fall on
the same second the one closest to it is kept. JSON that fails to load is retried with missing brackets added, so an
answer cut off by the output limit keeps the entries it completed.

## Scoring

| Sub-task | Score | Reward |
| --- | --- | --- |
| `vqa` | Judge rating 0-3 of the answer against the reference answer | 1 if the rating is 2 or 3, else 0 |
| `temporal`, `temporal_2` | IoU of the answered span with the annotated span | the IoU |
| `spatial`, `spatial_2` | IoU with the annotated box at every annotated second; 0 where the answer has no box for that second | the mean IoU |

The judge receives the V-STaR rating instructions as a system message and
`Question: ... Groundtruth answer: ... Candidate answer: ... Your response: ` as the user message. Its reply must be
a single integer; any other reply is recorded as rating -1, which counts as 0. A model reply with no text is rated 0
without calling the judge.

### Aggregation

`compute_metrics` groups rollouts by `item_index`:

- An item whose VQA answer is empty contributes nothing to the temporal and spatial means or to the per-group
  metrics. It still counts in the denominator of the overall `VQA_acc` and of the combined-success rates.
- `AM_k = (VQA_acc + temporal mIoU_k + spatial mIoU_k) / 3` and
  `LGM_k = -(log(1 - VQA_acc) + log(1 - temporal mIoU_k) + log(1 - spatial mIoU_k)) / 3` for chain `k`; `mAM` and
  `mLGM` average the two chains. An infinite LGM (a component equal to 1) is left out of the reported metrics.
- `VQA&Temp_k`, `VQA&Spat_k`, `Temp&Spat_k` and `VQA&Temp&Spat_k` are the fractions of items that pass the named
  parts together: rating of at least 2, temporal IoU of at least 0.3, mean box IoU of at least 0.1.
- `length/...` and `domain/...` repeat the main metrics for each video-length group and domain, over the answered
  items of the group.
- With several repeats, each repeat is aggregated on its own and the metrics are averaged.

`get_key_metrics` returns `mAM`, `mLGM`, `VQA_acc`, the four mIoU values and `parsed_rate_mean`.

## Configuration

| Field | Meaning |
| --- | --- |
| `judge_model_server` | Model server that rates VQA answers. Without one, a VQA task with a non-empty answer fails |
| `judge_responses_create_params` | Sampling for the judge; default temperature 0 and 512 output tokens |

## Example data

`data/example.jsonl` holds the five tasks of one synthetic item (a rectangle moving over a dark background, embedded
as PNG images). Regenerate it with `python -m benchmarks.vstar_spatiotemporal.tests.synthetic`.

## Tests

```bash
gym env test --resources-server vstar_spatiotemporal
```

`tests/verifier_cases.jsonl` holds full-reward, zero-reward and malformed cases; its VQA cases are rated by exact
match with the reference answer instead of a model.
