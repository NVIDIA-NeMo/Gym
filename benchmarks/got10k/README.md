# GOT-10k

Single-object tracking on the [GOT-10k](http://got-10k.aitestunion.com) validation split, posed as multi-image
localization. The model is shown the first frame of a sequence together with the target's bounding box, then later
frames of the same sequence, and returns one box per later frame.

- Integration profile: `custom-gym-verifier`
- Scorer: [`resources_servers/got10k`](../../resources_servers/got10k/README.md)
- Harness: `simple_agent`, single turn, no tools

## Provenance

| | |
| --- | --- |
| Source | GOT-10k (Huang, Zhao, Huang; TPAMI 2021), validation split |
| Tasks | 180 sequences, 2,002 scored-candidate frames, 1,998 of them with the target visible |
| Data license | CC BY-NC-SA 4.0 (non-commercial); the benchmark code is Apache-2.0 |
| Access | Registration required; the data is not downloaded automatically and is not redistributed here |

Every sequence listed in `val/list.txt` is used; none are filtered. `prepare.py` refuses a partial extraction: each
sequence must have one annotation and one `cover.label` entry per frame and a valid `anno_fps` in `meta_info.ini`.

## Task contract

One task is one sequence. The single user turn is built by `prepare.py` and stored in the row
(`prompt_source: prepared`):

1. `Image 0 is the reference image. The target object is at xyxy bbox [x1,y1,x2,y2] on a 0-1000 grid.` followed by
   the sequence's first frame.
2. `Image k:` followed by a frame, for each target frame `k = 1..N`.
3. An instruction to return `<answer>{"boxes":[{"frame":1,"bbox":[x1,y1,x2,y2]},...]}</answer>` with integer
   coordinates on a 0-1000 grid.

Target frames are sampled at 1 frame per second of video after the first frame, using each sequence's annotation
rate (10 Hz for every validation sequence, so every 10th frame). Sequences have 5 to 47 target frames. Frames are
embedded as data URLs with their original bytes: no resizing or re-encoding. The 0-1000 grid is per axis and
independent of pixel size, so an x of 500 is the horizontal middle of the image.

Ground-truth boxes and visibility labels are in `verifier_metadata` only. The reference box is the only annotation
the model sees.

## Scoring

Each predicted box is clamped to the grid, mapped to pixels and scored by IoU against the ground-truth box. A frame
with no usable prediction scores 0. Frames where GOT-10k marks the target as fully occluded (`cover.label` of 0) are
excluded. See the [scorer README](../../resources_servers/got10k/README.md) for the answer formats accepted.

| Metric | Meaning |
| --- | --- |
| `visible_gt_AO` | Mean IoU over all visible target frames (headline; GOT-10k average overlap) |
| `visible_gt_SR@0.5`, `visible_gt_SR@0.75` | Share of visible target frames with IoU above 0.5 / 0.75 |
| `parse_pass_ratio` | Share of target frames for which a usable box was recovered |
| `mean/reward` | Mean over sequences of per-sequence mean IoU |

All are in [0, 1], higher is better. The headline metrics weight every frame equally, so long sequences count for
more; `mean/reward` weights every sequence equally and is therefore a different number.

These are not the official GOT-10k toolkit numbers: the toolkit scores a box on every annotated frame of the test
split, while this benchmark scores 1 frame per second of the validation split.

## Run conditions

Scoring is deterministic and uses no judge. The numbers this benchmark was set up to reproduce were obtained with
temperature 0.6, top-p 0.95, a 16,384-token output budget, one repeat, and reasoning enabled; pass these through
`responses_create_params` when comparing against them. Sampling is stochastic, so expect run-to-run variation.

## Prepare and run

```bash
export GOT10K_DATASET_ROOT=/path/to/GOT-10k   # the directory containing val/
gym eval prepare --benchmark got10k
gym env validate got10k
gym eval run --benchmark got10k --model-type vllm_model --split benchmark --limit 5 --num-repeats 1 \
    --output results/got10k_smoke.jsonl
```

The prepared JSONL is about 170 MB because it embeds 2,182 frames.

## Limitations

- Prompts carry up to 48 images; the model server must accept that many images per request and enough context.
- Rollout files repeat the embedded frames, so each repeat adds roughly the size of the prepared JSONL.
