# MM-Lifelong v1.1, agentic

The 1,023 MM-Lifelong v1.1 questions of `benchmarks/mm_lifelong_v1_1` (`month_val`, `week_test`, `day_test`), each
answered by stock OpenCode in a sandbox that holds the original videos of the question's split, with the frame tools
`watch_video` and `view_image` (`resources_servers/video_qa_rubric/configs/video_qa_rubric_agentic.yaml`; the tools and
the sandbox are described in `resources_servers/video_mcqa/README.md`). The dataset's grader scores each final answer
from 0 to 5 against the question's frozen rubric, as in the single-turn benchmark (`grading.yaml` there).

## Videos: one image per split, built by you

The videos are third-party recordings that may not be redistributed, and too large to upload for every episode:

| Domain | Files | Size | Largest file |
| --- | --- | --- | --- |
| day | 1 | 8.1 GiB | 8.1 GiB |
| week | 6,266 clips in 7 day folders | 95.7 GiB | 0.03 GiB |
| month | 23 | 85.3 GiB | 10.1 GiB |

So no video or image is published: you build one sandbox image per domain from your own copy of the videos, and each
episode's sandbox starts from its domain's image with nothing uploaded (video_mcqa's preloaded media).
`sandbox_image/build.py`:

1. lists the domain's files at the pinned revision with the SHA-256 and size the Hub keeps for each and checks your
   copy against them;
2. measures each file's playable duration as the dataset's agentic runner does and writes `<output>/<domain>/`:
   `catalog.json` (path, bytes and SHA-256 of every file; the image holds it) and `video_catalog.json` (the runner's
   catalog that the model reads: each file's path, duration, and the recording's place in time from the dataset's
   `src/codex_video_timeline.json` at revision `b8a87afff4e24e76f6333937fb7b1c77dd5036fc`);
3. builds the image from the video tools' image (`resources_servers/video_mcqa/sandbox_image` of this checkout): the
   videos under `/testbed/videos/<domain>/`, one read-only layer per file or day folder, and the catalog. The build
   reads a temporary folder in `<media_dir>` that holds only the catalog's files, as hard links to your copy (copies
   if `<media_dir>` is on another file system), so other files in your copy stay out of the image; the script lists
   them.

```bash
hf download MM-Lifelong/MM-Lifelong --repo-type dataset --revision efa197479b25fbcf4f7881f9218d703c57dfc9bb \
    --include 'videos/day/*' --local-dir <data_dir>
python benchmarks/mm_lifelong_v1_1_agentic/sandbox_image/build.py --domain day --data-dir <data_dir> \
    --base-image <video tools image> --tag <registry>/mm-lifelong-v1-1-day:<tag> --output-dir <media_dir>
```

and likewise for `week` and `month`. Push the images to a registry your sandbox provider pulls from, and pull them onto
its nodes before a run (the week and month images are about 100 GB). Set `MM_LIFELONG_V1_1_DAY_IMAGE`,
`MM_LIFELONG_V1_1_WEEK_IMAGE` and `MM_LIFELONG_V1_1_MONTH_IMAGE` to them.

At each seed the resources server starts the sandbox from the row's image, copies the domain's `video_catalog.json` to
`/testbed/video_catalog.json`, and checks that the image's tools are this checkout's, that its catalog has the row's
SHA-256 and that every file it lists is there with its size; an image that differs is refused.

## Rows

`prepare.py` takes the questions as the single-turn benchmark does (`mm_lifelong_v1_1_agentic_data_dir` receives the
annotation files; no frames are read) and `build.py`'s output (`mm_lifelong_v1_1_agentic_media_dir`). Each row pins its
domain's image catalog and model catalog by SHA-256 (`preloaded_media`) and asks with the dataset agentic runner's
description of the inputs:

- `VIDEO INPUTS`: the catalog's records, one JSON line per file, or for the week's 6,266 clips the clips per day folder
  and where the catalog is;
- `TIME COORDINATES`: the runner's note on physical calendar times and clip-local playback seconds;
- `QUESTION`: the question as it is.

The runner's other instructions (its rules, programs and answer format) belong to its own harness and are left out, as
is its renaming of the week clips, whose file names (wall-clock start times) the images keep.

## Running

```bash
gym eval run --benchmark mm_lifelong_v1_1_agentic --model-type vllm_model --split benchmark \
    --output results/mm_lifelong_v1_1_agentic.jsonl \
    ++mm_lifelong_v1_1_agentic_data_dir=<data_dir> ++mm_lifelong_v1_1_agentic_media_dir=<media_dir> \
    ++policy_base_url=<vLLM URL> ++policy_model_name=<served model>
```

with `OPENSANDBOX_DOMAIN`, `OPENSANDBOX_API_KEY` and the judge's key set. A question can take hours of video work:
OpenCode stops at 7,200 s, the episode at 9,000 s, and the sandbox lives 10,800 s. The first `watch_video` call on a long
recording reads the whole file to index it, so `video_tools.profile.decode_timeout_seconds` is 900 s here. Scores are
those of the single-turn benchmark's metrics (`subsets/<split>/rubric_score`, `mean_of_subsets/rubric_score`); an agent
run that fails or stops at a limit scores 0.
