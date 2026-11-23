# MM-Lifelong v1.1

[MM-Lifelong](https://huggingface.co/datasets/MM-Lifelong/MM-Lifelong) asks open-ended questions about a recorded
history: one 23.6-hour gameplay stream (Day), 52 hours of egocentric video over seven days in 6,266 clips (Week) and
105.6 hours of livestreams over 51 days in 23 recordings (Month). This benchmark is the dataset's evaluation splits as
a question set of the `video_qa_rubric` resources server, single turn: the model answers from frames of the split's
video, and the dataset's own grader scores each answer from 0 to 5 against the question's frozen rubric. The agentic
version, with the original videos in a sandbox, is `benchmarks/mm_lifelong_v1_1_agentic`.

## Data

| | |
| --- | --- |
| Source | Hugging Face `MM-Lifelong/MM-Lifelong`, revision `efa197479b25fbcf4f7881f9218d703c57dfc9bb` |
| Splits | `month_val` (623), `week_test` (200), `day_test` (200): the 1,023 questions with frozen rubrics; `month_dev` has none |
| License | MIT for the annotations, grader and frame cache; the videos are third-party recordings whose copyrights remain with their owners |
| Use | Evaluation only: the dataset has no training split |

`prepare.py` reads the release's `SHA256SUMS` (its own SHA-256 is pinned), `manifest.json` and the three annotation
files from `mm_lifelong_v1_1_data_dir`, downloading any that are missing from the pinned revision, and checks each
against `SHA256SUMS`. Nothing of the dataset is committed here. The videos are not redistributed: the model sees each
split through the release's frame cache (`frames/<domain>/shard_*.lmdb`, JPEG frames at 2 fps on the split's
timeline, 110.5 GB), which you download yourself into the same directory:

```bash
hf download MM-Lifelong/MM-Lifelong --repo-type dataset --revision efa197479b25fbcf4f7881f9218d703c57dfc9bb \
    --include 'frames/*' --local-dir <data_dir>
gym eval prepare --benchmark mm_lifelong_v1_1 ++mm_lifelong_v1_1_data_dir=<data_dir>
```

Every shard the frames come from is checked against `SHA256SUMS` before it is read. The prepare step needs PyAV,
lmdb and Pillow (`prepare_dependencies`, installed by `gym eval prepare`).

## Task

- **Input.** Every question of a split asks about the split's whole recording, so a split has one video:
  `num_frames` (256) frames at the centres of equal bins over the timeline (frame `int((i + 0.5) * N / 256)` of the
  cache's `N`), written once as an H.264 MP4 at 1 fps (`<frames_dir>/<split>.mp4`; `<split>.json` lists the cache
  frame and timeline second of each frame). Frame times inside the MP4 are not timeline times. A row sends that MP4 as
  a `file://` URL under `mm_lifelong_v1_1_frames_url_dir` and the question as it is, with no system prompt; the
  reference answer, evidence intervals and rubric never reach the model. Rows take the splits in turn, so any prefix
  (`--limit`) covers all three.
- **Serving.** The model server must read the MP4s at `mm_lifelong_v1_1_frames_url_dir` and keep all 256 frames. With
  vLLM: `--allowed-local-media-path <frames_url_dir>` and `--media-io-kwargs '{"video": {"num_frames": 256}}'`, and a
  context window that holds 256 frames plus the answer. The week video's 256 frames of 1408 x 1408 decode to 1.52 GB;
  vLLM builds that cap the decoded media of one request (`VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES`, 715,827,880 bytes in
  some builds) refuse it with HTTP 400 unless the cap is raised. The answer budget and sampling are the run's
  (`gym eval run --max-output-tokens ... --temperature ... --top-p ...`).

## Scoring

`video_qa_rubric`'s `reference_levels` protocol with the dataset's grader (`grading.yaml`): the grader's system prompt
and user template from `v1.1/grading/prompts.json` (a unit test rebuilds that file from them byte for byte), the
question, the reference answer, the frozen rubric as `json.dumps(rubric, ensure_ascii=False)` and the final answer
(the text after any reasoning block). The judge returns `{"score": 0-5, "reason": ...}` and the reward is the score /
5. The judge is the dataset's grader model, `openai/openai/gpt-5.6-luna` on NVIDIA's inference API (`JUDGE_BASE_URL`,
`JUDGE_API_KEY` or `INFERENCE_API_KEY`, `JUDGE_MODEL` override it), with no sampling settings and one completion, as
the grader sends. A reply that is not exactly that JSON is asked for again, up to 3 attempts; a judge that fails all
of them masks the row (`judge_failed`), never scores it 0.

| Metric | Meaning |
| --- | --- |
| `subsets/<split>/rubric_score` | The split's mean score / 5; × 100 is the dataset's split score |
| `mean_of_subsets/rubric_score` | The mean of the three split scores (each split weighs the same) |
| `mean/rubric_score` | The mean over all questions; × 100 is the overall score of the dataset's `summarize_scores.py` |

Differences from the dataset's `grade_predictions.py`: an attempt without a final answer (only reasoning, an
unfinished reasoning block, or an empty message) scores 0 without a judge call, where the grader would ask the judge
about an empty prediction; and judge failures are masked rows for re-judging rather than a stopped run.

## Running

```bash
gym eval run --benchmark mm_lifelong_v1_1 --model-type vllm_model \
    ++mm_lifelong_v1_1_data_dir=<data_dir> ++mm_lifelong_v1_1_frames_url_dir=<where vLLM reads frames_256> \
    ++policy_base_url=<vLLM URL> ++policy_model_name=<served model> \
    --max-output-tokens 65536 --temperature 1.0 --top-p 0.95
```

`--limit 6` gives a smoke run of two questions per split. The dataset card's results come from the release's own
harness, which samples frames differently and wraps the question in its own prompt, so they are not this
benchmark's protocol.
