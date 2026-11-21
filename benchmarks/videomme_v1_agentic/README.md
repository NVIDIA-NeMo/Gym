# Video-MME, agentic

Video-MME (2,700 questions over 900 videos, without subtitles) answered by stock OpenCode in a sandbox that holds the
question's video, with the frame tools `watch_video` and `view_image`. The environment is the agentic configuration
of the `videomme_v1` resources server (`resources_servers/videomme_v1/configs/videomme_v1_agentic.yaml`; the tools,
the sandbox image and the lending are described in `resources_servers/video_mcqa/README.md`); this benchmark adds the
data and the answer judge.

## Data

`prepare.py` reads the authors' annotation, `videomme/test-00000-of-00001.parquet` of the Hugging Face dataset
[`lmms-lab/Video-MME`](https://huggingface.co/datasets/lmms-lab/Video-MME) at revision `ead1408f`, which it downloads
and checks by SHA-256 (`VIDEOMME_V1_PARQUET` names a local copy instead). The source directory
(`videomme_v1_agentic_source_dir`) holds the videos and their checksums:

- `video/<videoID>.mp4`, the dataset's 900 videos named by their `videoID`;
- `FILES.sha256`, their SHA-256 as `sha256sum` writes it, with paths relative to the directory
  (`video/fFjv93ACGo8.mp4`).

`resources_servers/video_mcqa/prepare.py`'s `prepare_dataset` writes `data/videomme_v1_agentic_benchmark.jsonl` and
the frozen cohort `data/videomme_v1_agentic_cohort.json`. No video is read: each row binds its video by path and
SHA-256, and the resources server checks that hash in the sandbox before the episode starts. Set
`videomme_v1_agentic_video_dir` when this host reads the `video/` directory at another path.

| Field | Value |
| --- | --- |
| Task ID | `videomme_v1:<question_id>` |
| Prompt | `Video: /testbed/videos/<videoID>.mp4`, then the benchmark's letter prompt (`prompt="videomme_v1"` in `resources_servers/video_mcqa/README.md`): `Select the best answer ...`, the question, the `A. ...` options, `The best answer is:` |
| Answer policy | `llm_extraction_v1`: a judge model reads the chosen option from the final answer |
| Categories | duration, domain, sub-category and task type, as `video_mcqa` reports them |

The dataset's terms restrict its use to academic research (see `resources_servers/videomme_v1/README.md`).

## Judge

`judge_base_url`, `judge_api_key` and `judge_model_name` default to gpt-6.1-sol on the inference API at max reasoning
effort (`JUDGE_BASE_URL`, `JUDGE_API_KEY` or `INFERENCE_API_KEY`, and `JUDGE_MODEL` override them). The judge sees
the prompt text and the final answer, never the video or the gold label. Its server listens on 127.0.0.1, out of the
lent sandbox's reach.

## Running

```bash
gym eval run --benchmark videomme_v1_agentic \
    ++videomme_v1_agentic_source_dir=<directory with FILES.sha256 and video/> \
    ++policy_base_url=<vLLM URL> ++policy_model_name=<served model>
```

with `VIDEO_MCQA_SANDBOX_IMAGE` set to an image built from this commit's `resources_servers/video_mcqa/sandbox_image`
(the seed refuses an image whose video tools differ from the checkout's), and `OPENSANDBOX_DOMAIN` and
`OPENSANDBOX_API_KEY` set. The policy model server's address must be reachable from the sandbox (`use_absolute_ip`,
as the agentic configuration sets).
