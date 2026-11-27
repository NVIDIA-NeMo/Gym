# LVBench, agentic

LVBench ([THUDM/LVBench](https://huggingface.co/datasets/THUDM/LVBench), **not LongVideoBench**: 1,549 questions over
103 long videos) answered by stock OpenCode in a sandbox that holds the question's video, with the frame tools
`watch_video` and `view_image`. The environment is the agentic configuration of the `lvbench` resources server
(`resources_servers/lvbench/configs/lvbench_agentic.yaml`; the tools, the sandbox image and the lending are described
in `resources_servers/video_mcqa/README.md`); this benchmark adds the data and the answer judge. The single-turn
version is `benchmarks/lvbench`.

## Data

`prepare.py` reads the authors' annotation, `video_info.meta.jsonl` of the Hugging Face dataset
[`THUDM/LVBench`](https://huggingface.co/datasets/THUDM/LVBench) at revision `0caedb92`, which it downloads and checks
by SHA-256 (`LVBENCH_ANNOTATIONS` names a local copy instead). The source directory (`lvbench_agentic_source_dir`)
holds the videos and their checksums:

- `video/<key>.mp4`, the dataset's 103 videos named by their `key`;
- `FILES.sha256`, their SHA-256 as `sha256sum` writes it, with paths relative to the directory
  (`video/Cm73ma6Ibcs.mp4`).

`resources_servers/video_mcqa/prepare.py`'s `prepare_dataset` writes `data/lvbench_agentic_benchmark.jsonl` and the
frozen cohort `data/lvbench_agentic_cohort.json`. No video is read: each row binds its video by path and SHA-256, and
the resources server checks that hash in the sandbox before the episode starts. Set `lvbench_agentic_video_dir` when
this host reads the `video/` directory at another path.

| Field | Value |
| --- | --- |
| Task ID | `lvbench:<uid>` |
| Prompt | `Video: /testbed/videos/<key>.mp4`, then the benchmark's letter prompt (`prompt="lvbench"` in `resources_servers/video_mcqa/README.md`): the question, the `(A) ...` options, `Answer with the option's letter from the given choices directly.` |
| Answer policy | `llm_extraction_v1`: a judge model reads the chosen option from the final answer |
| Categories | `domain` (annotation `type`) and overlapping capabilities (`task_type`: entity recognition, event understanding, key information retrieval, temporal grounding, reasoning, summarization), as `video_mcqa` reports them under `LVBench_agentic` |

The dataset's terms are CC-BY-NC-SA-4.0 with research-only / non-commercial restrictions (see
`resources_servers/lvbench/README.md`).

## Judge

`judge_base_url`, `judge_api_key` and `judge_model_name` default to gpt-6.1-sol on the inference API at max reasoning
effort (`JUDGE_BASE_URL`, `JUDGE_API_KEY` or `INFERENCE_API_KEY`, and `JUDGE_MODEL` override them). The judge sees
the question text, its options and the final answer, never the video or the gold label. Its server listens on
127.0.0.1, out of the lent sandbox's reach.

## Running

```bash
gym eval run --benchmark lvbench_agentic --model-type vllm_model --split benchmark \
    --output results/lvbench_agentic.jsonl \
    ++lvbench_agentic_source_dir=<directory with FILES.sha256 and video/> \
    ++policy_base_url=<vLLM URL> ++policy_model_name=<served model>
```

with `VIDEO_MCQA_SANDBOX_IMAGE` set to an image built from this commit's `resources_servers/video_mcqa/sandbox_image`
(the seed refuses an image whose video tools differ from the checkout's), and `OPENSANDBOX_DOMAIN` and
`OPENSANDBOX_API_KEY` set. The policy model server's address must be reachable from the sandbox (`use_absolute_ip`,
as the agentic configuration sets). Several LVBench videos are over 1.5 GiB; when many concurrent seeds upload large
videos at once, raise the OpenSandbox request timeout if uploads time out (for example
`++sandbox.opensandbox.connection.request_timeout_s=900`, as described in `resources_servers/video_mcqa/README.md`).
